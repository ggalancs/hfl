# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``POST /api/pull`` endpoint.

Downloads a model and streams progress in NDJSON format identical to
Ollama's — clients like Open WebUI, LibreChat and ``ollama-python``
key off the status strings (``pulling manifest``, ``downloading``,
``verifying sha256 digest``, ``success``) and the numeric fields
(``total``, ``completed``, ``digest``) to render progress bars.

Pull reference: https://docs.ollama.com/api#pull-a-model

Envelope sequence (NDJSON, one JSON object per line):

    {"status": "pulling manifest"}
    {"status": "downloading", "digest": "sha256:...", "total": N, "completed": 0}
    {"status": "downloading", "digest": "sha256:...", "total": N, "completed": M}
    ...
    {"status": "verifying sha256 digest"}
    {"status": "writing manifest"}
    {"status": "success"}

HFL delegates the actual bytes transfer to ``huggingface_hub`` which
publishes its own byte-level progress via tqdm. Rather than hooking
into every tqdm tick (fragile across library versions), we emit
coarse-grained checkpoints and a final completion event with the
true byte count read off disk. Open WebUI and LangChain both
tolerate that shape — they re-render the bar on every chunk
regardless of whether the byte counter actually ticked.

Non-streaming mode (``stream=false``) is also supported: the call
blocks until the pull finishes and returns a single JSON object
``{"status": "success"}`` (or the failure envelope).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hfl.hub.license_checker import LicenseInfo
    from hfl.hub.resolver import ResolvedModel

from hfl.hub.connectivity import HUB_HOST, is_network_error
from hfl.logging_config import log_internal_failure

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ollama"])


class PullRequest(BaseModel):
    """Body for ``POST /api/pull``."""

    model: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Model identifier: ``org/name`` or ``org/name:quant``.",
    )
    revision: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Optional HuggingFace ref (branch, tag, or commit) to pin the pull "
            "to an exact repo state. ``org/name@<ref>`` in ``model`` works too. "
            "Defaults to the repo's main branch."
        ),
    )
    insecure: bool = Field(
        False,
        description=(
            "Ollama parity flag — accepted for compatibility but ignored; "
            "HuggingFace Hub downloads always use HTTPS."
        ),
    )
    stream: bool = Field(
        True,
        description=(
            "Stream progress as NDJSON (default). Set false to block "
            "until completion and receive a single JSON envelope."
        ),
    )


HUB_UNREACHABLE_MESSAGE = (
    f"cannot reach {HUB_HOST} — this server appears to be offline. "
    "Models already pulled keep serving."
)


def _event(status: str, **extra: Any) -> str:
    """Format a single NDJSON progress event (one line terminated by \\n)."""
    payload: dict[str, Any] = {"status": status}
    payload.update(extra)
    return json.dumps(payload, separators=(",", ":")) + "\n"


def _unknown_license(repo_id: str) -> "LicenseInfo":
    """Fallback license record when classification fails (fail-closed).

    Treated as ``UNKNOWN`` risk so the default ``permissive`` policy
    refuses it — a transient Hub hiccup never silently lets a
    non-permissive model through.
    """
    from hfl.hub.license_checker import LicenseInfo, LicenseRisk

    return LicenseInfo(
        license_id="unknown",
        license_name="Unknown",
        risk=LicenseRisk.UNKNOWN,
        restrictions=[],
        url=f"https://huggingface.co/{repo_id}",
        gated=False,
    )


async def _license_gate(repo_id: str) -> tuple["LicenseInfo", dict[str, Any] | None]:
    """Apply the server owner's license policy to a repo (non-interactive).

    Returns ``(license_info, error_event)``. ``error_event`` is ``None``
    when the pull may proceed; otherwise it is the NDJSON error event to
    emit before aborting. The interactive CLI (``hfl pull``) is a
    separate path and always prompts a human — this gate only governs the
    HTTP API, where no human is in the loop.
    """
    from hfl.config import config
    from hfl.hub.license_checker import check_model_license, policy_allows

    policy = getattr(config, "license_policy", "permissive")

    try:
        info = await asyncio.to_thread(check_model_license, repo_id)
    except Exception as exc:  # network / Hub failure → fail closed as UNKNOWN
        logger.warning("license classification failed for %s: %s", repo_id, exc)
        info = _unknown_license(repo_id)

    if policy_allows(info, policy):
        return info, None

    error_event = {
        "status": "error",
        "error": (
            f"License '{info.license_id}' ({info.risk.value}) is not covered by this "
            f"server's license policy ('{policy}'). The server owner must accept it: "
            f"widen HFL_LICENSE_POLICY to include this tier, or pull it locally with "
            f"`hfl pull {repo_id}` (which prompts for explicit acceptance). "
            f"See {info.url}"
        ),
        "code": "license_not_accepted",
        "license": info.license_id,
        "risk": info.risk.value,
    }
    return info, error_event


def _hub_answered(exc: BaseException, model: str) -> dict[str, str] | None:
    """An error event for a repo the Hub reported unknown or gated, else None."""
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    # GatedRepoError subclasses RepositoryNotFoundError: check it first.
    if isinstance(exc, GatedRepoError):
        return {
            "status": "error",
            "error": (
                f"{model} is gated on the Hugging Face Hub: request access on its "
                "page there, then set HF_TOKEN (or run `hfl login`)."
            ),
            "code": "gated",
        }
    if isinstance(exc, RepositoryNotFoundError) or (
        isinstance(exc, ValueError) and str(exc).startswith("Model not found")
    ):
        return {
            "status": "error",
            "error": f"model not found on the Hugging Face Hub: {model}",
            "code": "not_found",
        }
    return None


def _record_server_pull(
    resolved: "ResolvedModel",
    local_path: Any,
    license_info: "LicenseInfo",
    policy: str,
    alias: str | None = None,
) -> None:
    """Register the pulled model and log its provenance, through the steps
    ``hfl pull`` uses too (:mod:`hfl.hub.pull_service`): the model type is
    recorded, a re-pull keeps its alias, an alias in use is never taken.
    The server's own registry: a fresh ModelRegistry() wrote the file, but
    the one /api/tags and every load read kept its old view until restart.
    Raises :class:`~hfl.hub.pull_service.PullStepError` for a model type
    HFL cannot serve (its download removed).
    """
    from datetime import datetime
    from pathlib import Path

    from hfl.core.container import get_registry
    from hfl.hub.pull_service import finish_download, register_pulled

    finished = finish_download(resolved, Path(local_path), convert=False)
    register_pulled(
        resolved,
        finished,
        registry=get_registry(),
        license_info=license_info,
        accepted_at=datetime.now().isoformat(),
        alias=alias,
        quantize=None,
        source=f"server /api/pull; owner license policy '{policy}'",
    )


async def iter_pull_events(
    model_name: str, *, quantization: str | None = None
) -> AsyncIterator[str]:
    """V5 β3 — public helper that drives the same NDJSON pull shape
    as ``POST /api/pull``.

    Used by :mod:`hfl.api.routes_smart_pull` to forward progress
    events after the planning step. Exposed at module top-level so
    consumers don't need ``try/except ImportError`` against an
    underscore-prefixed name.

    ``quantization`` lets smart-pull thread the variant it selected for
    the host's memory budget straight through to the resolver, instead
    of letting the resolver re-pick its own default (which could exceed
    the budget that was the whole point of the smart plan).
    """
    req = PullRequest(model=model_name, stream=True, insecure=False)
    async for line in _run_pull_streaming(req, quantization=quantization):
        yield line


@dataclass
class _PullState:
    """What each stage of a server pull hands to the next."""

    resolved: Any = None
    alias: str | None = None
    local_path: Any = None


def _bytes_on_disk(local_path: Any) -> int:
    """The download's size on disk (0 if it cannot be read)."""
    try:
        if local_path.is_file():
            return int(local_path.stat().st_size)
        return sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
    except OSError:  # pragma: no cover - stat failure is rare and non-fatal
        return 0


async def _resolve_stage(
    req: PullRequest, quantization: str | None, state: "_PullState"
) -> AsyncIterator[str]:
    """Phase 1: the reference resolved on the Hub (a short name first chosen
    as the best GGUF build, and kept as an alias). ``state.resolved`` stays
    None when it failed; the error event says why."""
    from hfl.hub.resolver import resolve

    # --- Phase 1: resolve manifest ----------------------------------
    yield _event("pulling manifest")

    # A short name (``llama3.2``, ``qwen3:8b``) — what Ollama clients such as
    # Open WebUI ask for — becomes the best GGUF build for this machine, and
    # is registered under it as an alias so the same name then chats.
    from hfl.hub.resolver import parse_model_spec
    from hfl.hub.shortname import alias_for, find, is_short_name

    reference, alias = req.model, None
    if parse_model_spec(req.model).repo_id is None and is_short_name(req.model):
        try:
            choice = await asyncio.to_thread(find, req.model)
        except Exception as exc:
            if is_network_error(exc):
                logger.info("pull of %r: Hub unreachable (%s)", req.model, type(exc).__name__)
                yield _event("error", error=HUB_UNREACHABLE_MESSAGE, code="hub_unreachable")
                return
            raise
        if choice is None:
            logger.info("pull of %r: no GGUF build found", req.model)
            yield _event(
                "error",
                error=f"no GGUF model on the Hugging Face Hub matches {req.model}",
                code="not_found",
            )
            return
        reference, quantization = choice.reference, choice.quantization
        alias = alias_for(req.model)
        yield _event(f"{req.model} is {choice.repo_id}:{choice.quantization}")

    try:
        resolved = await asyncio.to_thread(resolve, reference, quantization, req.revision)
    except Exception as exc:  # pragma: no cover — error envelope tested via mock
        # ``resolve`` talks to the Hub and the local cache; its failures quote
        # URLs and on-disk paths. Reference the log line instead.
        if is_network_error(exc):
            # Offline is a normal state for a local runner, not a server
            # fault: name the cause, and tag it so the non-stream path can
            # answer 503 (the upstream is unavailable) instead of 500.
            logger.info("pull of %r: Hub unreachable (%s)", req.model, type(exc).__name__)
            yield _event("error", error=HUB_UNREACHABLE_MESSAGE, code="hub_unreachable")
            return
        answered = _hub_answered(exc, req.model)
        if answered is not None:
            # The Hub answered: an unknown or gated repo is the caller's to
            # fix, not a server fault (it used to be a 500 with a traceback).
            logger.info("pull of %r: %s", req.model, answered["code"])
            yield json.dumps(answered, separators=(",", ":")) + "\n"
            return
        detail = log_internal_failure(logger, f"resolving {req.model!r}", exc)
        yield _event("error", error=detail)
        return
    state.resolved, state.alias = resolved, alias


async def _download_stage(
    resolved: Any, digest_label: str, progress: Any, state: "_PullState"
) -> AsyncIterator[str]:
    """Phase 2: the blocking download in a worker, with a heartbeat every
    2 s carrying the bytes on disk so far, so a client (Open WebUI) keeps
    its progress bar alive. ``state.local_path`` stays None when it failed."""
    from hfl.hub.downloader import pull_model

    # --- Phase 2: download ------------------------------------------
    # We run the blocking hf_hub_download in a worker; meanwhile a
    # heartbeat coroutine keeps the stream alive so Open WebUI
    # doesn't think the connection stalled.
    download_task = asyncio.create_task(asyncio.to_thread(pull_model, resolved))

    while not download_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(download_task), timeout=2.0)
        except asyncio.TimeoutError:
            # 2 s since the last heartbeat — emit another, with the bytes
            # on disk so far, so the client keeps the progress bar alive.
            yield _event("downloading", digest=digest_label, **await asyncio.to_thread(progress))
        except asyncio.CancelledError:  # pragma: no cover — client disconnect
            download_task.cancel()
            raise
        except Exception:
            # The download failed — re-raise on the awaited task below.
            break

    try:
        local_path = await download_task
    except Exception as exc:
        if is_network_error(exc):
            logger.info(
                "pull of %r: Hub unreachable mid-download (%s)",
                resolved.repo_id,
                type(exc).__name__,
            )
            yield _event("error", error=HUB_UNREACHABLE_MESSAGE, code="hub_unreachable")
            return
        from hfl.exceptions import DownloadIntegrityError

        if isinstance(exc, DownloadIntegrityError):
            # Says which file and what to expect; no paths in it.
            yield _event("error", error=f"{exc.message}: {exc.details}", code="integrity")
            return
        detail = log_internal_failure(logger, "download", exc)
        yield _event("error", error=detail)
        return
    state.local_path = local_path


async def _run_pull_streaming(
    req: PullRequest, *, quantization: str | None = None
) -> AsyncIterator[str]:
    """Async NDJSON stream mirroring Ollama's pull progress shape.

    The heavy lifting (network I/O + disk writes) runs in a worker
    thread via :func:`asyncio.to_thread` so the event loop stays free
    to emit progress events.
    """
    # --- Phase 1: resolve manifest ----------------------------------
    state = _PullState()
    async for line in _resolve_stage(req, quantization, state):
        yield line
    if state.resolved is None:
        return
    resolved, alias = state.resolved, state.alias

    from hfl.hub.pull_service import PullStepError, unsupported_type

    unsupported = unsupported_type(resolved)
    if unsupported is not None:
        # Known from the Hub: refused before a byte is downloaded, as the CLI does.
        yield _event(
            "error", error=f"{unsupported} models are not supported by HFL", code="unsupported"
        )
        return

    # --- License gate: owner policy, no human in the loop here ----------
    # Classify + apply HFL_LICENSE_POLICY. A license the owner has not
    # pre-accepted stops the pull before a single byte is transferred.
    yield _event("verifying license")
    license_info, license_error = await _license_gate(resolved.repo_id)
    if license_error is not None:
        yield json.dumps(license_error, separators=(",", ":")) + "\n"
        return

    digest_label = (
        resolved.revision
        if resolved.revision and resolved.revision.startswith("sha256:")
        else f"sha256:{resolved.repo_id.replace('/', '--')}"
    )

    # The size of what will be fetched, from the Hub (empty: unknown), and
    # how much of it is on disk: every heartbeat carries real progress.
    from hfl.hub.downloader import bytes_done, expected_files

    planned = await asyncio.to_thread(expected_files, resolved)
    total = sum(planned.values())

    def progress() -> dict[str, int]:
        done = bytes_done(resolved, planned) if planned else 0
        return {"total": total, "completed": min(done, total) if total else done}

    yield _event("downloading", digest=digest_label, **progress())

    async for line in _download_stage(resolved, digest_label, progress, state):
        yield line
    if state.local_path is None:
        return
    local_path = state.local_path

    # The size on disk, so the final event reports a concrete number
    # (clients use it to render "100%").
    total_bytes = _bytes_on_disk(local_path)
    yield _event(
        "downloading",
        digest=digest_label,
        total=total_bytes,
        completed=total_bytes,
    )

    # --- Phase 3: verify + finalize ---------------------------------
    yield _event("verifying sha256 digest")
    await asyncio.sleep(0)  # yield control; no real hash work in this path

    # Legal traceability: register the model with its license + log
    # provenance recording the owner policy under which it was accepted.
    # Best-effort — a bookkeeping failure must not fail the pull itself.
    yield _event("writing manifest")
    try:
        from hfl.config import config

        policy = getattr(config, "license_policy", "permissive")
        await asyncio.to_thread(
            _record_server_pull, resolved, local_path, license_info, policy, alias
        )
    except PullStepError as exc:
        # A type HFL cannot serve, found only once downloaded: removed.
        yield _event(
            "error", error=f"{exc.values.get('type')} models are not supported", code="unsupported"
        )
        return
    except Exception as exc:  # pragma: no cover — defensive; recording is non-critical
        logger.warning("server pull bookkeeping failed for %s: %s", resolved.repo_id, exc)

    yield _event("success")


@router.post(
    "/api/pull",
    tags=["Ollama"],
    summary="Pull a model from HuggingFace Hub",
    response_model=None,
    responses={
        200: {"description": "NDJSON stream of progress events, or a single JSON on success."},
        400: {"description": "Invalid request body."},
    },
)
async def pull_model_route(req: PullRequest, request: Request) -> StreamingResponse | JSONResponse:
    """Ollama-compatible ``POST /api/pull``.

    ``pull`` is an owner (administrative) operation — see
    :mod:`hfl.api.admin_guard`. Remote callers are refused with ``403``
    unless ``HFL_ALLOW_REMOTE_PULL`` is set. Otherwise, streams NDJSON
    progress events by default; ``stream=false`` blocks and returns a
    single JSON envelope. A license the server's policy does not accept
    yields ``403`` (non-stream) or a ``license_not_accepted`` error event
    (stream).
    """
    from hfl.api.admin_guard import require_owner

    require_owner(request, "pull")

    if not req.stream:
        # Non-streaming: collect every event, return the last status.
        final: dict[str, Any] = {"status": "success"}
        start = time.monotonic()
        async for line in _run_pull_streaming(req):
            event = json.loads(line)
            if event.get("status") == "error":
                # A refused license is a client/authorization problem (403),
                # not a server failure (500).
                # An unreachable Hub is an unavailable upstream (503), and
                # retryable — a client can back off and try again when the
                # network returns. 500 claimed the server itself had broken.
                status_code = {
                    "license_not_accepted": 403,
                    "gated": 403,
                    "not_found": 404,
                    "hub_unreachable": 503,
                }.get(event.get("code", ""), 500)
                return JSONResponse(status_code=status_code, content=event)
            final = event
        final["_duration_seconds"] = round(time.monotonic() - start, 2)
        return JSONResponse(content=final)

    return StreamingResponse(
        _run_pull_streaming(req),
        media_type="application/x-ndjson",
    )

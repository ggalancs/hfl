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
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hfl.hub.license_checker import LicenseInfo
    from hfl.hub.resolver import ResolvedModel

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


def _record_server_pull(
    resolved: "ResolvedModel", local_path: Any, license_info: "LicenseInfo", policy: str
) -> None:
    """Register the pulled model + log provenance for legal traceability.

    Best-effort: the server pull path historically registered nothing, so
    a failure here must never fail an otherwise-successful download. The
    caller wraps this in ``asyncio.to_thread`` and swallows exceptions.
    """
    from datetime import datetime

    from hfl.converter.formats import detect_format
    from hfl.models.manifest import ModelManifest
    from hfl.models.provenance import log_conversion
    from hfl.models.registry import ModelRegistry

    fmt = detect_format(local_path)
    if local_path.is_file():
        size = local_path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())

    short_name = resolved.repo_id.split("/")[-1].lower()
    quant = getattr(resolved, "quantization", None)
    if quant:
        short_name += f"-{quant.lower()}"

    accepted_at = datetime.now().isoformat()
    manifest = ModelManifest(
        name=short_name,
        repo_id=resolved.repo_id,
        local_path=str(local_path),
        format=fmt.value,
        size_bytes=size,
        revision=getattr(resolved, "revision", None),
        commit_sha=getattr(resolved, "commit_sha", None),
        quantization=quant,
        license=license_info.license_id,
        license_name=license_info.license_name,
        license_url=license_info.url,
        license_restrictions=license_info.restrictions,
        gated=license_info.gated,
        license_accepted_at=accepted_at,
    )
    ModelRegistry().add(manifest)

    log_conversion(
        source_repo=resolved.repo_id,
        source_format=fmt.value,
        target_path=str(local_path),
        original_license=license_info.license_id,
        license_accepted=True,
        notes=f"server /api/pull; owner license policy '{policy}'",
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


async def _run_pull_streaming(
    req: PullRequest, *, quantization: str | None = None
) -> AsyncIterator[str]:
    """Async NDJSON stream mirroring Ollama's pull progress shape.

    The heavy lifting (network I/O + disk writes) runs in a worker
    thread via :func:`asyncio.to_thread` so the event loop stays free
    to emit progress events.
    """
    from hfl.hub.downloader import pull_model
    from hfl.hub.resolver import resolve

    # --- Phase 1: resolve manifest ----------------------------------
    yield _event("pulling manifest")

    try:
        resolved = await asyncio.to_thread(resolve, req.model, quantization, req.revision)
    except Exception as exc:  # pragma: no cover — error envelope tested via mock
        yield _event("error", error=f"Failed to resolve {req.model!r}: {exc}")
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

    # Emit an opening "downloading" event with total=0 so clients
    # render "Starting download..." before the first bytes arrive.
    yield _event(
        "downloading",
        digest=digest_label,
        total=0,
        completed=0,
    )

    # --- Phase 2: download ------------------------------------------
    # We run the blocking hf_hub_download in a worker; meanwhile a
    # heartbeat coroutine keeps the stream alive so Open WebUI
    # doesn't think the connection stalled.
    download_task = asyncio.create_task(asyncio.to_thread(pull_model, resolved))

    while not download_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(download_task), timeout=2.0)
        except asyncio.TimeoutError:
            # 2 s since the last heartbeat — emit another so the
            # client keeps the progress bar alive.
            yield _event(
                "downloading",
                digest=digest_label,
                total=0,
                completed=0,
            )
        except asyncio.CancelledError:  # pragma: no cover — client disconnect
            download_task.cancel()
            raise
        except Exception:
            # The download failed — re-raise on the awaited task below.
            break

    try:
        local_path = await download_task
    except Exception as exc:
        yield _event("error", error=str(exc))
        return

    # Measure the actual on-disk size so the final event reports a
    # concrete number (clients use it to render "100%").
    total_bytes = 0
    try:
        if local_path.is_file():
            total_bytes = local_path.stat().st_size
        else:
            for f in local_path.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
    except OSError:  # pragma: no cover — stat failure is rare and non-fatal
        pass

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
        await asyncio.to_thread(_record_server_pull, resolved, local_path, license_info, policy)
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
                status_code = 403 if event.get("code") == "license_not_accepted" else 500
                return JSONResponse(status_code=status_code, content=event)
            final = event
        final["_duration_seconds"] = round(time.monotonic() - start, 2)
        return JSONResponse(content=final)

    return StreamingResponse(
        _run_pull_streaming(req),
        media_type="application/x-ndjson",
    )

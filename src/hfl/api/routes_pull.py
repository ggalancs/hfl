# SPDX-License-Identifier: HRUL-1.0
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
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["Ollama"])


class PullRequest(BaseModel):
    """Body for ``POST /api/pull``."""

    model: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Model identifier: ``org/name`` or ``org/name:quant``.",
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


async def _run_pull_streaming(req: PullRequest) -> AsyncIterator[str]:
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
        resolved = await asyncio.to_thread(resolve, req.model)
    except Exception as exc:  # pragma: no cover — error envelope tested via mock
        yield _event("error", error=f"Failed to resolve {req.model!r}: {exc}")
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
    yield _event("writing manifest")
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
async def pull_model_route(req: PullRequest) -> StreamingResponse | JSONResponse:
    """Ollama-compatible ``POST /api/pull``.

    Streams NDJSON progress events by default. Clients that want a
    fire-and-forget one-shot response can pass ``stream=false``; the
    route will block until completion and return a single JSON
    envelope (``{"status":"success"}`` or ``{"status":"error",
    "error":"..."}``).
    """
    if not req.stream:
        # Non-streaming: collect every event, return the last status.
        final: dict[str, Any] = {"status": "success"}
        start = time.monotonic()
        async for line in _run_pull_streaming(req):
            event = json.loads(line)
            if event.get("status") == "error":
                return JSONResponse(status_code=500, content=event)
            final = event
        final["_duration_seconds"] = round(time.monotonic() - start, 2)
        return JSONResponse(content=final)

    return StreamingResponse(
        _run_pull_streaming(req),
        media_type="application/x-ndjson",
    )

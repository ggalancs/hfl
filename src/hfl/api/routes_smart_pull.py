# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 ``POST /api/pull/smart`` — best-variant pull for the current host.

Builds a :class:`hfl.hub.smart_pull.SmartPullPlan`, then delegates
the actual byte transfer to the existing ``/api/pull`` endpoint
shape so clients keep the same NDJSON parser. The response carries
the plan's reasoning as the first event so the operator sees what
the server picked and why before bytes start moving.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


class SmartPullRequest(BaseModel):
    """Request body for ``POST /api/pull/smart``.

    ``model`` is the *base* repo (e.g. ``meta-llama/Llama-3.1-8B-Instruct``);
    smart-pull will resolve to a community fork or quant variant when
    available.
    """

    model: str = Field(..., min_length=1, max_length=256)
    max_vram_gb: float | None = Field(default=None, gt=0, le=4096)
    stream: bool = Field(default=True)


async def _stream_smart_pull(req: SmartPullRequest) -> AsyncIterator[str]:
    """Plan + delegate pull, emitting NDJSON status events.

    Event grammar:

    - ``{"status": "planning"}``
    - ``{"status": "planned", "target_repo_id": "...", ...}``
    - then forwards every chunk from the existing ``/api/pull``
      handler verbatim.
    """
    yield json.dumps({"status": "planning"}) + "\n"

    from hfl.hub.smart_pull import build_smart_plan

    try:
        plan = build_smart_plan(req.model, max_vram_gb=req.max_vram_gb)
    except ValueError as exc:
        yield json.dumps({"status": "failed", "error": str(exc)}) + "\n"
        return
    except Exception as exc:
        logger.exception("smart_pull planning failed")
        yield json.dumps({"status": "failed", "error": f"planning failed: {exc}"}) + "\n"
        return

    yield (
        json.dumps(
            {
                "status": "planned",
                "target_repo_id": plan.target_repo_id,
                "quantization": plan.quantization,
                "estimated_vram_gb": plan.estimated_vram_gb,
                "reason": plan.reason,
                "fallback_chain": plan.fallback_chain,
            }
        )
        + "\n"
    )

    # Delegate to the existing pull machinery. We can't easily call
    # the FastAPI endpoint internally without an HTTP round-trip, so
    # we re-use the handler's helper — ``_iter_pull_events`` — that
    # produces the same NDJSON shape /api/pull emits.
    try:
        from hfl.api.routes_pull import _iter_pull_events  # type: ignore[attr-defined]
    except ImportError:
        # Fallback: emit the plan and let the client invoke /api/pull
        # itself on the resolved repo.
        yield (
            json.dumps(
                {
                    "status": "delegated",
                    "next": {"endpoint": "/api/pull", "model": plan.target_repo_id},
                }
            )
            + "\n"
        )
        return

    async for chunk in _iter_pull_events(plan.target_repo_id):
        yield chunk


@router.post(
    "/api/pull/smart",
    response_model=None,
    summary="Smart-pull: best Hub variant for the current hardware",
    responses={
        200: {"description": "Smart pull NDJSON stream or final envelope"},
        400: {"description": "Malformed request"},
    },
)
async def api_pull_smart(req: SmartPullRequest) -> StreamingResponse | JSONResponse:
    """V4: pick the optimal Hub variant for this host and pull it.

    Body:

    ```
    { "model": "meta-llama/Llama-3.1-8B-Instruct", "max_vram_gb": 12.0 }
    ```

    On Apple Silicon with mlx-lm installed, this resolves to the
    matching ``mlx-community`` fork at MLX-4bit/8bit. On a 24 GB
    CUDA host without MLX it picks Q5_K_M from the
    ``bartowski/<name>-GGUF`` fork. On CPU-only it falls back to
    the smallest quant that fits.
    """
    if not req.stream:
        # Drain the stream into a single envelope (last event wins).
        last: dict[str, Any] = {"status": "planning"}
        async for line in _stream_smart_pull(req):
            try:
                last = json.loads(line.strip())
            except (json.JSONDecodeError, ValueError):
                continue
        if last.get("status") == "failed":
            raise HTTPException(status_code=400, detail=last.get("error", "unknown"))
        return JSONResponse(content=last)

    return StreamingResponse(
        _stream_smart_pull(req),
        media_type="application/x-ndjson",
    )

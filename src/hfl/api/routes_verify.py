# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``POST /api/verify/{model}`` — model sanity-check endpoint."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from hfl.api.model_loader import load_llm

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


@router.post(
    "/api/verify/{model:path}",
    response_model=None,
    summary="Sanity-check a registered model",
    responses={
        200: {"description": "VerifyResult"},
        403: {"description": "Remote caller (owner-only diagnostic)"},
        404: {"description": "Model not found"},
        503: {"description": "Engine unavailable"},
    },
)
async def api_verify(model: str, request: Request) -> dict[str, Any] | JSONResponse:
    """Run verification probes against ``model``.

    Output shape:

    ```
    {
        "model": "qwen-coder-7b",
        "overall_pass": true,
        "duration_ms": 142.5,
        "checks": [
          { "name": "tokenizer_round_trip", "passed": true, "detail": "..." },
          ...
        ]
    }
    ```
    """
    from hfl.api.admin_guard import require_owner
    from hfl.engine.verifier import verify_model

    # A diagnostic for whoever runs the server: it loads the model and
    # generates on it, so a remote user could keep it busy at will.
    require_owner(request, "verify")
    try:
        engine, manifest = await load_llm(model)
    except FileNotFoundError as exc:
        # ``load_llm`` raises ``ModelNotFoundError`` for an unregistered
        # model; a ``FileNotFoundError`` here means the engine could not
        # open the blob, and the OS message spells out its path. Name the
        # model the caller asked for instead (py/stack-trace-exposure).
        raise HTTPException(
            status_code=404, detail=f"model not found or unreadable: {model}"
        ) from exc

    if engine is None:
        raise HTTPException(status_code=503, detail="engine not available")

    from types import MethodType

    from hfl.api.helpers import queue_response_from_error, run_dispatched
    from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

    # The probes generate on the model: through its queue, off the event
    # loop. Called synchronously here they froze the whole server, health
    # checks included, and raced whatever reply the model was producing.
    # Bound to the engine so ``run_dispatched`` picks *its* dispatcher.
    try:
        result = await run_dispatched(
            MethodType(verify_model, engine), manifest, operation="verify"
        )
    except (QueueFullError, QueueTimeoutError) as exc:
        return queue_response_from_error(exc)
    return {
        "model": result.model,
        "overall_pass": result.overall_pass,
        "duration_ms": result.duration_ms,
        "checks": [asdict(c) for c in result.checks],
    }

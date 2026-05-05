# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 ``POST /api/verify/{model}`` — model sanity-check endpoint."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException

from hfl.api.model_loader import load_llm

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


@router.post(
    "/api/verify/{model:path}",
    response_model=None,
    summary="Sanity-check a registered model",
    responses={
        200: {"description": "VerifyResult"},
        404: {"description": "Model not found"},
        503: {"description": "Engine unavailable"},
    },
)
async def api_verify(model: str) -> dict[str, Any]:
    """Run V4 verification probes against ``model``.

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
    from hfl.engine.verifier import verify_model

    try:
        engine, manifest = await load_llm(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if engine is None:
        raise HTTPException(status_code=503, detail="engine not available")

    result = verify_model(engine, manifest)
    return {
        "model": result.model,
        "overall_pass": result.overall_pass,
        "duration_ms": result.duration_ms,
        "checks": [asdict(c) for c in result.checks],
    }

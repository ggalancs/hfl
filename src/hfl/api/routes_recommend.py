# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 ``GET /api/recommend`` — HW-aware model recommendation.

Wraps :func:`hfl.hub.recommend.recommend_models` with HTTP plumbing
and exposes the host hardware profile in the response so the client
can render "we picked these because your machine has X RAM / Y GPU".
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from hfl.hub.hw_profile import get_hw_profile
from hfl.hub.recommend import Recommendation, recommend_models

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


@router.get(
    "/api/recommend",
    response_model=None,
    summary="Recommend HuggingFace Hub models for the current host",
    responses={
        200: {"description": "Top-N recommendations"},
        503: {"description": "Hub unavailable"},
    },
)
async def api_recommend(
    task: str | None = Query(default=None, max_length=32),
    family: str | None = Query(default=None, max_length=64),
    quantization: str | None = Query(default=None, max_length=32),
    top_n: int = Query(default=5, ge=1, le=50),
) -> dict[str, Any]:
    """Pick top-N HF Hub models that fit the current hardware.

    The response always carries the ``hardware_profile`` so the
    client can explain *why* these picks: "we have 16 GB unified
    memory on Apple Silicon, that's why we recommend MLX 4-bit
    variants over GGUF Q4_K_M".
    """
    profile = get_hw_profile()
    valid_tasks = {"chat", "code", "vision", "embeddings", "tools"}
    if task is not None and task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"task must be one of {sorted(valid_tasks)}; got {task!r}",
        )

    try:
        recommendations = recommend_models(
            task=task,  # type: ignore[arg-type]
            profile=profile,
            family=family,
            quantization=quantization,
            top_n=top_n,
        )
    except Exception as exc:
        logger.exception("recommendation failed")
        raise HTTPException(status_code=503, detail=f"Hub unavailable: {exc}")

    return {
        "hardware_profile": asdict(profile),
        "task": task,
        "family": family,
        "quantization": quantization,
        "total": len(recommendations),
        "recommendations": [_serialise(r) for r in recommendations],
    }


def _serialise(rec: Recommendation) -> dict[str, Any]:
    return {
        "repo_id": rec.repo_id,
        "family": rec.family,
        "quantization": rec.quantization,
        "parameter_estimate_b": rec.parameter_estimate_b,
        "likes": rec.likes,
        "downloads": rec.downloads,
        "license": rec.license,
        "gated": rec.gated,
        "estimated_vram_gb": rec.estimated_vram_gb,
        "score": rec.score,
        "reasoning": rec.reasoning,
    }

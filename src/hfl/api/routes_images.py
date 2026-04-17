# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``POST /api/images/generate`` — image generation (Phase 16 — V2 row 17)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hfl.engine.diffusers_engine import DEFAULT_SIZE, DEFAULT_STEPS, DiffusersEngine, is_available

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Images"])


class ImageRequest(BaseModel):
    """Body for ``POST /api/images/generate``."""

    model: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=4096)
    negative_prompt: str | None = Field(None, max_length=4096)
    size: str = Field("1024x1024", pattern=r"^\d{2,5}x\d{2,5}$")
    steps: int = Field(DEFAULT_STEPS, ge=1, le=200)
    guidance_scale: float = Field(7.5, ge=0.0, le=50.0)
    seed: int | None = Field(None)


@router.post("/api/images/generate", response_model=None)
async def api_images_generate(req: ImageRequest) -> dict[str, Any] | JSONResponse:
    """Generate an image with a local diffusion pipeline.

    Loads the pipeline per request (cold-start on the first call;
    OS-level page cache keeps subsequent calls warm). Returns a
    single base64-encoded PNG — clients decode locally.
    """
    if not is_available():
        raise HTTPException(
            status_code=501,
            detail="Image-generation backend not installed. `pip install 'hfl[imagegen]'`.",
        )
    try:
        width_str, height_str = req.size.split("x")
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"bad size: {req.size!r}")
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="size must be positive")

    engine = DiffusersEngine()
    try:
        engine.load(req.model)
    except Exception:
        logger.exception("image engine load failed for %s", req.model)
        raise HTTPException(status_code=500, detail="image engine load failed")
    try:
        result = engine.generate(
            req.prompt,
            negative_prompt=req.negative_prompt,
            width=width or DEFAULT_SIZE,
            height=height or DEFAULT_SIZE,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )
    except Exception:
        logger.exception("image generation failed")
        raise HTTPException(status_code=500, detail="image generation failed")
    finally:
        engine.unload()

    return {
        "model": req.model,
        "prompt": req.prompt,
        "width": result.width,
        "height": result.height,
        "seed": result.seed,
        "duration_s": result.duration_s,
        "image": {
            "b64": result.image_png_base64,
            "format": "png",
        },
    }


__all__ = ["router"]

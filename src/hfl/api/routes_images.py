# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Image generation: ``POST /api/images/generate`` and OpenAI's
``POST /v1/images/generations`` (Phase 16 — V2 row 17).

``model`` is a Hub repo id that diffusers would download on first use.
Only a caller who may fetch models (the owner) can trigger that; anyone
else is served pipelines already on disk. The load and the rendering run
off the event loop — they used to freeze the whole server.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hfl.engine.diffusers_engine import (
    DEFAULT_SIZE,
    DEFAULT_STEPS,
    DiffusersEngine,
    ImageResult,
    is_available,
)
from hfl.hub.local_cache import hub_model_available_locally

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Images"])

_SIZE = r"^\d{2,5}x\d{2,5}$"


class ImageRequest(BaseModel):
    """Body for ``POST /api/images/generate``."""

    model: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=4096)
    negative_prompt: str | None = Field(None, max_length=4096)
    size: str = Field("1024x1024", pattern=_SIZE)
    steps: int = Field(DEFAULT_STEPS, ge=1, le=200)
    guidance_scale: float = Field(7.5, ge=0.0, le=50.0)
    seed: int | None = Field(None)


class OpenAIImageRequest(BaseModel):
    """Body for ``POST /v1/images/generations``. ``quality``, ``style`` and
    ``user`` are accepted and not used."""

    model: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=4096)
    n: int = Field(1, ge=1, le=4)
    size: str = Field("1024x1024", pattern=_SIZE)
    response_format: Literal["b64_json", "url"] = "b64_json"
    quality: str | None = None
    style: str | None = None
    user: str | None = None


def _dimensions(size: str) -> tuple[int, int]:
    try:
        width_str, height_str = size.split("x")
        width, height = int(width_str), int(height_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"bad size: {size!r}") from None
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="size must be positive")
    return width, height


def _render_sync(model: str, local_only: bool, count: int, **params: Any) -> list[ImageResult]:
    engine = DiffusersEngine()
    try:
        engine.load(model, local_files_only=local_only)
    except Exception:
        logger.exception("image engine load failed for %s", model)
        raise HTTPException(status_code=500, detail="image engine load failed") from None
    try:
        return [engine.generate(**params) for _ in range(count)]
    except Exception:
        logger.exception("image generation failed")
        raise HTTPException(status_code=500, detail="image generation failed") from None
    finally:
        engine.unload()


async def _render(request: Request, model: str, count: int, **params: Any) -> list[ImageResult]:
    if not is_available():
        raise HTTPException(
            status_code=501,
            detail="Image-generation backend not installed. `pip install 'hfl[imagegen]'`.",
        )
    from hfl.api.admin_guard import may_fetch_models

    local_only = not may_fetch_models(request)
    if local_only and not await asyncio.to_thread(hub_model_available_locally, model):
        raise HTTPException(
            status_code=404,
            detail={
                "error": (
                    f"{model} is not on this server, and only the server's owner "
                    "can download models."
                ),
                "code": "model_not_local",
            },
        )
    return await asyncio.to_thread(_render_sync, model, local_only, count, **params)


@router.post("/api/images/generate", response_model=None)
async def api_images_generate(req: ImageRequest, request: Request) -> dict[str, Any] | JSONResponse:
    """Generate an image with a local diffusion pipeline.

    Loads the pipeline per request (cold-start on the first call;
    OS-level page cache keeps subsequent calls warm). Returns a
    single base64-encoded PNG — clients decode locally.
    """
    width, height = _dimensions(req.size)
    (result,) = await _render(
        request,
        req.model,
        1,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=width or DEFAULT_SIZE,
        height=height or DEFAULT_SIZE,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
    )
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


@router.post("/v1/images/generations", response_model=None, tags=["OpenAI"])
async def openai_images(req: OpenAIImageRequest, request: Request) -> dict[str, Any]:
    """OpenAI-compatible image generation, returned as ``b64_json``.

    ``url`` is refused: HFL does not host files for clients to fetch."""
    if req.response_format == "url":
        raise HTTPException(
            status_code=400,
            detail="response_format 'url' is not supported: HFL returns images as b64_json.",
        )
    width, height = _dimensions(req.size)
    results = await _render(
        request,
        req.model,
        req.n,
        prompt=req.prompt,
        width=width,
        height=height,
        steps=DEFAULT_STEPS,
    )
    return {
        "created": int(time.time()),
        "data": [{"b64_json": r.image_png_base64, "revised_prompt": req.prompt} for r in results],
    }


__all__ = ["router"]

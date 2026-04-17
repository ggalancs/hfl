# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Diffusers-backed image generation engine (Phase 16 — V2 row 17).

Scaffolding only: HFL's stated non-goal is to reinvent a diffusion
UI, but the `/api/images/generate` endpoint is a small enough slice
that agent stacks can call it to illustrate their answers. We lean
on ``diffusers.DiffusionPipeline`` so any of SDXL / SDXL-Turbo /
FLUX / Stable Diffusion 3 / etc. work with a single code path — the
specific pipeline is chosen at load time.

The extra is intentionally heavy (diffusers + torch + accelerate +
Pillow) and opt-in via ``pip install 'hfl[imagegen]'``.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DiffusersEngine",
    "ImageResult",
    "is_available",
    "DEFAULT_SIZE",
    "DEFAULT_STEPS",
]


DEFAULT_SIZE = 1024
DEFAULT_STEPS = 30


def is_available() -> bool:
    try:
        import diffusers  # type: ignore  # noqa: F401
    except ImportError:
        return False
    try:
        import torch  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class ImageResult:
    """One generated image + metadata."""

    image_png_base64: str
    seed: int | None = None
    duration_s: float = 0.0
    width: int = 0
    height: int = 0


class DiffusersEngine:
    """Wrap a ``diffusers`` pipeline behind HFL's engine-ish shape.

    Doesn't implement ``InferenceEngine`` because image generation
    lives outside the LLM dispatcher contract.
    """

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._model: str | None = None
        self._device: str | None = None

    def load(self, model: str, *, device: str | None = None) -> None:
        if not is_available():
            raise RuntimeError("Diffusers backend not installed. `pip install 'hfl[imagegen]'`.")
        import torch  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore

        selected = device or (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        dtype = torch.float16 if selected != "cpu" else torch.float32
        pipeline = DiffusionPipeline.from_pretrained(model, torch_dtype=dtype)
        pipeline.to(selected)
        self._pipeline = pipeline
        self._model = model
        self._device = selected

    def unload(self) -> None:
        self._pipeline = None
        self._model = None
        self._device = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        return self._model or "diffusers"

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> ImageResult:
        if not self.is_loaded:
            raise RuntimeError("Diffusers engine not loaded")
        import torch  # type: ignore

        start = time.perf_counter()
        generator = torch.Generator(device=self._device or "cpu")
        if seed is not None:
            generator = generator.manual_seed(int(seed))
        output = self._pipeline(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = output.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return ImageResult(
            image_png_base64=encoded,
            seed=seed,
            duration_s=time.perf_counter() - start,
            width=image.size[0],
            height=image.size[1],
        )

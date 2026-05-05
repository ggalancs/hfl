# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Probe the host hardware for capacity-aware model recommendation.

Used by:

- ``GET /api/recommend`` (V4 F1.2): pick top-N HF Hub models that
  fit the current machine.
- ``POST /api/pull/smart`` (V4 F2.1): pick the best variant of a
  repo for the current machine.

The probe is intentionally cheap (no GPU work, only metadata reads)
so it can be invoked per request without a measurable cost. All
fields are best-effort — when something can't be detected we leave
it ``None`` and the heuristic layer assumes the conservative case
(no GPU, CPU-only).
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


GpuKind = Literal["cuda", "metal", "rocm", "vulkan", "none"]


@dataclass(frozen=True)
class HardwareProfile:
    """Snapshot of capacity-relevant host attributes."""

    os: str
    """``"darwin"``, ``"linux"``, ``"windows"`` (lowercase)."""

    arch: str
    """``"arm64"``, ``"x86_64"``, etc. (lowercase)."""

    system_ram_gb: float
    """Total system RAM. ``0`` when probe failed (treat as CPU-only)."""

    gpu_kind: GpuKind
    """Coarse classification of the primary GPU. ``"none"`` if absent."""

    gpu_vram_gb: float | None
    """VRAM budget in GB. ``None`` when undetectable. On Apple Silicon
    Metal returns the unified memory pool (== system_ram_gb)."""

    has_mlx: bool
    """``mlx_lm`` importable AND host is Darwin-arm64."""

    has_cuda: bool
    """``torch.cuda.is_available()``."""

    has_rocm: bool
    """ROCm runtime detected."""


def _system_ram_gb() -> float:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # pragma: no cover — defensive, psutil is in [dev]
        return 0.0


def _has_mlx() -> bool:
    if platform.system().lower() != "darwin":
        return False
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("mlx_lm") is not None
    except Exception:  # pragma: no cover
        return False


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _has_rocm() -> bool:
    """Detect ROCm via the public env var convention.

    A real probe would call ``rocminfo``; we keep this cheap by
    checking ``HIP_VISIBLE_DEVICES`` / ``ROCM_PATH`` which most
    ROCm-aware Python stacks set up themselves at install time.
    """
    import os

    return bool(os.environ.get("HIP_VISIBLE_DEVICES")) or bool(os.environ.get("ROCM_PATH"))


def _cuda_vram_gb() -> float | None:
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return None
        # Pick the largest visible GPU — that's the one HFL would
        # pin a single-GPU run to.
        sizes = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            sizes.append(props.total_memory / (1024**3))
        return round(max(sizes), 1) if sizes else None
    except Exception:
        return None


def get_hw_profile() -> HardwareProfile:
    """Build the ``HardwareProfile`` for the current host.

    Public entrypoint. Cheap to call (no GPU work).
    """
    os_name = platform.system().lower()
    arch = platform.machine().lower()
    system_ram = _system_ram_gb()
    has_mlx = _has_mlx()
    has_cuda = _has_cuda()
    has_rocm = _has_rocm()

    if has_cuda:
        gpu_kind: GpuKind = "cuda"
        vram = _cuda_vram_gb()
    elif has_mlx:
        # Apple Silicon Metal uses the unified memory pool — VRAM
        # equals system RAM, less the OS reservation. Conservative
        # 70% gives the operator a usable budget.
        gpu_kind = "metal"
        vram = round(system_ram * 0.7, 1) if system_ram else None
    elif has_rocm:
        gpu_kind = "rocm"
        vram = None  # Not probed in this version.
    else:
        gpu_kind = "none"
        vram = None

    return HardwareProfile(
        os=os_name,
        arch=arch,
        system_ram_gb=system_ram,
        gpu_kind=gpu_kind,
        gpu_vram_gb=vram,
        has_mlx=has_mlx,
        has_cuda=has_cuda,
        has_rocm=has_rocm,
    )

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""VRAM detection + context-length tier selection (Phase 11 P1 — V2 row 13).

Ollama auto-sizes ``num_ctx`` based on detected VRAM:

    < 24 GB  → 4 096
    24–48 GB → 32 768
    ≥ 48 GB  → 262 144

We replicate the same ladder. Probing is best-effort and never
raises — a failed probe returns ``None`` and callers fall back to
``config.default_ctx_size`` or the model's advertised context.

Probe order:
  1. NVIDIA via ``pynvml`` (preferred).
  2. Apple Metal via ``torch.mps.recommended_max_memory`` / ctypes
     ``sysctl hw.memsize`` fallback.
  3. AMD ROCm via ``/sys/class/drm/*/mem_info_vram_total``.
  4. Nothing found → ``None``.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "CtxTier",
    "detect_vram_gib",
    "pick_ctx_size",
    "CTX_TIERS",
]


# 4k, 32k, 256k thresholds in GiB, mirroring Ollama.
CTX_TIERS: tuple[tuple[float, int], ...] = (
    (48.0, 262144),  # ≥ 48 GB
    (24.0, 32768),  # 24–48 GB
    (0.0, 4096),  # everything below 24 GB
)


@dataclass
class CtxTier:
    vram_gib: float | None
    ctx: int


# ----------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------


def _probe_nvidia() -> float | None:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        count = pynvml.nvmlDeviceGetCount()
        total = 0
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total += info.total
        if total <= 0:
            return None
        return total / (1024**3)
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _probe_metal() -> float | None:
    if platform.system() != "Darwin":
        return None
    try:
        import torch  # type: ignore

        if getattr(torch.backends, "mps", None) is None:
            return None
        if not torch.backends.mps.is_available():
            return None
        fn = getattr(torch.mps, "recommended_max_memory", None)
        if fn is not None:
            try:
                return fn() / (1024**3)
            except Exception:
                pass
    except ImportError:
        pass

    # Fallback: total physical RAM via sysctl — Apple Silicon uses
    # unified memory, so sysctl's ``hw.memsize`` is the upper bound
    # on what Metal can address.
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        name = b"hw.memsize"
        size = ctypes.c_uint64(0)
        length = ctypes.c_size_t(ctypes.sizeof(size))
        libc.sysctlbyname(name, ctypes.byref(size), ctypes.byref(length), None, 0)
        if size.value > 0:
            return size.value / (1024**3)
    except Exception:
        pass
    return None


def _probe_rocm() -> float | None:
    root = Path("/sys/class/drm")
    if not root.exists():
        return None
    total = 0
    for card in _iter_rocm_cards(root):
        try:
            raw = (card / "device" / "mem_info_vram_total").read_text().strip()
            total += int(raw)
        except Exception:
            continue
    if total <= 0:
        return None
    return total / (1024**3)


def _iter_rocm_cards(root: Path) -> Iterator[Path]:
    try:
        for entry in sorted(root.iterdir()):
            if entry.name.startswith("card") and entry.name[4:].isdigit():
                yield entry
    except Exception:
        return


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def detect_vram_gib() -> float | None:
    """Return total VRAM in gibibytes, or ``None`` if no probe succeeded.

    Tries each backend in order; the first non-``None`` wins. The
    ``HFL_VRAM_OVERRIDE_GIB`` env var short-circuits everything for
    tests / container deployments that know their budget better than
    automatic probing.
    """
    override = os.environ.get("HFL_VRAM_OVERRIDE_GIB")
    if override:
        try:
            return float(override)
        except ValueError:
            logger.warning("ignoring malformed HFL_VRAM_OVERRIDE_GIB=%r", override)

    for probe in (_probe_nvidia, _probe_metal, _probe_rocm):
        try:
            value = probe()
        except Exception:
            logger.debug("vram probe %s raised", probe.__name__, exc_info=True)
            continue
        if value is not None and value > 0:
            return value
    return None


def pick_ctx_size(vram_gib: float | None = None) -> CtxTier:
    """Pick a context size from ``vram_gib`` using the Ollama tiers.

    ``vram_gib=None`` (no probe succeeded) returns the floor tier —
    we assume CPU-only or tiny-GPU deployments and don't extrapolate.
    """
    probed = vram_gib if vram_gib is not None else detect_vram_gib()
    if probed is None:
        return CtxTier(vram_gib=None, ctx=CTX_TIERS[-1][1])
    for threshold, ctx in CTX_TIERS:
        if probed >= threshold:
            return CtxTier(vram_gib=probed, ctx=ctx)
    return CtxTier(vram_gib=probed, ctx=CTX_TIERS[-1][1])

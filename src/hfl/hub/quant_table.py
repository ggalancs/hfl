# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""VRAM estimates for quantised LLM inference.

Used by V4 ``recommend`` and ``pull/smart`` to decide which models
(and which variants) fit on the current host. The numbers are
*conservative* upper bounds — they include weights AND a 4k-context
KV cache at the requested cache dtype, plus a 1.2× safety multiplier.

These are heuristics, not measurements. The point is to avoid
recommending a 70B Q4_K_M to someone with 8GB of VRAM, not to
predict allocation to the megabyte.
"""

from __future__ import annotations

from dataclasses import dataclass

# Quantisation level → bits-per-weight (effective average).
# Mixed quants (``Q4_K_M`` interleaves Q4 weights with Q6 outliers)
# are reported by llama.cpp publications as 4.8 bits/weight average.
_BITS_PER_WEIGHT = {
    "f16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.6,
    "q5_k_m": 5.7,
    "q5_0": 5.5,
    "q4_k_m": 4.85,
    "q4_0": 4.5,
    "q3_k_m": 3.9,
    "q2_k": 3.0,
    # MLX uses different shapes; an "MLX-4bit" pack averages ~4.5 bpw.
    "mlx-4bit": 4.5,
    "mlx-8bit": 8.5,
    # AWQ / GPTQ at 4 bits.
    "awq": 4.5,
    "gptq": 4.5,
    "int4": 4.5,
    "int8": 8.5,
}


# KV cache element size by ``HFL_KV_CACHE_TYPE``.
_KV_BYTES_PER_ELEMENT = {
    "f16": 2.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
}


# Conservative head-dim × kv-heads product for typical LLM
# architectures, indexed by parameter count. Values are rough
# averages from llama.cpp's gguf metadata across published 7B/13B/
# 70B Llama-family variants.
_KV_HIDDEN_PER_LAYER_BY_SIZE = {
    1: 1024,
    3: 2048,
    7: 4096,
    8: 4096,
    13: 5120,
    27: 5120,
    32: 5120,
    34: 7168,
    70: 8192,
    72: 8192,
    180: 12288,
}

# Layer count by parameter count (same source).
_LAYERS_BY_SIZE = {
    1: 16,
    3: 28,
    7: 32,
    8: 32,
    13: 40,
    27: 56,
    32: 60,
    34: 64,
    70: 80,
    72: 80,
    180: 96,
}


@dataclass(frozen=True)
class FitEstimate:
    """Output of :func:`estimate_vram_gb`. Carries the breakdown so
    the caller can render it in a recommendation table."""

    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_gb: float


def _round_to_known_size(params_b: float) -> int:
    """Pick the closest known param size from the lookup tables.

    Lookups are bucketed (1B, 3B, 7B, 13B, 27B, 70B); a 9B variant
    rounds to the closest bucket so the caller still gets an
    estimate even for non-canonical sizes.
    """
    if params_b <= 0:
        return 7  # Default to 7B when the input is missing/zero.
    return min(_LAYERS_BY_SIZE.keys(), key=lambda k: abs(k - params_b))


def estimate_vram_gb(
    *,
    params_b: float,
    quantization: str,
    n_ctx: int = 4096,
    kv_cache_type: str = "f16",
) -> FitEstimate:
    """Conservative VRAM estimate (GB) for serving this configuration.

    Total = weights + KV cache + overhead × 1.2 safety.

    Args:
        params_b: Parameter count in billions.
        quantization: One of the keys in ``_BITS_PER_WEIGHT``.
            Unknown strings fall back to ``f16`` (the worst case).
        n_ctx: Context length used to size the KV cache.
        kv_cache_type: ``f16`` / ``q8_0`` / ``q4_0``.
    """
    bits = _BITS_PER_WEIGHT.get(quantization.lower(), _BITS_PER_WEIGHT["f16"])
    weights_gb = (params_b * 1e9 * bits) / (8 * 1024**3)

    bucket = _round_to_known_size(params_b)
    layers = _LAYERS_BY_SIZE[bucket]
    hidden = _KV_HIDDEN_PER_LAYER_BY_SIZE[bucket]
    kv_bytes = _KV_BYTES_PER_ELEMENT.get(kv_cache_type.lower(), 2.0)

    # KV cache: 2 (K+V) × layers × ctx × hidden × bytes_per_element.
    kv_total_bytes = 2 * layers * n_ctx * hidden * kv_bytes
    kv_cache_gb = kv_total_bytes / (1024**3)

    # Process / runtime overhead: tokenizer buffers, embedding tables,
    # CUDA context, etc. Empirically 0.6-1.5 GB; pick 1.0 as a flat.
    overhead_gb = 1.0

    raw_total = weights_gb + kv_cache_gb + overhead_gb
    safe_total = round(raw_total * 1.2, 2)

    return FitEstimate(
        weights_gb=round(weights_gb, 2),
        kv_cache_gb=round(kv_cache_gb, 2),
        overhead_gb=round(overhead_gb, 2),
        total_gb=safe_total,
    )


def fits_in(
    *,
    params_b: float,
    quantization: str,
    budget_gb: float,
    n_ctx: int = 4096,
    kv_cache_type: str = "f16",
) -> bool:
    """Convenience wrapper: True iff ``estimate_vram_gb`` ≤ budget."""
    estimate = estimate_vram_gb(
        params_b=params_b,
        quantization=quantization,
        n_ctx=n_ctx,
        kv_cache_type=kv_cache_type,
    )
    return estimate.total_gb <= budget_gb

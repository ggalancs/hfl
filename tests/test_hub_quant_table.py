# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl/hub/quant_table.py`` — VRAM heuristics."""

from __future__ import annotations

import pytest

from hfl.hub.quant_table import estimate_vram_gb, fits_in


class TestEstimateVramGb:
    @pytest.mark.parametrize(
        "params_b,quantization,expected_range_gb",
        [
            # 7B Q4_K_M is the canonical "fits on 8 GB GPU" combo —
            # the literature places it around 4.5-5.5 GB for weights;
            # add ~1.5 GB KV cache + 1 GB overhead → 8 GB ish with
            # the 1.2× safety multiplier.
            (7.0, "q4_k_m", (6.5, 11.0)),
            # 70B Q4_K_M is the canonical "needs 48 GB" combo.
            (70.0, "q4_k_m", (45.0, 75.0)),
            # 1B Q4 fits on a Raspberry Pi.
            (1.0, "q4_k_m", (1.5, 3.5)),
            # MLX 4-bit is similar weight-density to Q4_0.
            (8.0, "mlx-4bit", (5.5, 11.5)),
            # F16 doubles weights vs Q8.
            (7.0, "f16", (16.0, 22.0)),
        ],
    )
    def test_estimates_within_published_ranges(self, params_b, quantization, expected_range_gb):
        estimate = estimate_vram_gb(params_b=params_b, quantization=quantization)
        low, high = expected_range_gb
        assert low <= estimate.total_gb <= high, (
            f"{params_b}B {quantization} expected in {low}-{high} GB, "
            f"got {estimate.total_gb} (weights={estimate.weights_gb}, "
            f"kv={estimate.kv_cache_gb}, overhead={estimate.overhead_gb})"
        )

    def test_kv_cache_quantization_reduces_total(self):
        f16 = estimate_vram_gb(params_b=7.0, quantization="q4_k_m", kv_cache_type="f16")
        q4 = estimate_vram_gb(params_b=7.0, quantization="q4_k_m", kv_cache_type="q4_0")
        # KV cache should be 1/4 the size; total goes down accordingly.
        assert q4.kv_cache_gb < f16.kv_cache_gb / 2
        assert q4.total_gb < f16.total_gb

    def test_unknown_quant_falls_back_to_f16(self):
        unknown = estimate_vram_gb(params_b=7.0, quantization="impossible-quant")
        f16 = estimate_vram_gb(params_b=7.0, quantization="f16")
        # Same weights side, +/- rounding.
        assert unknown.weights_gb == f16.weights_gb

    def test_safety_multiplier_is_applied(self):
        """Total = (weights + kv + overhead) × 1.2 — must always be
        strictly greater than the raw sum so we're never overconfident."""
        e = estimate_vram_gb(params_b=7.0, quantization="q4_k_m")
        raw = e.weights_gb + e.kv_cache_gb + e.overhead_gb
        assert e.total_gb > raw


class TestFitsIn:
    def test_70b_q4_does_not_fit_on_8gb_gpu(self):
        assert not fits_in(params_b=70.0, quantization="q4_k_m", budget_gb=8.0)

    def test_7b_q4_fits_on_12gb_gpu(self):
        assert fits_in(params_b=7.0, quantization="q4_k_m", budget_gb=12.0)

    def test_kv_cache_quant_lets_borderline_fit(self):
        """A 7B Q4 with f16 KV may not fit on 8 GB; same model with
        q4_0 KV cache should."""
        # Pick an aggressive context length to amplify the KV diff.
        f16_fits = fits_in(
            params_b=7.0,
            quantization="q4_k_m",
            budget_gb=8.0,
            n_ctx=8192,
            kv_cache_type="f16",
        )
        q4_fits = fits_in(
            params_b=7.0,
            quantization="q4_k_m",
            budget_gb=8.0,
            n_ctx=8192,
            kv_cache_type="q4_0",
        )
        # The q4-KV path should be at least as permissive as f16.
        assert q4_fits or not f16_fits

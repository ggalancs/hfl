# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for the scoring components in ``hfl/hub/recommend.py``.

The integration tests in ``test_routes_recommend.py`` exercise these
transitively, but a regression in any single scorer (``_recency``
returning negative values, ``_popularity`` over-weighting downloads,
``_capability_fit`` ignoring case) would only surface as a strange
ordering at the top of the recommendation list — which is hard to
spot. These tests pin each component independently.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hfl.hub.discovery import DiscoveryEntry
from hfl.hub.hw_profile import HardwareProfile
from hfl.hub.recommend import (
    _capability_fit,
    _hardware_fit,
    _popularity,
    _recency,
)


def _entry(**overrides) -> DiscoveryEntry:
    """Build a DiscoveryEntry with reasonable defaults for the field
    not exercised by the test under hand."""
    base = dict(
        repo_id="user/model",
        likes=100,
        downloads=10_000,
        last_modified=None,
        pipeline_tag=None,
        library=None,
        license=None,
        gated=False,
        tags=[],
        family=None,
        quantization=None,
        parameter_estimate_b=7.0,
    )
    base.update(overrides)
    return DiscoveryEntry(**base)


def _profile(*, kind="cuda", vram=24.0, ram=64.0, mlx=False) -> HardwareProfile:
    return HardwareProfile(
        os="linux",
        arch="x86_64",
        system_ram_gb=ram,
        gpu_kind=kind,
        gpu_vram_gb=vram,
        has_mlx=mlx,
        has_cuda=(kind == "cuda"),
        has_rocm=(kind == "rocm"),
    )


# --- _hardware_fit ----------------------------------------------------------


class TestHardwareFit:
    def test_overflow_returns_zero_score(self):
        # 70B Q4 ≈ 50 GB; on a 16 GB CUDA host nothing fits.
        score, vram, reasons = _hardware_fit(
            _entry(parameter_estimate_b=70.0, quantization="q4_k_m"),
            _profile(kind="cuda", vram=16.0),
        )
        assert score == 0.0
        assert vram > 16.0  # estimate is honest about the overflow
        assert any("overflow" in r.lower() for r in reasons)

    def test_comfortable_fit_returns_full_score(self):
        score, vram, reasons = _hardware_fit(
            _entry(parameter_estimate_b=7.0, quantization="q4_k_m"),
            _profile(kind="cuda", vram=24.0),
        )
        assert score == 1.0
        assert vram < 24.0
        assert any("fits comfortably" in r for r in reasons)

    def test_tight_fit_score_is_between_half_and_one(self):
        """A model that just barely fits should land between 0.5 and
        1.0 (the linear ramp). 7B Q4_K_M ≈ 8 GB; a 9 GB budget leaves
        ~11% headroom which is below the 30% comfortable threshold."""
        score, vram, reasons = _hardware_fit(
            _entry(parameter_estimate_b=7.0, quantization="q4_k_m"),
            _profile(kind="cuda", vram=9.0),
        )
        assert 0.5 <= score < 1.0
        assert vram <= 9.0
        assert any("tight" in r for r in reasons)

    def test_apple_silicon_uses_metal_budget(self):
        """On Apple Silicon Metal we use the unified-memory pool
        budget (set on the profile), not the system RAM total."""
        score, vram, _ = _hardware_fit(
            _entry(parameter_estimate_b=8.0, quantization="mlx-4bit"),
            _profile(kind="metal", vram=22.0, ram=32.0, mlx=True),
        )
        assert score > 0.0
        assert vram < 22.0

    def test_cpu_only_falls_back_to_70_percent_ram(self):
        """No GPU → budget is 70% of system RAM."""
        score, vram, _ = _hardware_fit(
            _entry(parameter_estimate_b=7.0, quantization="q4_k_m"),
            _profile(kind="none", vram=None, ram=16.0),
        )
        # 16 × 0.7 = 11.2 GB budget; 7B Q4 ≈ 7-8 GB → fits.
        assert score > 0.0
        assert vram < 11.2


# --- _capability_fit --------------------------------------------------------


class TestCapabilityFit:
    def test_no_task_returns_neutral(self):
        score, reasons = _capability_fit(_entry(tags=["llama"]), task=None)
        assert score == 0.5
        assert reasons == []

    @pytest.mark.parametrize(
        "task,tags",
        [
            ("chat", ["instruct"]),
            ("code", ["code", "deepseek-coder"]),
            ("vision", ["vision"]),
            ("embeddings", ["sentence-transformers"]),
            ("tools", ["function-calling"]),
        ],
    )
    def test_matching_tag_gets_full_score(self, task, tags):
        score, reasons = _capability_fit(_entry(tags=tags), task=task)
        assert score == 1.0
        assert any(task in r for r in reasons)

    def test_non_matching_tag_gets_low_score(self):
        score, reasons = _capability_fit(_entry(tags=["llama"]), task="vision")
        assert score == 0.2
        assert any("weak match" in r for r in reasons)

    def test_match_via_pipeline_tag(self):
        """``pipeline_tag`` is part of the search pool too — a model
        with ``pipeline_tag="image-to-text"`` should match ``vision``
        even with no relevant ``tags``."""
        score, _ = _capability_fit(
            _entry(tags=[], pipeline_tag="image-to-text"),
            task="vision",
        )
        assert score == 1.0


# --- _popularity ------------------------------------------------------------


class TestPopularity:
    def test_zero_traffic_does_not_raise(self):
        # log10(0) is undefined; the function must guard with max(1, ...).
        score, _ = _popularity(_entry(likes=0, downloads=0))
        assert score >= 0.0

    def test_mainstream_pick_label(self):
        score, reasons = _popularity(_entry(likes=10_000, downloads=10_000_000))
        assert score >= 0.7
        assert any("mainstream" in r for r in reasons)

    def test_niche_pick_label(self):
        score, reasons = _popularity(_entry(likes=1, downloads=10))
        assert score < 0.3
        assert any("niche" in r for r in reasons)

    def test_score_capped_at_one(self):
        """Even a wildly popular model must not score above 1.0 (the
        weighted total expects components in [0, 1])."""
        score, _ = _popularity(_entry(likes=10_000_000, downloads=1_000_000_000))
        assert score <= 1.0


# --- _recency ---------------------------------------------------------------


class TestRecency:
    def test_no_timestamp_returns_neutral(self):
        score, _ = _recency(_entry(last_modified=None))
        assert score == 0.5

    def test_brand_new_model_full_score(self):
        now = datetime.now(timezone.utc)
        score, _ = _recency(_entry(last_modified=now.isoformat()))
        # Today should land at ~1.0 (within rounding).
        assert score >= 0.99

    def test_two_year_old_model_floor(self):
        old = datetime.now(timezone.utc) - timedelta(days=730)
        score, _ = _recency(_entry(last_modified=old.isoformat()))
        # Decay floors at 0 around the 730-day mark.
        assert 0.0 <= score <= 0.05

    def test_invalid_iso_timestamp_returns_neutral(self):
        score, _ = _recency(_entry(last_modified="not-a-date"))
        assert score == 0.5

    def test_z_suffix_is_handled(self):
        """The Hub frequently stamps timestamps with trailing ``Z``;
        the parser must accept that without raising."""
        now = datetime.now(timezone.utc)
        z_stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        score, _ = _recency(_entry(last_modified=z_stamp))
        assert score >= 0.99

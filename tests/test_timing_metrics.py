# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for uniform timing metrics on generation responses.

Phase 5 P1-3. Every Ollama-native response now carries nanosecond
timings sourced from the engine's ``GenerationResult``. Clients that
compute tokens/sec, render progress bars, or feed monitoring
dashboards rely on these being real numbers (they used to be
hard-coded to 0).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationResult


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _install_engine(sample_manifest, *, result: GenerationResult):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.chat = MagicMock(return_value=result)
    engine.generate = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = sample_manifest
    return engine


# ----------------------------------------------------------------------
# GenerationConfig / GenerationResult defaults
# ----------------------------------------------------------------------


class TestGenerationResultDefaults:
    def test_default_timings_are_zero(self):
        r = GenerationResult(text="x")
        assert r.total_duration == 0
        assert r.load_duration == 0
        assert r.prompt_eval_duration == 0
        assert r.eval_duration == 0

    def test_explicit_timings_roundtrip(self):
        r = GenerationResult(
            text="x",
            total_duration=1_000_000_000,
            load_duration=100_000_000,
            prompt_eval_duration=300_000_000,
            eval_duration=600_000_000,
        )
        assert r.total_duration == 1_000_000_000
        assert r.eval_duration == 600_000_000


# ----------------------------------------------------------------------
# /api/generate surfaces timings
# ----------------------------------------------------------------------


class TestGenerateTimings:
    def test_generate_returns_all_timing_fields(self, client, sample_manifest):
        r = GenerationResult(
            text="answer",
            tokens_generated=10,
            tokens_prompt=5,
            total_duration=2_000_000_000,
            load_duration=50_000_000,
            prompt_eval_duration=400_000_000,
            eval_duration=1_550_000_000,
        )
        _install_engine(sample_manifest, result=r)

        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
            },
        )
        assert response.status_code == 200
        body = response.json()

        # Every Ollama-contract timing field must be present and
        # reflect the engine's numbers, not 0.
        assert body["total_duration"] == 2_000_000_000
        assert body["load_duration"] == 50_000_000
        assert body["prompt_eval_duration"] == 400_000_000
        assert body["prompt_eval_count"] == 5
        assert body["eval_duration"] == 1_550_000_000
        assert body["eval_count"] == 10


# ----------------------------------------------------------------------
# /api/chat surfaces timings
# ----------------------------------------------------------------------


class TestChatTimings:
    def test_chat_returns_all_timing_fields(self, client, sample_manifest):
        r = GenerationResult(
            text="answer",
            tokens_generated=20,
            tokens_prompt=8,
            total_duration=3_000_000_000,
            load_duration=0,
            prompt_eval_duration=600_000_000,
            eval_duration=2_400_000_000,
        )
        _install_engine(sample_manifest, result=r)

        response = client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        body = response.json()

        assert body["total_duration"] == 3_000_000_000
        assert body["prompt_eval_count"] == 8
        assert body["prompt_eval_duration"] == 600_000_000
        assert body["eval_count"] == 20
        assert body["eval_duration"] == 2_400_000_000

    def test_timings_sum_close_to_total(self, client, sample_manifest):
        """Invariant the Ollama contract implies: the individual
        phase durations should roughly sum to total_duration.
        Engines may have slop (overhead, GC) so we allow 5%.
        """
        r = GenerationResult(
            text="x",
            tokens_generated=10,
            tokens_prompt=5,
            total_duration=1_000_000_000,
            load_duration=100_000_000,
            prompt_eval_duration=300_000_000,
            eval_duration=590_000_000,
        )
        _install_engine(sample_manifest, result=r)

        body = client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        ).json()

        total = body["total_duration"]
        sum_phases = body["load_duration"] + body["prompt_eval_duration"] + body["eval_duration"]
        # Allow 5% slop for the phase-sum vs. total relationship.
        assert abs(total - sum_phases) < total * 0.05


# ----------------------------------------------------------------------
# Timings are non-negative (guard against underflow bugs)
# ----------------------------------------------------------------------


class TestTimingsAreNonNegative:
    def test_generate_all_nonnegative(self, client, sample_manifest):
        _install_engine(
            sample_manifest,
            result=GenerationResult(text="x", total_duration=0),
        )
        body = client.post(
            "/api/generate",
            json={"model": sample_manifest.name, "prompt": "hi", "stream": False},
        ).json()
        for field in (
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        ):
            assert body[field] >= 0, f"{field} went negative: {body[field]}"

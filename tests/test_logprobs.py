# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for per-token logprobs on /api/generate (Phase 12 P1 — V2 row 7)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.converters import ollama_to_generation_config
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationConfig, GenerationResult
from hfl.models.manifest import ModelManifest


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _install(manifest, *, result: GenerationResult):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.generate = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = manifest
    return engine


class TestConverter:
    def test_default_is_zero(self):
        cfg = ollama_to_generation_config(None)
        assert cfg.logprobs == 0

    def test_positive_value_accepted(self):
        cfg = ollama_to_generation_config({"logprobs": 5})
        assert cfg.logprobs == 5

    def test_negative_clamped_to_zero(self):
        cfg = ollama_to_generation_config({"logprobs": -3})
        assert cfg.logprobs == 0

    def test_above_20_clamped(self):
        cfg = ollama_to_generation_config({"logprobs": 9999})
        assert cfg.logprobs == 20

    def test_non_numeric_becomes_zero(self):
        cfg = ollama_to_generation_config({"logprobs": "blah"})
        assert cfg.logprobs == 0


class TestGenerationResultField:
    def test_default_is_none(self):
        r = GenerationResult(text="x")
        assert r.logprobs is None


class TestRouteEnvelope:
    def test_logprobs_absent_by_default(self, client):
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        _install(
            manifest,
            result=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1),
        )
        resp = client.post(
            "/api/generate",
            json={"model": "m", "prompt": "hi", "stream": False},
        )
        assert "logprobs" not in resp.json()

    def test_logprobs_attached_when_engine_populates(self, client):
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        lp = [
            {"token": "h", "logprob": -0.1, "top_logprobs": [{"token": "h", "logprob": -0.1}]},
            {"token": "i", "logprob": -0.2, "top_logprobs": []},
        ]
        _install(
            manifest,
            result=GenerationResult(
                text="hi",
                tokens_generated=2,
                tokens_prompt=1,
                logprobs=lp,
            ),
        )
        resp = client.post(
            "/api/generate",
            json={
                "model": "m",
                "prompt": "hi",
                "options": {"logprobs": 5},
                "stream": False,
            },
        )
        body = resp.json()
        assert body["logprobs"] == lp


class TestEnginePlumbing:
    def test_logprobs_flows_to_engine_config(self, client):
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        engine = _install(
            manifest,
            result=GenerationResult(text="x", tokens_generated=1, tokens_prompt=1),
        )
        client.post(
            "/api/generate",
            json={
                "model": "m",
                "prompt": "hi",
                "options": {"logprobs": 3},
                "stream": False,
            },
        )
        cfg = engine.generate.call_args[0][1]
        assert isinstance(cfg, GenerationConfig)
        assert cfg.logprobs == 3

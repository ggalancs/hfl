# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the legacy ``context`` array on ``/api/generate`` (Phase 7 P2-4).

Ollama's ``/api/generate`` has historically returned a ``context``
array of encoded prompt+response tokens so clients can echo them
back on the next call for multi-turn continuation without a full
role-tagged chat loop. HFL opts into this on demand via
``options.keep_context=true`` — default off because the array is
large and most clients use ``/api/chat`` now.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.converters import ollama_to_generation_config
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationConfig, GenerationResult


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _install_engine(sample_manifest, *, result: GenerationResult):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.generate = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = sample_manifest
    return engine


# ----------------------------------------------------------------------
# GenerationConfig / Converter
# ----------------------------------------------------------------------


class TestConverterFlagThrough:
    def test_keep_context_defaults_to_false(self):
        cfg = ollama_to_generation_config(None)
        assert cfg.keep_context is False

    def test_keep_context_true_is_honoured(self):
        cfg = ollama_to_generation_config({"keep_context": True})
        assert cfg.keep_context is True

    def test_keep_context_absent_stays_false(self):
        cfg = ollama_to_generation_config({"temperature": 0.5})
        assert cfg.keep_context is False


class TestGenerationResultField:
    def test_context_tokens_defaults_to_none(self):
        r = GenerationResult(text="x")
        assert r.context_tokens is None

    def test_context_tokens_roundtrip(self):
        r = GenerationResult(text="x", context_tokens=[1, 2, 3])
        assert r.context_tokens == [1, 2, 3]


# ----------------------------------------------------------------------
# Route envelope
# ----------------------------------------------------------------------


class TestGenerateResponseEnvelope:
    def test_context_omitted_when_keep_context_false(self, client, sample_manifest):
        result = GenerationResult(
            text="hello",
            tokens_generated=1,
            tokens_prompt=1,
            context_tokens=[42, 7],
        )
        _install_engine(sample_manifest, result=result)
        resp = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert "context" not in resp.json()

    def test_context_present_when_keep_context_true(self, client, sample_manifest):
        result = GenerationResult(
            text="hello",
            tokens_generated=1,
            tokens_prompt=1,
            context_tokens=[42, 7],
        )
        _install_engine(sample_manifest, result=result)
        resp = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "options": {"keep_context": True},
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["context"] == [42, 7]

    def test_keep_context_plumbs_into_engine_config(self, client, sample_manifest):
        """The config passed to engine.generate reflects keep_context."""
        result = GenerationResult(text="x", tokens_generated=1, tokens_prompt=1)
        engine = _install_engine(sample_manifest, result=result)
        client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "options": {"keep_context": True},
                "stream": False,
            },
        )
        cfg = engine.generate.call_args[0][1]
        assert isinstance(cfg, GenerationConfig)
        assert cfg.keep_context is True

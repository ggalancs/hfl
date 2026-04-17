# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for multi-level thinking on /api/chat + /api/generate (Phase 10 P1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.routes_native import _resolve_thinking_level
from hfl.api.schemas.ollama import ChatRequest, GenerateRequest
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationConfig, GenerationResult
from hfl.models.manifest import ModelManifest


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _install_engine(manifest, *, result: GenerationResult):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.chat = MagicMock(return_value=result)
    engine.generate = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = manifest
    return engine


class TestLevelResolver:
    def test_none_is_off(self):
        assert _resolve_thinking_level(None) == "off"

    def test_false_is_off(self):
        assert _resolve_thinking_level(False) == "off"

    def test_true_is_medium(self):
        assert _resolve_thinking_level(True) == "medium"

    def test_strings_pass_through(self):
        assert _resolve_thinking_level("low") == "low"
        assert _resolve_thinking_level("medium") == "medium"
        assert _resolve_thinking_level("high") == "high"

    def test_case_insensitive(self):
        assert _resolve_thinking_level("HIGH") == "high"
        assert _resolve_thinking_level("  medium  ") == "medium"

    def test_unknown_becomes_off(self):
        assert _resolve_thinking_level("ultra") == "off"


class TestGenerationConfigField:
    def test_thinking_level_defaults_to_off(self):
        assert GenerationConfig().thinking_level == "off"


class TestSchemaAccepts:
    def test_generate_think_string_accepted(self):
        req = GenerateRequest(model="m", prompt="p", think="high")
        assert req.think == "high"

    def test_generate_think_bool_still_accepted(self):
        req = GenerateRequest(model="m", prompt="p", think=True)
        assert req.think is True

    def test_chat_think_string_accepted(self):
        req = ChatRequest(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            think="low",
        )
        assert req.think == "low"


class TestRoutePlumbing:
    def test_think_high_maps_to_config(self, client):
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        engine = _install_engine(
            manifest,
            result=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1),
        )
        resp = client.post(
            "/api/generate",
            json={"model": "m", "prompt": "hi", "think": "high", "stream": False},
        )
        assert resp.status_code == 200
        cfg = engine.generate.call_args[0][1]
        assert cfg.thinking_level == "high"
        assert cfg.expose_reasoning is True

    def test_think_false_leaves_level_off(self, client):
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        engine = _install_engine(
            manifest,
            result=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1),
        )
        resp = client.post(
            "/api/generate",
            json={"model": "m", "prompt": "hi", "think": False, "stream": False},
        )
        assert resp.status_code == 200
        cfg = engine.generate.call_args[0][1]
        assert cfg.thinking_level == "off"
        assert cfg.expose_reasoning is False

    def test_chat_think_medium_string(self, client):
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        engine = _install_engine(
            manifest,
            result=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1),
        )
        resp = client.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "think": "medium",
                "stream": False,
            },
        )
        assert resp.status_code == 200
        cfg = engine.chat.call_args[0][1]
        assert cfg.thinking_level == "medium"

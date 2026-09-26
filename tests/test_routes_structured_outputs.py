# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for structured-output request fields.

Exercises ``format`` on Ollama routes and ``response_format`` on
OpenAI chat completions. The backend chat() is mocked so these tests
run without the ``[llama]`` extra — they pin the wire contract (how
the field flows from HTTP body to GenerationConfig) not the grammar
compilation (which lives in test_engine_llama_cpp.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _mock_llm(sample_manifest):
    """Wire a chat-capable mock engine onto state."""
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.chat = MagicMock(
        return_value=MagicMock(
            text='{"key":"value"}',
            tokens_generated=1,
            tokens_prompt=1,
            stop_reason="stop",
        )
    )
    engine.generate = MagicMock(
        return_value=MagicMock(
            text='{"key":"value"}',
            tokens_generated=1,
            tokens_prompt=1,
            stop_reason="stop",
        )
    )
    engine.supports_structured_output = True  # stands for llama.cpp
    state.engine = engine
    state.current_model = sample_manifest
    return engine


class TestABackendThatCannotConstrainRefuses:
    """MLX and Transformers ignored a response format and answered prose; the
    request is now refused before any inference, in each API's dialect."""

    @pytest.mark.parametrize(
        ("path", "body", "shape"),
        [
            ("/api/generate", {"prompt": "x", "format": "json", "stream": False}, "flat"),
            ("/api/chat", {"messages": [{"role": "user", "content": "x"}], "format": "json",
                           "stream": False}, "flat"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}],
                                      "response_format": {"type": "json_object"}}, "openai"),
        ],
    )  # fmt: skip
    def test_refused(self, client, sample_manifest, path, body, shape):
        engine = _mock_llm(sample_manifest)
        engine.supports_structured_output = False
        response = client.post(path, json={"model": sample_manifest.name, **body})
        assert response.status_code == 400
        error = response.json()["error"]
        message = error if shape == "flat" else error["message"]
        assert "cannot constrain its output" in message
        engine.chat.assert_not_called()
        engine.generate.assert_not_called()

    def test_without_a_format_nothing_is_refused(self, client, sample_manifest):
        engine = _mock_llm(sample_manifest)
        engine.supports_structured_output = False
        body = {"model": sample_manifest.name, "prompt": "x", "stream": False}
        assert client.post("/api/generate", json=body).status_code == 200


class TestOllamaGenerateFormat:
    def test_format_json_literal_flows_to_engine(self, client, sample_manifest):
        engine = _mock_llm(sample_manifest)
        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "give me json",
                "stream": False,
                "format": "json",
            },
        )
        assert response.status_code == 200
        # Verify the engine received response_format="json" in its config
        assert engine.generate.called
        # generate(prompt, config) — config is arg index 1
        call_config = engine.generate.call_args[0][1]
        assert call_config.response_format == "json"

    def test_format_schema_dict_flows_to_engine(self, client, sample_manifest):
        engine = _mock_llm(sample_manifest)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "give me a named person",
                "stream": False,
                "format": schema,
            },
        )
        assert response.status_code == 200
        call_config = engine.generate.call_args[0][1]
        assert call_config.response_format == schema

    def test_invalid_format_string_rejected(self, client, sample_manifest):
        _mock_llm(sample_manifest)
        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "stream": False,
                "format": "yaml",
            },
        )
        assert response.status_code == 400


class TestOllamaChatFormat:
    def test_format_json_on_chat(self, client, sample_manifest):
        engine = _mock_llm(sample_manifest)
        response = client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "format": "json",
            },
        )
        assert response.status_code == 200
        call_config = engine.chat.call_args[0][1]
        assert call_config.response_format == "json"


class TestOpenAIResponseFormat:
    def test_type_text_leaves_config_default(self, client, sample_manifest):
        engine = _mock_llm(sample_manifest)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "text"},
            },
        )
        assert response.status_code == 200
        call_config = engine.chat.call_args[0][1]
        assert call_config.response_format is None

    def test_type_json_object_sets_literal(self, client, sample_manifest):
        engine = _mock_llm(sample_manifest)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            },
        )
        assert response.status_code == 200
        call_config = engine.chat.call_args[0][1]
        assert call_config.response_format == "json"

    def test_json_schema_unwraps_to_inner(self, client, sample_manifest):
        engine = _mock_llm(sample_manifest)
        inner = {"type": "object", "properties": {"x": {"type": "integer"}}}
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "X", "schema": inner},
                },
            },
        )
        assert response.status_code == 200
        call_config = engine.chat.call_args[0][1]
        assert call_config.response_format == inner

    def test_unknown_response_format_type_rejected(self, client, sample_manifest):
        _mock_llm(sample_manifest)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "yaml"},
            },
        )
        assert response.status_code == 400

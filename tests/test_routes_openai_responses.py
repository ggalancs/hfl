# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``POST /v1/responses`` (OpenAI Responses API).

Pins the wire format that ``client.responses.create(...)`` keys on.
The Responses API is the higher-level wrapper OpenAI introduced in
2025; HFL implements it on top of the existing chat-completion path
without a new engine.
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


@pytest.fixture
def llm_manifest():
    from hfl.models.manifest import ModelManifest

    return ModelManifest(
        name="qwen-coder-7b",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        local_path="/tmp/qwen-coder-7b.gguf",
        format="gguf",
        architecture="qwen",
        parameters="7B",
        quantization="Q4_K_M",
        size_bytes=4_200_000_000,
    )


def _wire_engine(manifest, *, text="hello world"):
    """Mock a loaded engine returning a deterministic chat result."""
    state = get_state()
    engine = MagicMock()
    engine.is_loaded = True
    result = MagicMock()
    result.text = text
    result.tokens_prompt = 5
    result.tokens_generated = 7
    result.stop_reason = "stop"
    result.tool_calls = None
    result.reasoning_text = None
    engine.chat = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = manifest


class TestResponsesNonStream:
    def test_string_input_produces_output_message(self, client, llm_manifest):
        _wire_engine(llm_manifest, text="forty-two")

        body = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "What is the answer?"},
        ).json()

        assert body["object"] == "response"
        assert body["model"] == llm_manifest.name
        assert body["status"] == "completed"
        # Output is a heterogeneous list — find the assistant message.
        messages = [item for item in body["output"] if item["type"] == "message"]
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["type"] == "output_text"
        assert messages[0]["content"][0]["text"] == "forty-two"

    def test_list_input_with_typed_parts_is_flattened(self, client, llm_manifest):
        _wire_engine(llm_manifest, text="ack")

        response = client.post(
            "/v1/responses",
            json={
                "model": llm_manifest.name,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Hello "},
                            {"type": "input_text", "text": "world"},
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200

        # Inspect what we forwarded to engine.chat — first positional
        # arg is the messages list.
        engine = get_state().engine
        sent_messages = engine.chat.call_args.args[0]
        assert sent_messages[-1].role == "user"
        assert sent_messages[-1].content == "Hello world"

    def test_instructions_become_system_message(self, client, llm_manifest):
        _wire_engine(llm_manifest, text="ok")

        client.post(
            "/v1/responses",
            json={
                "model": llm_manifest.name,
                "input": "hello",
                "instructions": "You are a polite assistant.",
            },
        )

        sent_messages = get_state().engine.chat.call_args.args[0]
        assert sent_messages[0].role == "system"
        assert sent_messages[0].content == "You are a polite assistant."
        assert sent_messages[1].role == "user"
        assert sent_messages[1].content == "hello"

    def test_reasoning_effort_is_forwarded_as_thinking_level(self, client, llm_manifest):
        _wire_engine(llm_manifest, text="ok")

        client.post(
            "/v1/responses",
            json={
                "model": llm_manifest.name,
                "input": "puzzle",
                "reasoning": {"effort": "high"},
            },
        )

        # Second positional arg is the GenerationConfig.
        cfg = get_state().engine.chat.call_args.args[1]
        assert cfg.thinking_level == "high"
        assert cfg.expose_reasoning is True

    def test_usage_block_is_populated(self, client, llm_manifest):
        _wire_engine(llm_manifest, text="ok")

        body = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "hi"},
        ).json()

        assert body["usage"] == {
            "input_tokens": 5,
            "output_tokens": 7,
            "total_tokens": 12,
        }

    def test_tool_calls_are_emitted_as_function_call_items(self, client, llm_manifest):
        """When the engine surfaces structured ``tool_calls``, they
        become ``function_call`` items in ``output[]`` rather than
        polluting the ``message.content``."""
        _wire_engine(llm_manifest, text="")
        get_state().engine.chat.return_value.tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Madrid"}'},
            }
        ]

        body = client.post(
            "/v1/responses",
            json={
                "model": llm_manifest.name,
                "input": "weather?",
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        ).json()

        function_calls = [item for item in body["output"] if item["type"] == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "get_weather"
        assert function_calls[0]["arguments"] == '{"city": "Madrid"}'

    def test_response_id_is_unique_per_request(self, client, llm_manifest):
        _wire_engine(llm_manifest, text="ok")

        a = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "1"},
        ).json()
        b = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "2"},
        ).json()
        assert a["id"] != b["id"]
        assert a["id"].startswith("resp_")


class TestResponsesValidation:
    def test_missing_input_field_is_400(self, client, llm_manifest):
        _wire_engine(llm_manifest, text="ok")

        response = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name},
        )
        assert response.status_code in (400, 422)

    def test_empty_model_is_rejected(self, client, llm_manifest):
        _wire_engine(llm_manifest, text="ok")

        response = client.post(
            "/v1/responses",
            json={"model": "", "input": "hi"},
        )
        assert response.status_code in (400, 422)

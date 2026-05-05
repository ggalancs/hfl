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


def _parse_sse_events(body: str) -> list[dict]:
    """Decode a ``text/event-stream`` body into the list of ``data:``
    payloads. Drops the ``[DONE]`` terminator so callers can assert on
    the actual events.
    """
    import json

    events: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


class TestResponsesStreaming:
    """Cover the SSE path of ``POST /v1/responses``.

    The server re-emits chat tokens as Responses-shaped events:
    ``response.created`` once at the start, ``response.output_text.delta``
    per chunk, ``response.completed`` with the final envelope, and a
    final ``[DONE]`` line.
    """

    def _wire_streaming_engine(self, manifest, *, tokens):
        """Mock a chat_stream that yields ``tokens`` then stops."""
        state = get_state()
        engine = MagicMock(is_loaded=True)
        engine.chat_stream = MagicMock(return_value=iter(list(tokens)))
        state.engine = engine
        state.current_model = manifest

    def test_emits_created_then_deltas_then_completed(self, client, llm_manifest):
        self._wire_streaming_engine(llm_manifest, tokens=["Hel", "lo", " world"])

        response = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "hi", "stream": True},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(response.text)
        types = [e["type"] for e in events]
        # Required event grammar.
        assert types[0] == "response.created"
        # One delta per yielded token.
        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        assert [d["delta"] for d in deltas] == ["Hel", "lo", " world"]
        # Last event is the completion envelope.
        assert types[-1] == "response.completed"

    def test_done_terminator_is_present(self, client, llm_manifest):
        """The SSE stream must end with the ``[DONE]`` sentinel that
        OpenAI SDKs key on to close the iterator cleanly."""
        self._wire_streaming_engine(llm_manifest, tokens=["x"])

        response = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "hi", "stream": True},
        )
        # Last non-empty line in the body.
        non_empty = [line for line in response.text.splitlines() if line]
        assert non_empty[-1] == "data: [DONE]"

    def test_completed_event_carries_full_response_envelope(self, client, llm_manifest):
        """``response.completed`` must include the same shape the
        non-streaming endpoint returns — id/object/model/output."""
        self._wire_streaming_engine(llm_manifest, tokens=["a", "b", "c"])

        response = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "hi", "stream": True},
        )
        events = _parse_sse_events(response.text)
        completed = next(e for e in events if e["type"] == "response.completed")
        envelope = completed["response"]
        assert envelope["object"] == "response"
        assert envelope["model"] == llm_manifest.name
        assert envelope["status"] == "completed"
        # Reconstructed text matches the streamed tokens.
        msg = next(item for item in envelope["output"] if item["type"] == "message")
        assert msg["content"][0]["text"] == "abc"

    def test_response_id_is_consistent_across_created_and_completed(self, client, llm_manifest):
        """Both wrapper events must carry the SAME ``response.id`` so
        clients can correlate them."""
        self._wire_streaming_engine(llm_manifest, tokens=["x"])

        response = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "hi", "stream": True},
        )
        events = _parse_sse_events(response.text)
        created = next(e for e in events if e["type"] == "response.created")
        completed = next(e for e in events if e["type"] == "response.completed")
        assert created["response"]["id"] == completed["response"]["id"]
        assert created["response"]["id"].startswith("resp_")

    def test_engine_none_at_load_path_returns_http_error(self, client, llm_manifest):
        """When ``state.current_model`` is set but ``state.engine`` is
        None, ``load_llm`` raises ``ModelNotReadyError`` and the
        endpoint returns an HTTP error envelope before the SSE stream
        starts. This is the "model registered but failed to load" path;
        the in-stream ``response.failed`` event is reserved for engines
        that vanish mid-flight (covered separately by the
        ``_get_state`` guard in ``_stream_response``)."""
        state = get_state()
        state.current_model = llm_manifest
        state.engine = None

        response = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "hi", "stream": True},
        )
        # ModelNotReadyError surfaces as 503; the body is JSON, not SSE.
        assert response.status_code in (500, 503)
        assert "text/event-stream" not in response.headers.get("content-type", "")

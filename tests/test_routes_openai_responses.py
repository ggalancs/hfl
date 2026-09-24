# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``POST /v1/responses`` (OpenAI Responses API).

Pins the wire format that ``client.responses.create(...)`` keys on.
The Responses API is the higher-level wrapper OpenAI introduced in
2025; HFL implements it on top of the existing chat-completion path
without a new engine.
"""

from __future__ import annotations

import json
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


class TestResponsesStreamingDispatcher:
    """CON/API regression: streaming ``/v1/responses`` must run on the inference
    dispatcher (serialised against other requests on the shared, non-reentrant
    model) instead of calling the engine directly, and must not leak raw
    tool-call markers as text."""

    def _wire_streaming_engine(self, manifest, *, tokens):
        state = get_state()
        engine = MagicMock(is_loaded=True)
        engine.chat_stream = MagicMock(return_value=iter(list(tokens)))
        state.engine = engine
        state.current_model = manifest
        return engine

    def test_streaming_acquires_dispatcher_slot(self, client, llm_manifest, monkeypatch):
        """The stream must go through ``acquire_stream_slot`` (slot held for the
        whole stream) — proof it serialises on the dispatcher like every other
        route, rather than bypassing it and racing the shared model's KV cache."""
        import hfl.api.helpers as helpers

        seen: list[str | None] = []
        real = helpers.acquire_stream_slot

        async def _spy(*, path=None):
            seen.append(path)
            return await real(path=path)

        monkeypatch.setattr(helpers, "acquire_stream_slot", _spy)
        self._wire_streaming_engine(llm_manifest, tokens=["hi"])

        resp = client.post(
            "/v1/responses",
            json={"model": llm_manifest.name, "input": "hi", "stream": True},
        )
        assert resp.status_code == 200
        assert "/v1/responses" in seen, "streaming path did not acquire a dispatcher slot"

    def test_streaming_tools_do_not_leak_markers_and_surface_function_call(
        self, client, llm_manifest
    ):
        """The fix for the parity break: with tools declared, the raw
        ``<tool_call>`` marker must never appear in an ``output_text.delta``,
        and the ``response.completed`` envelope must carry a structured
        ``function_call`` item (matching the non-streaming endpoint)."""
        marker = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        # Split across token chunks to mimic real streaming.
        self._wire_streaming_engine(llm_manifest, tokens=["<tool_call>", marker[11:]])

        resp = client.post(
            "/v1/responses",
            json={
                "model": llm_manifest.name,
                "input": "weather in Paris?",
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        )
        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)

        # No text delta may leak the raw marker.
        deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
        assert all("tool_call" not in (e.get("delta") or "") for e in deltas), (
            "raw tool-call marker leaked into output_text.delta"
        )

        # The completed envelope carries the structured function_call.
        completed = next(e for e in events if e.get("type") == "response.completed")
        items = completed["response"]["output"]
        fcs = [it for it in items if it.get("type") == "function_call"]
        assert fcs, "no function_call item in completed envelope"
        assert fcs[0]["name"] == "get_weather"


class TestResponsesDefaults:
    def test_default_max_tokens_matches_chat_route(self):
        """API-11: when max_output_tokens is omitted, default to a sane cap
        (2048, matching the chat route) instead of 0/unbounded."""
        from hfl.api.routes_openai_responses import ResponsesRequest, _build_gen_config

        cfg = _build_gen_config(ResponsesRequest(model="m", input="hi"))
        assert cfg.max_tokens == 2048

    def test_explicit_max_output_tokens_is_honoured(self):
        from hfl.api.routes_openai_responses import ResponsesRequest, _build_gen_config

        cfg = _build_gen_config(ResponsesRequest(model="m", input="hi", max_output_tokens=64))
        assert cfg.max_tokens == 64


class TestResponsesForAgents:
    """What an agent on the Responses API (Codex) needs, found by running
    Codex against HFL: it builds the turn from ``response.output_item.*``
    events — a stream with only created/completed reads as an empty turn —
    and sends the tool history back as ``function_call`` /
    ``function_call_output`` items with flat tool definitions."""

    READ = {
        "type": "function",
        "name": "Read",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }

    def _stream(self, client, manifest, tokens, **body):
        state = get_state()
        engine = MagicMock(is_loaded=True)
        engine.chat_stream = MagicMock(return_value=iter(list(tokens)))
        state.engine = engine
        state.current_model = manifest
        response = client.post(
            "/v1/responses",
            json={"model": manifest.name, "input": "hi", "stream": True, **body},
        )
        return engine, _parse_sse_events(response.text)

    def test_a_text_turn_streams_its_message_item(self, client, llm_manifest):
        _, events = self._stream(client, llm_manifest, ["Hel", "lo"])
        types = [e["type"] for e in events]
        assert types == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        added = events[2]["item"]
        done = events[8]["item"]
        assert added["type"] == "message" and added["id"] == done["id"]
        assert all(e["item_id"] == added["id"] for e in events[3:8])
        assert events[6]["text"] == "Hello"
        assert done["content"] == [{"type": "output_text", "text": "Hello", "annotations": []}]
        assert [e["sequence_number"] for e in events] == list(range(len(events)))

    def test_a_tool_turn_streams_its_function_call_item(self, client, llm_manifest):
        marker = '<tool_call>{"name": "Read", "arguments": {"path": "a.py"}}</tool_call>'
        _, events = self._stream(
            client, llm_manifest, [marker[:20], marker[20:]], tools=[self.READ]
        )
        types = [e["type"] for e in events]
        assert types == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
        item = events[5]["item"]
        assert item["type"] == "function_call" and item["name"] == "Read"
        assert json.loads(item["arguments"]) == {"path": "a.py"}
        assert item["call_id"].startswith("call_")
        assert events[4]["arguments"] == item["arguments"]
        assert events[-1]["response"]["output"] == [item]
        assert "<tool_call>" not in json.dumps(events)

    def test_flat_tools_reach_the_engine_nested_and_others_are_dropped(self, client, llm_manifest):
        engine, _ = self._stream(
            client,
            llm_manifest,
            ["ok"],
            tools=[self.READ, {"type": "web_search"}, {"type": "namespace", "name": "mcp"}],
        )
        tools = engine.chat_stream.call_args.kwargs["tools"]
        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": self.READ["parameters"],
                },
            }
        ]

    def test_the_tool_history_reaches_the_model(self, client, llm_manifest):
        engine, _ = self._stream(
            client,
            llm_manifest,
            ["done"],
            tools=[self.READ],
            input=[
                {"type": "message", "role": "developer", "content": "Be brief."},
                {"type": "message", "role": "user", "content": "Read a.py"},
                {"type": "reasoning", "summary": []},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"path": "a.py"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "x = 1"},
            ],
        )
        messages = engine.chat_stream.call_args.args[0]
        assert [(m.role, m.content) for m in messages] == [
            ("system", "Be brief."),
            ("user", "Read a.py"),
            ("assistant", ""),
            ("tool", "x = 1"),
        ]
        assert messages[2].tool_calls == [
            {"id": "call_1", "function": {"name": "Read", "arguments": {"path": "a.py"}}}
        ]
        assert messages[3].tool_call_id == "call_1"
        assert messages[3].name == "Read"

    def test_consecutive_calls_share_one_assistant_turn(self, client, llm_manifest):
        engine, _ = self._stream(
            client,
            llm_manifest,
            ["done"],
            tools=[self.READ],
            input=[
                {"type": "function_call", "call_id": "c1", "name": "Read", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "Read", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "1"},
                {"type": "function_call_output", "call_id": "c2", "output": "2"},
            ],
        )
        messages = engine.chat_stream.call_args.args[0]
        assert [m.role for m in messages] == ["assistant", "tool", "tool"]
        assert [c["id"] for c in messages[0].tool_calls] == ["c1", "c2"]

# SPDX-License-Identifier: HRUL-1.0
"""Tool-calling on the OpenAI-compatible /v1/chat/completions route.

Mirrors the acceptance suite in ``hfl-tool-calling-spec.md`` (T2-T5) but for
the OpenAI wire format consumed by OpenAI-SDK / Vercel-AI-SDK clients (e.g.
the Terax agent): structured ``tool_calls`` with string ``arguments`` and a
``finish_reason`` of ``tool_calls``.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state

WRITE_WIKI_TOOL = {
    "type": "function",
    "function": {
        "name": "write_wiki",
        "description": "Create or overwrite a wiki article",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}


def _result(text: str, **over):
    base = dict(
        text=text,
        tokens_generated=5,
        tokens_prompt=10,
        tokens_per_second=50.0,
        stop_reason="stop",
        tool_calls=None,
    )
    base.update(over)
    return MagicMock(**base)


class TestOpenAIToolCalling:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None
        yield
        get_state().api_key = None
        get_state().engine = None
        get_state().current_model = None

    def _wire(self, engine: MagicMock) -> TestClient:
        engine.is_loaded = True
        model = MagicMock()
        model.name = "qwen3-32b"
        get_state().engine = engine
        get_state().current_model = model
        return TestClient(app)

    def test_tool_call_is_exposed_in_message_tool_calls(self):
        """T2: a <tool_call> marker in the text becomes structured tool_calls."""
        engine = MagicMock()
        engine.chat.return_value = _result(
            '<tool_call>\n{"name": "write_wiki", '
            '"arguments": {"path": "topics/hello.md", "content": "Hello world."}}\n</tool_call>'
        )
        client = self._wire(engine)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-32b",
                "messages": [{"role": "user", "content": "Save Hello at topics/hello.md"}],
                "tools": [WRITE_WIKI_TOOL],
            },
        )
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        msg = choice["message"]
        assert msg["content"] is None
        calls = msg["tool_calls"]
        assert isinstance(calls, list) and len(calls) == 1
        call = calls[0]
        assert call["type"] == "function"
        assert call["id"]
        assert call["function"]["name"] == "write_wiki"
        # arguments MUST be a JSON string (OpenAI wire), not an object.
        assert isinstance(call["function"]["arguments"], str)
        args = json.loads(call["function"]["arguments"])
        assert args["path"] == "topics/hello.md"

        # tools were forwarded to the engine as a kwarg.
        assert engine.chat.call_args.kwargs.get("tools")

    def test_tools_field_forwards_engine_kwarg_and_parses(self):
        """The engine receives the tools list so its template can apply it."""
        engine = MagicMock()
        engine.chat.return_value = _result("plain answer, no tool", tool_calls=None)
        client = self._wire(engine)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-32b",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [WRITE_WIKI_TOOL],
            },
        )
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "plain answer, no tool"
        assert choice["message"].get("tool_calls") is None

    def test_multi_turn_forwards_tool_result_to_engine(self):
        """T3: a prior assistant tool_calls turn + role:tool result reach the engine."""
        engine = MagicMock()
        engine.chat.return_value = _result("It is 22C and sunny in Madrid.")
        client = self._wire(engine)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-32b",
                "messages": [
                    {"role": "user", "content": "Weather in Madrid?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Madrid"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "get_weather",
                        "content": "22C sunny",
                    },
                ],
                "tools": [WRITE_WIKI_TOOL],
            },
        )
        assert resp.status_code == 200
        # The engine saw all three turns, including the tool result.
        sent = engine.chat.call_args.args[0]
        roles = [m.role for m in sent]
        assert roles == ["user", "assistant", "tool"]
        assistant_turn = sent[1]
        assert assistant_turn.tool_calls
        # OpenAI arguments string was parsed back to a dict for the engine.
        assert assistant_turn.tool_calls[0]["function"]["arguments"] == {"city": "Madrid"}
        tool_turn = sent[2]
        assert tool_turn.content == "22C sunny"
        assert tool_turn.tool_call_id == "call_1"

    def test_empty_tools_does_not_break_plain_chat(self):
        """T4: tools=[] is treated as no tools; ordinary content is returned."""
        engine = MagicMock()
        engine.chat.return_value = _result("hello there")
        client = self._wire(engine)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-32b",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
            },
        )
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "hello there"
        # tools=[] collapses to None — engine called without a tools payload.
        assert engine.chat.call_args.kwargs.get("tools") is None

    def test_streaming_emits_tool_calls_in_final_delta(self):
        """T5: streaming buffers, then emits a structured tool_calls delta."""
        engine = MagicMock()
        engine.chat_stream.return_value = iter(
            [
                "<tool_call>\n",
                '{"name": "write_wiki", "arguments": {"path": "topics/hello.md"}}',
                "\n</tool_call>",
            ]
        )
        client = self._wire(engine)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-32b",
                "stream": True,
                "messages": [{"role": "user", "content": "Save Hello at topics/hello.md"}],
                "tools": [WRITE_WIKI_TOOL],
            },
        )
        assert resp.status_code == 200
        # Collect the streamed chunks (skip the [DONE] sentinel).
        chunks = []
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                continue
            chunks.append(json.loads(payload))

        # The raw <tool_call> marker never leaked as a content delta.
        for c in chunks:
            delta = c["choices"][0]["delta"]
            assert "tool_call" not in (delta.get("content") or "")

        tool_deltas = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        assert tool_deltas, "expected a tool_calls delta in the stream"
        tc = tool_deltas[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "write_wiki"
        assert isinstance(tc["function"]["arguments"], str)
        assert tc["index"] == 0

        finishes = [c["choices"][0]["finish_reason"] for c in chunks]
        assert "tool_calls" in finishes

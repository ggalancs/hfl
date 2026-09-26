# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A model's reasoning never reaches the client as its answer.

Measured with Qwen3-14B (2026-09-26): the Anthropic, OpenAI-stream and
Responses routes sent the whole ``<think>`` block as the reply's text; only
Ollama's ``/api/chat`` kept it apart. Each route now sends it where its own
API puts reasoning — Anthropic ``thinking`` blocks (when thinking is
enabled), OpenAI ``reasoning_content``, a Responses ``reasoning`` item — or
not at all.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state
from hfl.api.thinking import split_reasoning

# A reply streamed with its markers cut across chunks, as tokens cut them.
CHUNKS = ["<thi", "nk>\nseventeen", " times 23</th", "ink>\n\n", "39", "1"]
REPLY = "".join(CHUNKS)
USER = [{"role": "user", "content": "17*23?"}]


@pytest.fixture
def engine():
    state = get_state()
    fake = MagicMock()
    fake.is_loaded = True
    fake.chat.return_value = MagicMock(
        text=REPLY, tokens_generated=9, tokens_prompt=5, stop_reason="stop", tool_calls=None
    )
    fake.chat_stream.side_effect = lambda *a, **k: iter(CHUNKS)
    model = MagicMock()
    model.name = "m"
    state.engine, state.current_model, state.api_key = fake, model, None
    yield fake
    state.engine = state.current_model = None


def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]


class TestSplit:
    @pytest.mark.parametrize(
        ("text", "answer", "reasoning"),
        [
            (REPLY, "391", "seventeen times 23"),
            ("<think>cut off by the token cap", "", "cut off by the token cap"),
            ("opened by the template</think>\n391", "391", "opened by the template"),
            ("391, no reasoning <b>at all</b>", "391, no reasoning <b>at all</b>", None),
        ],
    )
    def test_cases(self, text, answer, reasoning):
        assert split_reasoning(text) == (answer, reasoning)


class TestAnthropic:
    def _post(self, **extra):
        body = {"model": "m", "max_tokens": 100, "messages": USER, **extra}
        return TestClient(app).post("/v1/messages", json=body)

    def test_thinking_off_the_answer_alone(self, engine):
        content = self._post().json()["content"]
        assert content == [{"type": "text", "text": "391"}]

    def test_thinking_enabled_a_thinking_block_first(self, engine):
        content = self._post(thinking={"type": "enabled", "budget_tokens": 1024}).json()["content"]
        assert [b["type"] for b in content] == ["thinking", "text"]
        assert content[0]["thinking"] == "seventeen times 23" and content[1]["text"] == "391"

    def _stream(self, **extra):
        events = _events(self._post(stream=True, **extra).text)
        blocks: dict[int, dict] = {}
        for ev in events:
            if ev["type"] == "content_block_start":
                assert ev["index"] not in blocks  # each index opened once
                blocks[ev["index"]] = {"type": ev["content_block"]["type"], "text": "", "open": 1}
            elif ev["type"] == "content_block_delta":
                block, delta = blocks[ev["index"]], ev["delta"]
                assert block["open"] == 1  # never written after its stop
                block["text"] += "".join(
                    delta.get(k) or "" for k in ("text", "thinking", "partial_json")
                )
            elif ev["type"] == "content_block_stop":
                blocks[ev["index"]]["open"] -= 1
        assert all(b["open"] == 0 for b in blocks.values())  # all closed once
        return [(blocks[i]["type"], blocks[i]["text"]) for i in sorted(blocks)]

    def test_stream_thinking_off(self, engine):
        assert self._stream() == [("text", "391")]

    def test_stream_thinking_enabled(self, engine):
        blocks = self._stream(thinking={"type": "enabled", "budget_tokens": 1024})
        assert blocks == [("thinking", "seventeen times 23"), ("text", "391")]

    def test_stream_all_reasoning_still_has_a_text_block(self, engine):
        engine.chat_stream.side_effect = lambda *a, **k: iter(["<think>", "cut off"])
        blocks = self._stream(thinking={"type": "enabled", "budget_tokens": 1024})
        assert blocks == [("thinking", "cut off"), ("text", "")]

    def test_stream_tool_call_after_reasoning(self, engine):
        call = '<tool_call>\n{"name": "calc", "arguments": {"x": "17*23"}}\n</tool_call>'
        engine.chat_stream.side_effect = lambda *a, **k: iter(["<think>use calc</think>", call])
        tools = [{"name": "calc", "input_schema": {"type": "object"}}]
        blocks = self._stream(tools=tools, thinking={"type": "enabled", "budget_tokens": 1024})
        assert blocks == [("thinking", "use calc"), ("tool_use", '{"x": "17*23"}')]

    def test_a_thinking_block_sent_back_is_accepted_and_ignored(self, engine):
        history = [
            *USER,
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "seventeen times 23", "signature": ""},
                    {"type": "text", "text": "391"},
                ],
            },
            {"role": "user", "content": "and 18*23?"},
        ]
        response = TestClient(app).post(
            "/v1/messages", json={"model": "m", "max_tokens": 100, "messages": history}
        )
        assert response.status_code == 200
        sent = engine.chat.call_args.args[0]
        assert [m.content for m in sent if m.role == "assistant"] == ["391"]


class TestOpenAI:
    def _post(self, **extra):
        body = {"model": "m", "messages": USER, **extra}
        return TestClient(app).post("/v1/chat/completions", json=body)

    def test_reasoning_content_beside_the_answer(self, engine):
        message = self._post().json()["choices"][0]["message"]
        assert message["content"] == "391"
        assert message["reasoning_content"] == "seventeen times 23"

    def test_reasoning_effort_none_no_reasoning(self, engine):
        message = self._post(reasoning_effort="none").json()["choices"][0]["message"]
        assert message["content"] == "391" and "reasoning_content" not in message

    def test_stream(self, engine):
        deltas = [
            e["choices"][0]["delta"] for e in _events(self._post(stream=True).text) if e["choices"]
        ]
        content = "".join(d.get("content") or "" for d in deltas)
        reasoning = "".join(d.get("reasoning_content") or "" for d in deltas)
        assert (content, reasoning) == ("391", "seventeen times 23")
        assert [d.get("role") for d in deltas if d.get("role")] == ["assistant"]

    def test_stream_reasoning_effort_none(self, engine):
        body = self._post(stream=True, reasoning_effort="none").text
        assert "reasoning_content" not in body and "seventeen" not in body


class TestResponses:
    def _post(self, **extra):
        body = {"model": "m", "input": "17*23?", **extra}
        return TestClient(app).post("/v1/responses", json=body)

    def test_a_reasoning_item_then_the_message(self, engine):
        output = self._post().json()["output"]
        assert [i["type"] for i in output] == ["reasoning", "message"]
        assert output[0]["summary"][0]["text"] == "seventeen times 23"
        assert output[1]["content"][0]["text"] == "391"

    def test_effort_none_no_reasoning(self, engine):
        output = self._post(reasoning={"effort": "none"}).json()["output"]
        assert [i["type"] for i in output] == ["message"]
        assert output[0]["content"][0]["text"] == "391"
        assert engine.chat.call_args.args[1].reasoning == "off"  # reaches the prompt

    def test_stream(self, engine):
        events = _events(self._post(stream=True).text)
        added = [
            (e["output_index"], e["item"]["type"])
            for e in events
            if e["type"] == "response.output_item.added"
        ]
        assert added == [(0, "reasoning"), (1, "message")]
        text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        summary = "".join(
            e["delta"] for e in events if e["type"] == "response.reasoning_summary_text.delta"
        )
        assert (text, summary) == ("391", "seventeen times 23")
        done = next(e for e in events if e["type"] == "response.completed")["response"]["output"]
        assert [i["type"] for i in done] == ["reasoning", "message"]


class TestOllama:
    def _post(self, **extra):
        body = {"model": "m", "messages": USER, "stream": False, **extra}
        return TestClient(app).post("/api/chat", json=body).json()["message"]

    def test_think_the_reasoning_apart(self, engine):
        message = self._post(think=True)
        assert (message["content"], message["thinking"]) == ("391", "seventeen times 23")

    def test_cut_short_mid_thought_the_reasoning_is_kept(self, engine):
        engine.chat.return_value.text = "<think>cut off by the token cap"
        message = self._post(think=True)
        assert (message["content"], message["thinking"]) == ("", "cut off by the token cap")

    def test_without_think_none(self, engine):
        message = self._post()
        assert message["content"] == "391" and "thinking" not in message

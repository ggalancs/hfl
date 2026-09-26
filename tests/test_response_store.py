# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``previous_response_id`` on ``/v1/responses``: a conversation the server
keeps, told apart from one it lost.

Checked for real (2026-09-26) with the official ``openai`` SDK and
Qwen2.5-0.5B-Instruct: a word given in one turn was answered two turns
later (streamed and not), a tool result sent with only
``previous_response_id`` was answered from, and ``store: false``, an
unknown id and a restarted server all gave ``previous_response_not_found``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.response_store import ResponseStore, get_response_store
from hfl.api.server import app
from hfl.api.state import get_state
from hfl.engine.base import ChatMessage


def _m(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class TestStore:
    def test_a_chain_oldest_first(self):
        store = ResponseStore()
        store.put("r1", None, [_m("user", "a"), _m("assistant", "b")])
        store.put("r2", "r1", [_m("user", "c"), _m("assistant", "d")])
        assert [m.content for m in store.history("r2")] == ["a", "b", "c", "d"]

    def test_a_lost_link_is_not_a_shorter_history(self):
        store = ResponseStore(max_turns=2)
        store.put("r1", None, [_m("user", "a")])
        store.put("r2", "r1", [_m("user", "b")])
        store.put("r3", "r2", [_m("user", "c")])  # r1 evicted
        assert store.history("r3") is None
        assert store.history("nope") is None

    def test_the_oldest_unused_go_first(self):
        store = ResponseStore(max_turns=2)
        store.put("r1", None, [_m("user", "a")])
        store.put("r2", None, [_m("user", "b")])
        store.history("r1")  # used: kept
        store.put("r3", None, [_m("user", "c")])
        assert store.history("r1") is not None and store.history("r2") is None

    def test_what_is_kept_cannot_be_edited_through_a_history(self):
        store = ResponseStore()
        store.put("r1", None, [ChatMessage(role="assistant", content="", tool_calls=[{"id": "c"}])])
        store.history("r1")[0].tool_calls.append({"id": "x"})
        assert store.history("r1")[0].tool_calls == [{"id": "c"}]


@pytest.fixture
def engine():
    get_response_store().clear()
    state = get_state()
    fake = MagicMock()
    fake.is_loaded = True
    replies = iter(["first", "second", "third"])
    fake.chat.side_effect = lambda messages, cfg, tools=None: MagicMock(
        text=next(replies), tokens_prompt=3, tokens_generated=1, tool_calls=None
    )
    model = MagicMock()
    model.name = "m"
    state.engine, state.current_model, state.api_key = fake, model, None
    yield fake
    state.engine = state.current_model = None
    get_response_store().clear()


def _post(**body):
    return TestClient(app).post("/v1/responses", json={"model": "m", **body})


def _sent(engine, call=-1):
    return [(m.role, m.content) for m in engine.chat.call_args_list[call].args[0]]


def test_the_conversation_is_supplied_without_the_old_instructions(engine):
    first = _post(instructions="Be terse.", input="the word is tangerine").json()
    second = _post(instructions="Be kind.", input="which word?", previous_response_id=first["id"])
    assert second.json()["previous_response_id"] == first["id"]
    assert _sent(engine) == [
        ("system", "Be kind."),
        ("user", "the word is tangerine"),
        ("assistant", "first"),
        ("user", "which word?"),
    ]


def test_store_false_and_unknown_ids_are_not_found(engine):
    lone = _post(input="x", store=False).json()
    assert lone["store"] is False
    for previous in (lone["id"], "resp_nope"):
        response = _post(input="y", previous_response_id=previous)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "previous_response_not_found"


def test_a_tool_result_after_previous_response_id_finds_its_call(engine):
    call = {"id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    engine.chat.side_effect = [
        MagicMock(text="", tokens_prompt=3, tokens_generated=1, tool_calls=[call]),
        MagicMock(text="31C", tokens_prompt=3, tokens_generated=1, tool_calls=None),
    ]
    tools = [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]
    first = _post(input="weather?", tools=tools).json()
    call_id = next(o["call_id"] for o in first["output"] if o["type"] == "function_call")
    result = {"type": "function_call_output", "call_id": call_id, "output": json.dumps({"t": 31})}
    _post(input=[result], tools=tools, previous_response_id=first["id"])
    sent = engine.chat.call_args_list[-1].args[0]
    assert sent[1].tool_calls[0]["id"] == call_id
    assert (sent[2].role, sent[2].tool_call_id, sent[2].name) == ("tool", call_id, "get_weather")


def test_a_streamed_response_is_kept_too(engine):
    engine.chat_stream.side_effect = lambda *a, **k: iter(["stre", "amed"])
    body = (
        TestClient(app)
        .post("/v1/responses", json={"model": "m", "input": "hi", "stream": True})
        .text
    )
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
    done = next(e for e in events if e["type"] == "response.completed")["response"]
    _post(input="again", previous_response_id=done["id"])
    assert _sent(engine) == [("user", "hi"), ("assistant", "streamed"), ("user", "again")]

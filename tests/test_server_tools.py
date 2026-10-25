# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Anthropic's web search server tool, run by HFL (hfl.api.server_tools):
Claude Code's WebSearch on a local model."""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api import server_tools as st
from hfl.api.schemas.anthropic import AnthropicMessagesRequest
from hfl.engine.base import ChatMessage
from hfl.tools.web_search import WebSearchUpstreamError

SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
RESULTS = {
    "results": [
        {
            "title": "Python Releases",
            "url": "https://www.python.org/downloads/",
            "content": "Python 3.15.1",
        },
        {"title": "Blocked", "url": "https://spam.example/x", "content": "spam"},
    ],
    "backend": "searxng",
}


def _tools(*extra: dict) -> list:
    return AnthropicMessagesRequest(
        model="m", max_tokens=10, messages=[{"role": "user", "content": "q"}], tools=[*extra]
    ).tools


def _reply(text: str = "", calls: list[dict] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text, tool_calls=calls, tokens_prompt=10, tokens_generated=4, stop_reason="stop"
    )


def _call(name: str, args: dict) -> dict:
    return {"function": {"name": name, "arguments": args}}


def _resolve(result, offered):
    calls = result.tool_calls or []
    return SimpleNamespace(content=result.text, tool_calls=calls, has_tool_calls=bool(calls))


def _run(tool, replies, search, tools=None):
    seen: list[list[ChatMessage]] = []
    queue = list(replies)
    _run.offered = []  # type: ignore[attr-defined]

    async def chat(history, offered):
        seen.append(list(history))
        _run.offered.append([t["function"]["name"] for t in offered])  # type: ignore[attr-defined]
        return queue.pop(0)

    turn = asyncio.run(
        st.run(
            tool,
            [ChatMessage(role="user", content="latest Python?")],
            tools,
            chat=chat,
            resolve=_resolve,
            to_tool_use=lambda calls: [
                {"type": "tool_use", "name": st._call_name(c)} for c in calls
            ],
            stop_reason=lambda r: "end_turn",
            search=search,
        )
    )
    return turn, seen


async def _found(query, n):
    return RESULTS


class TestTheTool:
    def test_found_among_the_tools_with_its_limits(self):
        tool = st.web_search_tool(
            _tools(
                {"name": "Read", "input_schema": {"type": "object"}},
                {**SEARCH_TOOL, "max_uses": 99, "blocked_domains": ["Spam.example"]},
            )
        )
        assert tool is not None and tool.name == "web_search"
        assert tool.max_uses == st.MAX_USES_CAP and tool.blocked_domains == ["spam.example"]

    def test_none_without_it(self):
        assert st.web_search_tool(_tools({"name": "Read"})) is None
        assert st.web_search_tool(None) is None

    def test_not_offered_to_the_client_as_a_function(self):
        from hfl.api.routes_anthropic import _anthropic_tools_to_payload

        req = AnthropicMessagesRequest(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "q"}],
            tools=[SEARCH_TOOL, {"name": "Read", "input_schema": {"type": "object"}}],
        )  # fmt: skip
        assert [t["function"]["name"] for t in _anthropic_tools_to_payload(req)] == ["Read"]
        only = AnthropicMessagesRequest(
            model="m",
            max_tokens=10,
            messages=[{"role": "user", "content": "q"}],
            tools=[SEARCH_TOOL],
        )
        assert _anthropic_tools_to_payload(only) is None


class TestTheTurn:
    def test_searches_then_answers_with_anthropics_blocks(self):
        tool = st.WebSearchTool(name="web_search", blocked_domains=["spam.example"])
        turn, seen = _run(
            tool,
            [_reply("Let me search.", [_call("web_search", {"query": "python release"})]),
             _reply("Python 3.15.1.")],
            _found,
        )  # fmt: skip
        kinds = [b["type"] for b in turn.blocks]
        assert kinds == ["text", "server_tool_use", "web_search_tool_result", "text"]
        use, result = turn.blocks[1], turn.blocks[2]
        assert use["input"] == {"query": "python release"} and use["id"].startswith("srvtoolu_")
        assert result["tool_use_id"] == use["id"]
        assert [r["url"] for r in result["content"]] == ["https://www.python.org/downloads/"]
        assert st._snippet(result["content"][0]["encrypted_content"]) == "Python 3.15.1"
        assert turn.usage["server_tool_use"] == {"web_search_requests": 1}
        assert turn.stop_reason == "end_turn"
        # The model got the results, not the spam the client filtered out.
        told = seen[1][-1]
        assert (
            told.role == "tool" and "Python 3.15.1" in told.content and "spam" not in told.content
        )

    def test_no_search_available_means_no_answer(self):
        """From memory, with sources it made up, is worse than nothing."""

        async def refuse(query, n):
            raise WebSearchUpstreamError("DuckDuckGo refused the search")

        turn, seen = _run(
            st.WebSearchTool(name="web_search"),
            [_reply("", [_call("web_search", {"query": "python"})]), _reply("Python 3.13.0 [1]")],
            refuse,
        )
        assert [b["type"] for b in turn.blocks] == ["server_tool_use", "web_search_tool_result"]
        assert turn.blocks[1]["content"] == {
            "type": "web_search_tool_result_error",
            "error_code": "unavailable",
        }
        assert len(seen) == 1  # the model never got to answer

    def test_past_max_uses_the_model_is_told_and_answers(self):
        searched: list[str] = []

        async def search(query, n):
            searched.append(query)
            return RESULTS

        turn, seen = _run(
            st.WebSearchTool(name="web_search", max_uses=1),
            [_reply("", [_call("web_search", {"query": "a"})]),
             _reply("", [_call("web_search", {"query": "b"})]),
             _reply("done")],
            search,
        )  # fmt: skip
        assert searched == ["a"]
        # Spent: the second call no longer offers the search.
        assert _run.offered[0] == ["web_search"] and _run.offered[1] == []  # type: ignore[attr-defined]
        errors = [b["content"] for b in turn.blocks if b["type"] == "web_search_tool_result"]
        assert errors[1] == {
            "type": "web_search_tool_result_error",
            "error_code": "max_uses_exceeded",
        }
        assert turn.blocks[-1] == {"type": "text", "text": "done"}

    def test_the_clients_own_tool_ends_the_turn(self):
        turn, _ = _run(
            st.WebSearchTool(name="web_search"),
            [_reply("", [_call("web_search", {"query": "a"}), _call("Read", {"path": "x"})])],
            _found,
        )
        assert [b["type"] for b in turn.blocks][-1] == "tool_use"
        assert turn.stop_reason == "tool_use"

    def test_arguments_as_a_json_string(self):
        turn, _ = _run(
            st.WebSearchTool(name="web_search"),
            [_reply("", [_call("web_search", '{"query": "q1"}')]), _reply("ok")],
            _found,
        )
        assert turn.blocks[0]["input"] == {"query": "q1"}


class TestHistory:
    def test_earlier_blocks_reach_the_model_as_text(self):
        from hfl.api.routes_anthropic import _request_to_messages

        result = {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {"type": "web_search_result", "url": "https://p.org", "title": "P",
                 "encrypted_content": st._opaque("snippet text")}
            ],
        }  # fmt: skip
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=10,
            messages=[
                {"role": "user", "content": "latest?"},
                {"role": "assistant", "content": [
                    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
                     "input": {"query": "latest python"}},
                    result,
                    {"type": "text", "text": "It is 3.15.1."}]},
                {"role": "user", "content": "thanks"},
            ],
        )  # fmt: skip
        said = _request_to_messages(req)[1].content
        assert "Searched the web for: latest python" in said
        assert "P (https://p.org): snippet text" in said and "It is 3.15.1." in said

    def test_an_error_block_and_a_foreign_one(self):
        error = SimpleNamespace(
            type="web_search_tool_result", content={"error_code": "unavailable"}
        )
        assert st.history_text(error) == "(The web search failed: unavailable)"
        assert st.history_text(SimpleNamespace(type="text")) is None
        assert st._snippet("not base64 json") == ""


def test_the_stream_has_anthropics_events():
    message = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
        "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "text", "text": "answer"},
        ],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 2,
                  "server_tool_use": {"web_search_requests": 1}},
    }  # fmt: skip
    events = st.sse_events(message)
    names = [e for e, _ in events]
    assert names == [
        "message_start",
        "content_block_start", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "message_delta", "message_stop",
    ]  # fmt: skip
    assert events[1][1]["content_block"]["input"] == {}
    assert json.loads(events[2][1]["delta"]["partial_json"]) == {"query": "q"}
    assert events[4][1]["content_block"]["type"] == "web_search_tool_result"
    assert events[-2][1]["usage"]["server_tool_use"] == {"web_search_requests": 1}


class TestTheRoute:
    @pytest.fixture
    def engine(self):
        from hfl.api.state import get_state

        engine = MagicMock()
        engine.is_loaded = True
        state = get_state()
        state.api_key = None
        state.engine = engine
        state.current_model = SimpleNamespace(name="qwen")
        yield engine
        state.engine = None
        state.current_model = None

    @pytest.mark.parametrize("stream", [False, True])
    def test_claude_codes_websearch_request(self, engine, monkeypatch, stream):
        import hfl.tools.web_search as ws

        engine.chat.side_effect = [
            _reply("", [_call("web_search", {"query": "latest python"})]),
            _reply("Python 3.15.1."),
        ]

        async def search(query, max_results=5):
            return RESULTS

        monkeypatch.setattr(ws, "search", search)
        body = {
            "model": "qwen", "max_tokens": 200, "stream": stream, "tools": [SEARCH_TOOL],
            "messages": [{"role": "user", "content": "Perform a web search for: latest python"}],
        }  # fmt: skip
        response = TestClient(app_()).post("/v1/messages", json=body)
        assert response.status_code == 200, response.text
        if stream:
            assert "event: content_block_start" in response.text
            assert '"type": "web_search_tool_result"' in response.text
            assert "Python 3.15.1." in response.text
        else:
            data = response.json()
            assert [b["type"] for b in data["content"]] == [
                "server_tool_use", "web_search_tool_result", "text",
            ]  # fmt: skip
            assert data["usage"]["server_tool_use"]["web_search_requests"] == 1
        # The model was offered the search as a function with a query.
        offered = engine.chat.call_args_list[0].kwargs["tools"]
        assert offered[-1]["function"]["parameters"]["required"] == ["query"]


def app_():
    from hfl.api.server import app

    return app


def test_opaque_is_base64_json():
    assert json.loads(base64.b64decode(st._opaque("x"))) == {"snippet": "x"}

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.api.agent_loop`` (Phase 10 P1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.agent_loop import DEFAULT_MAX_ITERATIONS, run_agent_loop
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult
from hfl.mcp import client as mcp
from hfl.models.manifest import ModelManifest


@pytest.fixture(autouse=True)
def _fresh_mcp():
    mcp.reset_client()
    yield
    mcp.reset_client()


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


# ----------------------------------------------------------------------
# Unit tests — direct run_agent_loop
# ----------------------------------------------------------------------


def _mk_result(text: str = "", tool_calls=None) -> GenerationResult:
    return GenerationResult(
        text=text,
        tokens_generated=1,
        tokens_prompt=1,
        tool_calls=list(tool_calls) if tool_calls else None,
    )


class TestAgentLoopDirect:
    async def test_single_turn_no_tool_calls(self):
        calls = []

        async def _chat(msgs, cfg, tools):
            calls.append(len(msgs))
            return _mk_result(text="final")

        result = await run_agent_loop(
            messages=[ChatMessage(role="user", content="hi")],
            config=GenerationConfig(),
            tools=None,
            chat_fn=_chat,
        )
        assert result.terminated_by == "final_answer"
        assert result.iterations == 1
        assert result.final.text == "final"
        assert result.tool_trace == []
        assert calls == [1]

    async def test_two_hop_loop_dispatches_tool(self):
        turns = iter(
            [
                _mk_result(
                    text="calling",
                    tool_calls=[
                        {
                            "id": "c1",
                            "function": {"name": "fs__echo", "arguments": {"m": "x"}},
                        }
                    ],
                ),
                _mk_result(text="done"),
            ]
        )

        async def _chat(msgs, cfg, tools):
            return next(turns)

        async def _mcp(name, args):
            return {"ok": True, "name": name, "args": args}

        result = await run_agent_loop(
            messages=[ChatMessage(role="user", content="do it")],
            config=GenerationConfig(),
            tools=[{"type": "function", "function": {"name": "fs__echo"}}],
            chat_fn=_chat,
            mcp_caller=_mcp,
        )
        assert result.iterations == 2
        assert result.terminated_by == "final_answer"
        assert result.final.text == "done"
        assert [e.name for e in result.tool_trace] == ["fs__echo"]
        assert result.tool_trace[0].error is None

    async def test_max_iterations_cap(self):
        async def _chat(msgs, cfg, tools):
            return _mk_result(
                text="busy",
                tool_calls=[
                    {
                        "id": "c",
                        "function": {"name": "fs__ping", "arguments": {}},
                    }
                ],
            )

        async def _mcp(name, args):
            return "pong"

        result = await run_agent_loop(
            messages=[ChatMessage(role="user", content="go")],
            config=GenerationConfig(),
            tools=[{"type": "function", "function": {"name": "fs__ping"}}],
            chat_fn=_chat,
            mcp_caller=_mcp,
            max_iterations=3,
        )
        assert result.terminated_by == "max_iterations"
        assert result.iterations == 3
        assert len(result.tool_trace) == 3

    async def test_unknown_tool_becomes_error_entry(self):
        async def _chat(msgs, cfg, tools):
            return _mk_result(
                text="",
                tool_calls=[{"id": "c", "function": {"name": "builtin_adder", "arguments": {}}}],
            )

        result = await run_agent_loop(
            messages=[ChatMessage(role="user", content="go")],
            config=GenerationConfig(),
            tools=None,
            chat_fn=_chat,
            mcp_caller=None,
            max_iterations=1,
        )
        assert result.tool_trace[0].error == "no handler registered for this tool"

    async def test_malformed_json_arguments_captured_as_error(self):
        async def _chat(msgs, cfg, tools):
            return _mk_result(
                text="",
                tool_calls=[{"id": "c", "function": {"name": "fs__x", "arguments": "{not-json"}}],
            )

        async def _mcp(name, args):
            return "ok"

        result = await run_agent_loop(
            messages=[ChatMessage(role="user", content="go")],
            config=GenerationConfig(),
            tools=None,
            chat_fn=_chat,
            mcp_caller=_mcp,
            max_iterations=1,
        )
        assert result.tool_trace[0].error == "malformed tool-call JSON"

    async def test_parallel_calls_fan_out(self):
        import asyncio

        async def _chat(msgs, cfg, tools):
            # Only emit tool_calls on the first turn; terminate on second.
            if not any(m.role == "tool" for m in msgs):
                return _mk_result(
                    text="parallel",
                    tool_calls=[
                        {"id": "a", "function": {"name": "fs__a", "arguments": {}}},
                        {"id": "b", "function": {"name": "fs__b", "arguments": {}}},
                    ],
                )
            return _mk_result(text="done")

        in_flight = {"count": 0, "max": 0}

        async def _mcp(name, args):
            in_flight["count"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["count"])
            await asyncio.sleep(0.01)
            in_flight["count"] -= 1
            return {"name": name}

        result = await run_agent_loop(
            messages=[ChatMessage(role="user", content="go")],
            config=GenerationConfig(),
            tools=None,
            chat_fn=_chat,
            mcp_caller=_mcp,
        )
        assert in_flight["max"] == 2  # both tools in flight simultaneously
        assert result.iterations == 2


# ----------------------------------------------------------------------
# Route integration
# ----------------------------------------------------------------------


def _install(manifest, *, results: list[GenerationResult]):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    # Each call returns the next result in the sequence.
    engine.chat = MagicMock(side_effect=results)
    state.engine = engine
    state.current_model = manifest
    return engine


def _mk_mcp_tool():
    c = mcp.get_client()
    tool = mcp.MCPTool(
        server_id="fs",
        name="echo",
        description="echo",
        input_schema={"type": "object"},
    )
    conn = mcp._ServerConnection(
        server_id="fs",
        transport="stdio",
        target="stdio://fake",
        session=MagicMock(),
        tools=[tool],
    )

    # call_tool is awaited — use an async mock by attaching an
    # AsyncMock directly on the connection's session.
    async def _call(name, args):
        return {"called": name, "args": args}

    conn.session.call_tool = _call  # type: ignore[assignment]
    c._servers["fs"] = conn


class TestChatRouteAgentLoop:
    def test_agent_loop_runs_two_hops_and_records_trace(self, client):
        _mk_mcp_tool()
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        # First call returns a tool_call; second returns the final
        # assistant reply.
        _install(
            manifest,
            results=[
                GenerationResult(
                    text="calling echo",
                    tokens_generated=1,
                    tokens_prompt=1,
                    tool_calls=[
                        {
                            "id": "c1",
                            "function": {
                                "name": "fs__echo",
                                "arguments": {"msg": "hi"},
                            },
                        }
                    ],
                ),
                GenerationResult(
                    text="all done",
                    tokens_generated=1,
                    tokens_prompt=1,
                ),
            ],
        )

        resp = client.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "agent_loop": True,
                "max_iterations": 4,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"]["content"] == "all done"
        assert body["tool_trace"]
        assert body["tool_trace"][0]["name"] == "fs__echo"
        assert body["tool_trace"][0]["error"] is None

    def test_agent_loop_false_preserves_legacy_behavior(self, client):
        manifest = ModelManifest(
            name="m",
            repo_id="org/m",
            local_path="/tmp/x.gguf",
            format="gguf",
        )
        _install(
            manifest,
            results=[GenerationResult(text="x", tokens_generated=1, tokens_prompt=1)],
        )
        resp = client.post(
            "/api/chat",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert "tool_trace" not in resp.json()


class TestDefaults:
    def test_default_max_iterations_is_6(self):
        assert DEFAULT_MAX_ITERATIONS == 6

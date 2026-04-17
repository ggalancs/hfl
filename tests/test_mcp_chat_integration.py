# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for MCP tools being merged into /api/chat.

The helper under test is ``_merge_mcp_tools`` in routes_native.py.
We exercise it directly (unit tests) plus the full route (integration)
to confirm MCP tools surface in the tools array passed to the engine.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.routes_native import _merge_mcp_tools
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationResult
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


def _install_engine(name="m", *, result: GenerationResult):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.chat = MagicMock(return_value=result)
    state.engine = engine
    state.current_model = ModelManifest(
        name=name,
        repo_id=f"org/{name}",
        local_path="/tmp/fake.gguf",
        format="gguf",
    )
    return engine


def _register_fake_mcp_tool():
    c = mcp.get_client()
    tool = mcp.MCPTool(
        server_id="fs",
        name="read",
        description="Read a file",
        input_schema={"type": "object"},
    )
    conn = mcp._ServerConnection(
        server_id="fs",
        transport="stdio",
        target="stdio://fake",
        session=MagicMock(),
        tools=[tool],
    )
    c._servers["fs"] = conn


# ----------------------------------------------------------------------
# Unit tests for the merge helper
# ----------------------------------------------------------------------


class TestMergeMCPTools:
    def test_no_mcp_no_existing_returns_none(self):
        assert _merge_mcp_tools(None) is None

    def test_no_mcp_with_existing_is_passthrough(self):
        existing = [{"type": "function", "function": {"name": "a"}}]
        assert _merge_mcp_tools(existing) == existing

    def test_mcp_and_no_existing_surfaces_mcp(self):
        _register_fake_mcp_tool()
        out = _merge_mcp_tools(None)
        assert out is not None
        assert len(out) == 1
        assert out[0]["function"]["name"] == "fs__read"

    def test_mcp_and_existing_concatenates(self):
        _register_fake_mcp_tool()
        existing = [{"type": "function", "function": {"name": "a"}}]
        out = _merge_mcp_tools(existing)
        assert len(out) == 2
        names = [t["function"]["name"] for t in out]
        assert names == ["a", "fs__read"]


# ----------------------------------------------------------------------
# End-to-end /api/chat integration
# ----------------------------------------------------------------------


class TestChatRouteSeesMCPTools:
    def test_mcp_tools_propagate_to_engine(self, client):
        _register_fake_mcp_tool()
        engine = _install_engine(
            result=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1),
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
        # The engine.chat call must include the MCP-provided tool.
        _, kwargs = engine.chat.call_args
        tools_arg = (
            engine.chat.call_args[0][2]
            if len(engine.chat.call_args[0]) > 2
            else kwargs.get("tools")
        )
        assert tools_arg is not None
        names = [t["function"]["name"] for t in tools_arg]
        assert "fs__read" in names

    def test_no_mcp_leaves_tools_unchanged(self, client):
        engine = _install_engine(
            result=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1),
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
        # With no MCP connected, tools should remain None.
        tools_arg = (
            engine.chat.call_args[0][2]
            if len(engine.chat.call_args[0]) > 2
            else engine.chat.call_args[1].get("tools")
        )
        assert tools_arg is None

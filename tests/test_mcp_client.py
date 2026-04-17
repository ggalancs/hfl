# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.mcp.client`` without requiring the real MCP SDK.

The MCP Python SDK is an optional dependency. These tests monkey-
patch the module's SDK loader so we can exercise the client's
behaviour on any environment — including CI runs without ``mcp``
installed. A second, SDK-gated test class checks the fallback path
when the SDK is genuinely absent.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hfl.mcp import client as mcp


@pytest.fixture(autouse=True)
def _fresh_client():
    mcp.reset_client()
    yield
    mcp.reset_client()


def _install_fake_sdk(monkeypatch):
    """Replace ``_require_sdk`` with a fake that returns controllable stand-ins.

    Returns the captured state so the test body can inspect calls.
    """
    state: dict = {
        "initialized": False,
        "listed_tools": [],
        "call_results": [],
        "closed": False,
    }

    class _Session:
        def __init__(self, read, write):
            state["session_args"] = (read, write)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            state["closed"] = True

        async def initialize(self):
            state["initialized"] = True

        async def list_tools(self):
            tool = SimpleNamespace(
                name="echo",
                description="echo tool",
                inputSchema={"type": "object", "properties": {"msg": {"type": "string"}}},
            )
            state["listed_tools"].append(tool)
            return SimpleNamespace(tools=[tool])

        async def call_tool(self, name, arguments):
            state["call_results"].append((name, arguments))
            return {"ok": True, "name": name, "arguments": arguments}

    class _StdioTransportCM:
        async def __aenter__(self):
            return ("read_stream", "write_stream")

        async def __aexit__(self, *_):
            return False

    def _stdio_client(params):  # noqa: ARG001
        return _StdioTransportCM()

    def _sse_client(url):  # noqa: ARG001
        return _StdioTransportCM()

    def fake_require_sdk():
        return {
            "ClientSession": _Session,
            "StdioServerParameters": MagicMock,
            "stdio_client": _stdio_client,
            "sse_client": _sse_client,
        }

    monkeypatch.setattr(mcp.MCPClient, "_require_sdk", staticmethod(fake_require_sdk))
    return state


# ----------------------------------------------------------------------
# SDK-unavailable fallback
# ----------------------------------------------------------------------


class TestSDKUnavailable:
    async def test_connect_raises_when_sdk_missing(self, monkeypatch):
        # Force _require_sdk to raise as if mcp was not installed.
        def _missing():
            raise mcp.MCPClientUnavailableError("SDK not present")

        monkeypatch.setattr(mcp.MCPClient, "_require_sdk", staticmethod(_missing))
        client = mcp.get_client()
        with pytest.raises(mcp.MCPClientUnavailableError):
            await client.connect("x", "stdio://echo")

    async def test_autoload_skips_silently_when_sdk_missing(self, monkeypatch, tmp_path):
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"servers": [{"id": "a", "target": "stdio://echo"}]}')

        def _missing():
            raise mcp.MCPClientUnavailableError("no SDK")

        monkeypatch.setattr(mcp.MCPClient, "_require_sdk", staticmethod(_missing))
        loaded = await mcp.autoload_servers(cfg)
        assert loaded == []


# ----------------------------------------------------------------------
# Connection / tool listing
# ----------------------------------------------------------------------


class TestConnect:
    async def test_stdio_connect_lists_tools(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        client = mcp.get_client()
        tools = await client.connect("fs", "stdio://echo hi")
        assert len(tools) == 1
        assert tools[0].qualified_name == "fs__echo"
        assert tools[0].description == "echo tool"

    async def test_sse_connect_lists_tools(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        client = mcp.get_client()
        tools = await client.connect("remote", "sse://localhost:9000/sse")
        assert tools[0].qualified_name == "remote__echo"

    async def test_unknown_transport_rejected(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        client = mcp.get_client()
        with pytest.raises(mcp.MCPConnectionError):
            await client.connect("bad", "weird://nothing")

    async def test_empty_stdio_target_rejected(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        client = mcp.get_client()
        with pytest.raises(mcp.MCPConnectionError):
            await client.connect("empty", "stdio://")

    async def test_connect_failure_becomes_connection_error(self, monkeypatch):
        _install_fake_sdk(monkeypatch)

        class _ExplodingCM:
            async def __aenter__(self):
                raise RuntimeError("boom inside")

            async def __aexit__(self, *_):
                return False

        def fake_require_sdk():
            return {
                "ClientSession": MagicMock,
                "StdioServerParameters": MagicMock,
                "stdio_client": lambda p: _ExplodingCM(),
                "sse_client": lambda url: _ExplodingCM(),
            }

        monkeypatch.setattr(mcp.MCPClient, "_require_sdk", staticmethod(fake_require_sdk))
        client = mcp.get_client()
        with pytest.raises(mcp.MCPConnectionError):
            await client.connect("bad", "stdio://thing")


# ----------------------------------------------------------------------
# Tool queries + call
# ----------------------------------------------------------------------


class TestToolQueries:
    async def test_list_tools_union_across_servers(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        client = mcp.get_client()
        await client.connect("a", "stdio://one")
        await client.connect("b", "stdio://two")
        qualified = {t.qualified_name for t in client.list_tools()}
        assert qualified == {"a__echo", "b__echo"}

    async def test_qualified_name_prefixes_server(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        client = mcp.get_client()
        await client.connect("fs", "stdio://x")
        tool = client.tool_by_qualified_name("fs__echo")
        assert tool is not None
        assert tool.server_id == "fs"

    async def test_call_tool_roundtrip(self, monkeypatch):
        state = _install_fake_sdk(monkeypatch)
        client = mcp.get_client()
        await client.connect("fs", "stdio://x")
        result = await client.call_tool("fs__echo", {"msg": "hi"})
        assert state["call_results"] == [("echo", {"msg": "hi"})]
        assert result == {"ok": True, "name": "echo", "arguments": {"msg": "hi"}}

    async def test_call_unknown_tool_raises(self, monkeypatch):
        _install_fake_sdk(monkeypatch)
        client = mcp.get_client()
        await client.connect("fs", "stdio://x")
        with pytest.raises(mcp.MCPConnectionError):
            await client.call_tool("fs__nope", {})


# ----------------------------------------------------------------------
# autoload_servers
# ----------------------------------------------------------------------


class TestAutoload:
    async def test_autoload_returns_connected_ids(self, monkeypatch, tmp_path):
        _install_fake_sdk(monkeypatch)
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            '{"servers": [{"id": "a", "target": "stdio://x"},{"id": "b", "target": "sse://y"}]}'
        )
        ids = await mcp.autoload_servers(cfg)
        assert set(ids) == {"a", "b"}

    async def test_autoload_missing_file_is_noop(self, tmp_path):
        loaded = await mcp.autoload_servers(tmp_path / "nope.json")
        assert loaded == []

    async def test_autoload_invalid_json_is_noop(self, tmp_path):
        cfg = tmp_path / "broken.json"
        cfg.write_text("not-json")
        loaded = await mcp.autoload_servers(cfg)
        assert loaded == []

    async def test_autoload_respects_env_var(self, monkeypatch, tmp_path):
        _install_fake_sdk(monkeypatch)
        cfg = tmp_path / "env.json"
        cfg.write_text('{"servers": [{"id": "e", "target": "stdio://z"}]}')
        monkeypatch.setenv("HFL_MCP_AUTOLOAD", str(cfg))
        loaded = await mcp.autoload_servers()
        assert loaded == ["e"]

    async def test_autoload_skips_entries_missing_id_or_target(self, monkeypatch, tmp_path):
        _install_fake_sdk(monkeypatch)
        cfg = tmp_path / "cfg.json"
        cfg.write_text(
            '{"servers": ['
            '{"target": "stdio://x"},'
            '{"id": "ok", "target": "stdio://x"},'
            '{"id": "empty"}'
            "]}"
        )
        loaded = await mcp.autoload_servers(cfg)
        assert loaded == ["ok"]


# ----------------------------------------------------------------------
# MCPTool serialisation
# ----------------------------------------------------------------------


class TestMCPTool:
    def test_to_ollama_tool_shape(self):
        t = mcp.MCPTool(
            server_id="fs",
            name="read",
            description="read a file",
            input_schema={"type": "object"},
        )
        tool = t.to_ollama_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "fs__read"
        assert tool["function"]["parameters"] == {"type": "object"}

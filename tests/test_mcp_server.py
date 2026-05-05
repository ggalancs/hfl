# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.mcp.server`` — exposing HFL tools as MCP server.

Like the client tests, we don't require the real MCP SDK: the
handlers are plain async functions so they are testable without it.
Only the SDK-bound ``build_server`` path is gated.
"""

from __future__ import annotations

import pytest

from hfl.mcp import server as mcp_server
from hfl.mcp.server import HFL_TOOLS, HFLMCPServer, MCPServerUnavailableError


class TestToolRegistry:
    def test_registry_has_expected_tools(self):
        assert {"web_search", "web_fetch", "model_list", "model_show"} <= set(HFL_TOOLS)

    def test_every_tool_has_schema(self):
        for name, spec in HFL_TOOLS.items():
            assert spec.name == name
            assert isinstance(spec.description, str) and spec.description
            assert spec.input_schema.get("type") == "object"


class TestCapabilityFiltering:
    def test_default_exposes_all_tools(self):
        s = HFLMCPServer()
        assert set(s.tool_names) == set(HFL_TOOLS)

    def test_explicit_capabilities_narrow_surface(self):
        s = HFLMCPServer(capabilities=["web_search"])
        assert s.tool_names == ["web_search"]

    def test_unknown_capabilities_are_dropped(self):
        s = HFLMCPServer(capabilities=["web_search", "nope"])
        assert s.tool_names == ["web_search"]

    def test_empty_capabilities_list_means_all(self):
        # Operator passed ``--capabilities`` with no value — we default
        # back to the full set rather than exposing nothing.
        s = HFLMCPServer(capabilities=[])
        assert set(s.tool_names) == set(HFL_TOOLS)


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------


class TestWebSearchHandler:
    async def test_happy_path(self, monkeypatch):
        from hfl.tools import web_search as ws

        class _B(ws.WebSearchBackend):
            name = "fake"

            async def search(self, q, mr):
                return [{"title": "T", "url": "U", "content": "C"}]

        monkeypatch.setattr(ws, "get_backend", lambda: _B())
        out = await HFL_TOOLS["web_search"].handler({"query": "x"})
        assert out[0]["type"] == "text"
        assert "T" in out[0]["text"]

    async def test_empty_query_rejected(self):
        with pytest.raises(ValueError):
            await HFL_TOOLS["web_search"].handler({"query": ""})

    async def test_backend_error_bubbles_as_valueerror(self, monkeypatch):
        from hfl.tools import web_search as ws

        class _B(ws.WebSearchBackend):
            name = "fake"

            async def search(self, q, mr):
                raise ws.WebSearchError("backend down")

        monkeypatch.setattr(ws, "get_backend", lambda: _B())
        with pytest.raises(ValueError):
            await HFL_TOOLS["web_search"].handler({"query": "x"})


class TestWebFetchHandler:
    async def test_happy_path(self, monkeypatch):
        from hfl.tools import web_fetch as wf

        async def _fake(url, **_k):
            return {"title": "T", "content": "C", "links": [], "url": url}

        monkeypatch.setattr(wf, "fetch", _fake)
        monkeypatch.setattr(mcp_server, "fetch", _fake)
        out = await HFL_TOOLS["web_fetch"].handler({"url": "https://x.example"})
        assert "T" in out[0]["text"]

    async def test_webfetch_error_becomes_valueerror(self, monkeypatch):
        from hfl.tools import web_fetch as wf

        async def _reject(url, **_k):
            raise wf.WebFetchError("bad scheme")

        monkeypatch.setattr(mcp_server, "fetch", _reject)
        with pytest.raises(ValueError):
            await HFL_TOOLS["web_fetch"].handler({"url": "ftp://x"})


class TestModelHandlers:
    async def test_model_list_returns_empty_when_registry_empty(self, temp_config, monkeypatch):
        from hfl.models.registry import reset_registry

        reset_registry()
        out = await HFL_TOOLS["model_list"].handler({})
        assert "[]" in out[0]["text"]

    async def test_model_show_rejects_unknown(self, temp_config):
        from hfl.models.registry import reset_registry

        reset_registry()
        with pytest.raises(ValueError):
            await HFL_TOOLS["model_show"].handler({"name": "ghost"})


# ----------------------------------------------------------------------
# SDK-absent gating
# ----------------------------------------------------------------------


class TestSDKGating:
    def test_build_server_raises_when_sdk_missing(self, monkeypatch):
        srv = HFLMCPServer()

        def _missing(self):
            raise MCPServerUnavailableError("no SDK")

        monkeypatch.setattr(HFLMCPServer, "_require_sdk", _missing)
        with pytest.raises(MCPServerUnavailableError):
            srv.build_server()

    def test_require_sdk_raises_on_missing_mcp(self, monkeypatch):
        """When ``mcp.server`` cannot be imported, ``_require_sdk``
        must wrap the ImportError as ``MCPServerUnavailableError``."""
        import sys

        # Inject ``mcp`` and ``mcp.server`` as None so the import
        # statement raises ImportError.
        for name in ("mcp", "mcp.server", "mcp.server.models", "mcp.types"):
            monkeypatch.setitem(sys.modules, name, None)

        srv = HFLMCPServer()
        with pytest.raises(MCPServerUnavailableError, match="MCP SDK"):
            srv._require_sdk()


def _install_fake_mcp(monkeypatch):
    """Inject a tiny fake ``mcp`` SDK so ``build_server`` runs."""
    import sys
    import types
    from unittest.mock import MagicMock

    handlers: dict = {}

    class _Server:
        def __init__(self, name):
            self.name = name

        def list_tools(self):
            def _decorator(fn):
                handlers["list_tools"] = fn
                return fn

            return _decorator

        def call_tool(self):
            def _decorator(fn):
                handlers["call_tool"] = fn
                return fn

            return _decorator

        async def run(self, read, write, opts):
            handlers["run_called"] = (read, write, opts)

        def create_initialization_options(self):
            return {"opts": True}

    class _Tool:
        def __init__(self, **kw):
            self._kw = kw

    class _TextContent:
        def __init__(self, type, text):
            self.type = type
            self.text = text

    mcp_pkg = types.ModuleType("mcp")
    mcp_server = types.ModuleType("mcp.server")
    mcp_server.Server = _Server
    mcp_server_models = types.ModuleType("mcp.server.models")
    mcp_server_models.InitializationOptions = MagicMock
    mcp_types = types.ModuleType("mcp.types")
    mcp_types.TextContent = _TextContent
    mcp_types.Tool = _Tool

    monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
    monkeypatch.setitem(sys.modules, "mcp.server", mcp_server)
    monkeypatch.setitem(sys.modules, "mcp.server.models", mcp_server_models)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types)
    return handlers, _TextContent


class TestBuildServerWithFakeSdk:
    """Walk ``HFLMCPServer.build_server`` end-to-end using a fake
    ``mcp`` SDK so the list_tools / call_tool decorators fire."""

    def test_build_server_returns_a_server(self, monkeypatch):
        _install_fake_mcp(monkeypatch)
        srv = HFLMCPServer().build_server()
        assert srv is not None
        assert getattr(srv, "name", None) == "hfl"

    @pytest.mark.asyncio
    async def test_list_tools_handler_returns_tool_objects(self, monkeypatch):
        handlers, _ = _install_fake_mcp(monkeypatch)
        HFLMCPServer().build_server()

        result = await handlers["list_tools"]()
        # One Tool per registered HFL spec.
        assert len(result) == len(HFL_TOOLS)

    @pytest.mark.asyncio
    async def test_call_tool_unknown_raises_value_error(self, monkeypatch):
        handlers, _ = _install_fake_mcp(monkeypatch)
        HFLMCPServer().build_server()

        with pytest.raises(ValueError, match="unknown tool"):
            await handlers["call_tool"]("not-a-real-tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_value_error_becomes_text_error(self, monkeypatch):
        handlers, TextContent = _install_fake_mcp(monkeypatch)
        srv = HFLMCPServer()
        # Replace one tool's handler with a stub that raises ValueError.
        from hfl.mcp.server import _ToolSpec as MCPToolSpec

        async def _bad(args):
            raise ValueError("bad input")

        srv._tools["echo"] = MCPToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object"},
            handler=_bad,
        )
        srv.build_server()

        out = await handlers["call_tool"]("echo", {})
        assert len(out) == 1
        assert "ERROR: bad input" in out[0].text

    @pytest.mark.asyncio
    async def test_call_tool_unexpected_exception_becomes_internal_error(self, monkeypatch):
        handlers, TextContent = _install_fake_mcp(monkeypatch)
        srv = HFLMCPServer()
        from hfl.mcp.server import _ToolSpec as MCPToolSpec

        async def _crash(args):
            raise RuntimeError("kernel panic")

        srv._tools["crash"] = MCPToolSpec(
            name="crash",
            description="crash",
            input_schema={"type": "object"},
            handler=_crash,
        )
        srv.build_server()

        out = await handlers["call_tool"]("crash", {})
        assert "internal server error" in out[0].text

    @pytest.mark.asyncio
    async def test_call_tool_marshals_text_payloads(self, monkeypatch):
        handlers, TextContent = _install_fake_mcp(monkeypatch)
        srv = HFLMCPServer()
        from hfl.mcp.server import _ToolSpec as MCPToolSpec

        async def _ok(args):
            return [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
                {"type": "image", "url": "ignored"},  # non-text dropped
            ]

        srv._tools["greet"] = MCPToolSpec(
            name="greet",
            description="g",
            input_schema={"type": "object"},
            handler=_ok,
        )
        srv.build_server()

        out = await handlers["call_tool"]("greet", None)
        assert [t.text for t in out] == ["hello", "world"]


class TestModelShowHandler:
    @pytest.mark.asyncio
    async def test_returns_text_payload_for_known_model(
        self, monkeypatch, sample_manifest, temp_config
    ):
        from hfl.mcp.server import _handle_model_show
        from hfl.models.registry import ModelRegistry

        sample_manifest.name = "qwen-1"
        ModelRegistry().add(sample_manifest)

        out = await _handle_model_show({"name": "qwen-1"})
        assert out and out[0]["type"] == "text"
        assert "qwen-1" in out[0]["text"]

    @pytest.mark.asyncio
    async def test_unknown_model_raises_value_error(self, monkeypatch, temp_config):
        from hfl.mcp.server import _handle_model_show

        with pytest.raises(ValueError, match="unknown model"):
            await _handle_model_show({"name": "never-was"})


class TestServeEntrypoints:
    @pytest.mark.asyncio
    async def test_serve_stdio_runs_server(self, monkeypatch):
        """``serve_stdio`` builds the server and drives it via the
        SDK's ``stdio_server`` context manager."""
        _install_fake_mcp(monkeypatch)

        # Add a fake stdio_server to the SDK.
        import sys
        import types

        async def _fake_run(read, write, opts):
            return None

        class _CM:
            async def __aenter__(self):
                return ("read", "write")

            async def __aexit__(self, *args):
                return False

        stdio_mod = types.ModuleType("mcp.server.stdio")
        stdio_mod.stdio_server = lambda: _CM()
        monkeypatch.setitem(sys.modules, "mcp.server.stdio", stdio_mod)

        # Patch Server.run to be a coroutine no-op so the await
        # doesn't hang.
        from hfl.mcp.server import HFLMCPServer

        builder = HFLMCPServer.build_server

        def _build_and_patch(self):
            srv = builder(self)
            srv.run = _fake_run
            return srv

        monkeypatch.setattr(HFLMCPServer, "build_server", _build_and_patch)

        from hfl.mcp.server import serve_stdio

        await serve_stdio(["web_search"])  # must not raise

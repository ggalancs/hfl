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

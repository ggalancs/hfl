# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""MCP client/server security and lifecycle (stdio env, timeouts, rebinding).

The real-SDK tests start a tiny stdio MCP server (a Python script in
tmp_path) — no network. They skip where ``mcp`` is not installed (CI venv).
"""

from __future__ import annotations

import asyncio
import gc
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hfl.mcp import client as mcp

try:
    import mcp as _sdk  # noqa: F401

    HAVE_SDK = True
except ImportError:
    HAVE_SDK = False

needs_sdk = pytest.mark.skipif(not HAVE_SDK, reason="mcp SDK not installed")

STUB = textwrap.dedent(
    """
    import os
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("stub")

    @app.tool()
    def env(name: str) -> str:
        return os.environ.get(name, "<unset>")

    app.run("stdio")
    """
)


@pytest.fixture(autouse=True)
def _fresh_client():
    mcp.reset_client()
    yield
    mcp.reset_client()


@pytest.fixture
def stub_target(tmp_path):
    script = tmp_path / "stub_mcp.py"
    script.write_text(STUB)
    return f"stdio://{sys.executable} {script}"


def _text(result) -> str:
    return result.content[0].text


# ----------------------------------------------------------------------
# Finding 1: stdio servers no longer inherit HFL's environment
# ----------------------------------------------------------------------


def _fake_sdk(monkeypatch, *, call_tool=None):
    seen: dict = {}

    class _Session:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(
                tools=[SimpleNamespace(name="t", description="", inputSchema={})]
            )

        async def call_tool(self, name, arguments):
            if call_tool is not None:
                return await call_tool(name, arguments)
            return "ok"

    class _CM:
        async def __aenter__(self):
            return ("r", "w")

        async def __aexit__(self, *_):
            return False

    def _params(**kwargs):
        seen["params"] = kwargs
        return kwargs

    monkeypatch.setattr(
        mcp.MCPClient,
        "_require_sdk",
        staticmethod(
            lambda: {
                "ClientSession": _Session,
                "StdioServerParameters": _params,
                "stdio_client": lambda p: _CM(),
                "sse_client": lambda u: _CM(),
            }
        ),
    )
    return seen


async def test_stdio_params_leave_env_to_the_sdk_default(monkeypatch):
    seen = _fake_sdk(monkeypatch)
    client = mcp.MCPClient()
    await client.connect("a", "stdio://x")
    # None → the SDK's get_default_environment() (PATH, HOME…), not ours.
    assert seen["params"]["env"] is None
    await client.connect("b", "stdio://x", env={"ONLY": "this"})
    assert seen["params"]["env"] == {"ONLY": "this"}
    await client.disconnect_all()


@needs_sdk
async def test_a_stdio_server_does_not_see_hfl_secrets(monkeypatch, stub_target):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("HFL_API_KEY", "hfl_secret_value")
    client = mcp.MCPClient()
    await client.connect("s", stub_target)
    await client.connect("d", stub_target, env={"DECLARED": "yes"})
    try:
        for sid in ("s", "d"):
            assert _text(await client.call_tool(f"{sid}__env", {"name": "HF_TOKEN"})) == "<unset>"
            assert (
                _text(await client.call_tool(f"{sid}__env", {"name": "HFL_API_KEY"})) == "<unset>"
            )
            # The SDK's safe default still reaches it (PATH to find programs).
            assert _text(await client.call_tool(f"{sid}__env", {"name": "PATH"})) != "<unset>"
        # What the config declares does reach it, on top of that default.
        assert _text(await client.call_tool("d__env", {"name": "DECLARED"})) == "yes"
    finally:
        await client.disconnect_all()


# ----------------------------------------------------------------------
# Finding 13: call_tool right after connect
# ----------------------------------------------------------------------


@needs_sdk
async def test_call_tool_works_after_connect_returns(monkeypatch, stub_target):
    """connect() entered the transport by hand and dropped it: GC finalised
    it from another task, killed the subprocess and cancelled the caller."""
    monkeypatch.setenv("PROBE", "x")
    client = mcp.MCPClient()
    await client.connect("s", stub_target)
    try:
        gc.collect()
        await asyncio.sleep(0.3)  # let any finaliser run
        assert _text(await client.call_tool("s__env", {"name": "PROBE"})) == "<unset>"
    finally:
        await client.disconnect_all()
    assert client.list_tools() == []


@needs_sdk
async def test_call_tool_works_from_another_task(stub_target):
    """The server connects in its lifespan and calls from request tasks."""
    client = mcp.MCPClient()
    await asyncio.create_task(client.connect("s", stub_target))
    try:
        result = await asyncio.create_task(client.call_tool("s__env", {"name": "NOPE"}))
        assert _text(result) == "<unset>"
    finally:
        await asyncio.create_task(client.disconnect_all())


# ----------------------------------------------------------------------
# Finding 12: bounded tool calls
# ----------------------------------------------------------------------


async def test_a_tool_call_that_never_answers_is_bounded(monkeypatch):
    async def _hang(name, arguments):
        await asyncio.sleep(3600)

    _fake_sdk(monkeypatch, call_tool=_hang)
    monkeypatch.setattr(mcp, "_CALL_TOOL_TIMEOUT_S", 0.2)
    client = mcp.MCPClient()
    await client.connect("s", "stdio://x")
    # The outer bound is the harness's own: without the client's, this test
    # fails (TimeoutError) instead of hanging.
    with pytest.raises(mcp.MCPConnectionError):
        await asyncio.wait_for(client.call_tool("s__t", {}), timeout=5)
    await client.disconnect_all()


# ----------------------------------------------------------------------
# Finding 3: DNS-rebinding protection on the SSE server
# ----------------------------------------------------------------------


@needs_sdk
@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
def test_sse_server_rejects_a_foreign_host_header(bind):
    from starlette.testclient import TestClient

    from hfl.mcp.server import _build_sse_app

    app = _build_sse_app(MagicMock(), bind)
    client = TestClient(app, raise_server_exceptions=False)
    body = b"{}"
    headers = {"content-type": "application/json"}
    # A rebinding page: the browser sends its own domain as Host.
    evil = client.post(
        "/messages/?session_id=0", content=body, headers={**headers, "host": "evil.example"}
    )
    assert evil.status_code == 421
    evil_origin = client.post(
        "/messages/?session_id=0",
        content=body,
        headers={**headers, "host": "127.0.0.1:8765", "origin": "http://evil.example"},
    )
    assert evil_origin.status_code == 403
    ok = client.post(
        "/messages/?session_id=0", content=body, headers={**headers, "host": "127.0.0.1:8765"}
    )
    assert ok.status_code not in (421, 403)  # past the guard (bad session id)


@needs_sdk
def test_sse_server_allows_the_named_bind_address():
    from starlette.testclient import TestClient

    from hfl.mcp.server import _build_sse_app

    client = TestClient(_build_sse_app(MagicMock(), "192.168.1.50"), raise_server_exceptions=False)
    r = client.post(
        "/messages/?session_id=0",
        content=b"{}",
        headers={"content-type": "application/json", "host": "192.168.1.50:8765"},
    )
    assert r.status_code not in (421, 403)

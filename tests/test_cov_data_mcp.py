# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Coverage for ``hfl.mcp`` paths that need the SDK, run against fakes.

The ``mcp`` SDK is an optional extra and is absent from the CI venv, so
every SDK surface these modules import is faked in ``sys.modules``. The
fakes are fresh modules installed with ``monkeypatch.setitem``, so the
real SDK (present in the dev venv) is never mutated and is restored after
each test.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sys
import types
from types import SimpleNamespace

import pytest

from hfl.mcp import client as mcp_client
from hfl.mcp import server as mcp_server


@pytest.fixture(autouse=True)
def _fresh_client():
    mcp_client.reset_client()
    yield
    mcp_client.reset_client()


def _module(monkeypatch, name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


# ----------------------------------------------------------------------
# Client: the real _require_sdk against a faked SDK
# ----------------------------------------------------------------------


class _FakeSession:
    """ClientSession stand-in; behaviour is driven by class attributes."""

    hang_initialize = False
    raise_on_exit: BaseException | None = None
    call_raises: BaseException | None = None

    def __init__(self, read, write):
        self.streams = (read, write)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self.raise_on_exit is not None:
            raise self.raise_on_exit
        return False

    async def initialize(self):
        if self.hang_initialize:
            await asyncio.Event().wait()

    async def list_tools(self):
        return SimpleNamespace(
            tools=[SimpleNamespace(name="echo", description=None, inputSchema=None)]
        )

    async def call_tool(self, name, arguments):
        if self.call_raises is not None:
            raise self.call_raises
        return {"name": name, "arguments": arguments}


class _Transport:
    async def __aenter__(self):
        return ("r", "w", "extra")

    async def __aexit__(self, *_):
        return False


def _install_client_sdk(monkeypatch, session_cls=_FakeSession):
    seen: dict = {}

    class _Params:
        def __init__(self, command, args, env):
            seen["params"] = (command, args, env)

    def _stdio_client(params):
        seen["stdio"] = params
        return _Transport()

    def _sse_client(url):
        seen["sse_url"] = url
        return _Transport()

    _module(monkeypatch, "mcp", ClientSession=session_cls, StdioServerParameters=_Params)
    _module(monkeypatch, "mcp.client")
    _module(monkeypatch, "mcp.client.sse", sse_client=_sse_client)
    _module(monkeypatch, "mcp.client.stdio", stdio_client=_stdio_client)
    return seen


class TestClientRequireSdk:
    def test_require_sdk_returns_the_sdk_surface(self, monkeypatch):
        _install_client_sdk(monkeypatch)
        sdk = mcp_client.MCPClient._require_sdk()
        assert set(sdk) == {"ClientSession", "StdioServerParameters", "sse_client", "stdio_client"}
        assert sdk["ClientSession"] is _FakeSession

    def test_require_sdk_translates_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mcp", None)
        with pytest.raises(mcp_client.MCPClientUnavailableError, match="hfl\\[mcp\\]"):
            mcp_client.MCPClient._require_sdk()

    async def test_stdio_connect_passes_command_args_and_no_inherited_env(self, monkeypatch):
        seen = _install_client_sdk(monkeypatch)
        client = mcp_client.MCPClient()
        tools = await client.connect("fs", "stdio://npx srv /tmp")
        # env=None lets the SDK pick its minimal default env (no HF_TOKEN leak).
        assert seen["params"] == ("npx", ["srv", "/tmp"], None)
        # Missing description/schema fall back to "" and a bare object schema.
        assert tools[0].description == ""
        assert tools[0].input_schema == {"type": "object"}
        await client.disconnect_all()
        assert client.list_tools() == []

    async def test_declared_env_is_forwarded(self, monkeypatch):
        seen = _install_client_sdk(monkeypatch)
        client = mcp_client.MCPClient()
        await client.connect("fs", "stdio://srv", env={"ONLY": "this"})
        assert seen["params"][2] == {"ONLY": "this"}
        await client.disconnect_all()

    async def test_sse_scheme_becomes_https_and_http_is_kept(self, monkeypatch):
        seen = _install_client_sdk(monkeypatch)
        client = mcp_client.MCPClient()
        await client.connect("a", "sse://host:8000/sse")
        assert seen["sse_url"] == "https://host:8000/sse"
        await client.connect("b", "http://127.0.0.1:9/sse")
        assert seen["sse_url"] == "http://127.0.0.1:9/sse"
        assert {t.qualified_name for t in client.list_tools()} == {"a__echo", "b__echo"}
        await client.disconnect_all()


class TestClientLifecycle:
    async def test_cancelled_connect_stops_the_owner_task(self, monkeypatch):
        class _Hanging(_FakeSession):
            hang_initialize = True

        _install_client_sdk(monkeypatch, _Hanging)
        client = mcp_client.MCPClient()
        connect = asyncio.create_task(client.connect("slow", "stdio://srv"))
        # Let the owner task start and block in initialize().
        for _ in range(5):
            await asyncio.sleep(0)
        owners = [t for t in asyncio.all_tasks() if t.get_name() == "mcp-slow"]
        assert len(owners) == 1
        connect.cancel()
        with pytest.raises(asyncio.CancelledError):
            await connect
        with pytest.raises(asyncio.CancelledError):
            await owners[0]
        assert owners[0].cancelled()
        # Nothing was registered for a connection that never came up.
        assert client.list_tools() == []

    async def test_owner_cancelled_before_session_is_up_cancels_connect(self, monkeypatch):
        class _Hanging(_FakeSession):
            hang_initialize = True

        _install_client_sdk(monkeypatch, _Hanging)
        client = mcp_client.MCPClient()
        connect = asyncio.create_task(client.connect("slow", "stdio://srv"))
        for _ in range(5):
            await asyncio.sleep(0)
        (owner,) = [t for t in asyncio.all_tasks() if t.get_name() == "mcp-slow"]
        owner.cancel()  # e.g. loop shutdown: the waiting connect must not hang
        with pytest.raises(asyncio.CancelledError):
            await connect
        assert owner.cancelled()
        assert client.list_tools() == []

    async def test_error_after_session_is_up_is_only_logged(self, monkeypatch, caplog):
        class _BadExit(_FakeSession):
            raise_on_exit = RuntimeError("pipe closed")

        _install_client_sdk(monkeypatch, _BadExit)
        client = mcp_client.MCPClient()
        await client.connect("fs", "stdio://srv")
        with caplog.at_level(logging.WARNING, logger="hfl.mcp.client"):
            await client.disconnect("fs")
        assert "MCP connection ended: RuntimeError" in caplog.text
        assert client.tool_by_qualified_name("fs__echo") is None

    async def test_disconnect_unknown_id_is_a_noop(self):
        client = mcp_client.MCPClient()
        await client.disconnect("never-connected")
        await client.disconnect("never-connected")
        assert client.list_tools() == []

    async def test_stop_owner_without_task_only_sets_stop(self):
        conn = mcp_client._ServerConnection(server_id="x", transport="stdio", target="t")
        await mcp_client.MCPClient._stop_owner(conn)  # stop and task both None
        conn.stop = asyncio.Event()
        await mcp_client.MCPClient._stop_owner(conn)
        assert conn.stop.is_set()

    async def test_stop_owner_logs_a_failing_owner_task(self, caplog):
        async def _boom():
            raise RuntimeError("transport wedged")

        conn = mcp_client._ServerConnection(server_id="x", transport="stdio", target="t")
        conn.stop = asyncio.Event()
        conn.task = asyncio.create_task(_boom())
        with caplog.at_level(logging.ERROR, logger="hfl.mcp.client"):
            await mcp_client.MCPClient._stop_owner(conn)
        assert "MCP disconnect error for x" in caplog.text


class TestClientCallTool:
    async def test_call_tool_without_session_raises(self):
        client = mcp_client.MCPClient()
        tool = mcp_client.MCPTool("fs", "read", "", {"type": "object"})
        client._servers["fs"] = mcp_client._ServerConnection(
            server_id="fs", transport="stdio", target="t", session=None, tools=[tool]
        )
        with pytest.raises(mcp_client.MCPConnectionError, match="no active session") as exc:
            await client.call_tool("fs__read")
        assert exc.value.server_id == "fs"

    async def test_call_tool_failure_is_curated(self, monkeypatch):
        class _Failing(_FakeSession):
            call_raises = OSError("secret /path detail")

        _install_client_sdk(monkeypatch, _Failing)
        client = mcp_client.MCPClient()
        await client.connect("fs", "stdio://srv")
        with pytest.raises(mcp_client.MCPConnectionError) as exc:
            await client.call_tool("fs__echo", None)
        assert "invocation failed" in str(exc.value)
        assert "secret" not in str(exc.value)
        await client.disconnect_all()

    async def test_call_tool_defaults_arguments_to_empty_dict(self, monkeypatch):
        _install_client_sdk(monkeypatch)
        client = mcp_client.MCPClient()
        await client.connect("fs", "stdio://srv")
        assert await client.call_tool("fs__echo") == {"name": "echo", "arguments": {}}
        await client.disconnect_all()


class TestAutoloadEdges:
    async def test_no_path_and_no_env_means_nothing(self, monkeypatch):
        monkeypatch.delenv("HFL_MCP_AUTOLOAD", raising=False)
        assert await mcp_client.autoload_servers() == []

    async def test_a_broken_entry_does_not_block_the_others(self, monkeypatch, tmp_path, caplog):
        _install_client_sdk(monkeypatch)
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            '{"servers": [{"id": "bad", "target": "ftp://nope"},'
            ' {"id": "good", "target": "stdio://srv"}]}'
        )
        with caplog.at_level(logging.WARNING, logger="hfl.mcp.client"):
            loaded = await mcp_client.autoload_servers(cfg)
        assert loaded == ["good"]
        assert "MCP autoload skipped bad" in caplog.text
        await mcp_client.get_client().disconnect_all()

    async def test_get_client_is_a_singleton(self):
        assert mcp_client.get_client() is mcp_client.get_client()


# ----------------------------------------------------------------------
# Server: SDK 2.x wiring against a faked ``mcp.types``
# ----------------------------------------------------------------------


class _Tool:
    def __init__(self, name, description, inputSchema):
        self.name, self.description, self.inputSchema = name, description, inputSchema


class _Text:
    def __init__(self, type, text):
        self.type, self.text = type, text


class _Server2x:
    def __init__(self, name, on_list_tools=None, on_call_tool=None):
        self.name = name
        self.on_list_tools, self.on_call_tool = on_list_tools, on_call_tool


def _install_server_sdk_2x(monkeypatch):
    class _ListToolsResult:
        def __init__(self, tools):
            self.tools = tools

    class _CallToolResult:
        def __init__(self, content, isError):
            self.content, self.isError = content, isError

    mcp_types = _module(
        monkeypatch,
        "mcp.types",
        Tool=_Tool,
        TextContent=_Text,
        ListToolsResult=_ListToolsResult,
        CallToolResult=_CallToolResult,
    )
    _module(monkeypatch, "mcp", types=mcp_types)
    _module(monkeypatch, "mcp.server", Server=_Server2x)
    _module(monkeypatch, "mcp.server.models", InitializationOptions=object)


class TestServer2x:
    async def test_constructor_handlers_list_and_call(self, monkeypatch):
        _install_server_sdk_2x(monkeypatch)

        async def echo(arguments):
            if arguments.get("boom"):
                raise ValueError("bad input")
            return [{"type": "text", "text": f"hi {arguments.get('who')}"}]

        srv = mcp_server.HFLMCPServer()
        srv._tools = {"echo": mcp_server._ToolSpec("echo", "says hi", {"type": "object"}, echo)}
        server = srv.build_server()
        assert isinstance(server, _Server2x) and server.name == "hfl"

        listing = await server.on_list_tools(None, None)
        assert [(t.name, t.inputSchema) for t in listing.tools] == [("echo", {"type": "object"})]

        ok = await server.on_call_tool(None, SimpleNamespace(name="echo", arguments={"who": "b"}))
        assert ok.isError is False and [c.text for c in ok.content] == ["hi b"]

        bad = await server.on_call_tool(None, SimpleNamespace(name="echo", arguments={"boom": 1}))
        assert bad.isError is True and bad.content[0].text == "ERROR: bad input"

    async def test_capabilities_narrow_the_listing(self, monkeypatch):
        _install_server_sdk_2x(monkeypatch)
        server = mcp_server.HFLMCPServer([" web_fetch ", ""]).build_server()
        listing = await server.on_list_tools(None, None)
        assert [t.name for t in listing.tools] == ["web_fetch"]


# ----------------------------------------------------------------------
# Server: SSE transport security + entry point
# ----------------------------------------------------------------------


class TestBindHostNames:
    def test_specific_host_adds_only_itself(self):
        names = mcp_server._bind_host_names("192.168.1.10")
        assert names == ["127.0.0.1", "localhost", "[::1]", "192.168.1.10"]

    def test_ipv6_literal_is_bracketed_and_scope_dropped(self):
        names = mcp_server._bind_host_names("FE80::1%en0")
        assert names[-1] == "[fe80::1]"

    def test_loopback_host_is_not_duplicated(self):
        assert mcp_server._bind_host_names("LOCALHOST") == ["127.0.0.1", "localhost", "[::1]"]

    def test_wildcard_enumerates_interfaces_and_hostname(self, monkeypatch):
        fake_psutil = types.ModuleType("psutil")
        fake_psutil.net_if_addrs = lambda: {
            "en0": [
                SimpleNamespace(family=socket.AF_INET, address="10.0.0.5"),
                SimpleNamespace(family=socket.AF_INET6, address="fe80::abcd%en0"),
                SimpleNamespace(family=-1, address="aa:bb:cc:dd:ee:ff"),  # MAC: skipped
            ],
            "lo0": [SimpleNamespace(family=socket.AF_INET, address="127.0.0.1")],
        }
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(socket, "gethostname", lambda: "MyMac")
        names = mcp_server._bind_host_names("0.0.0.0")
        assert names == [
            "127.0.0.1",
            "localhost",
            "[::1]",
            "10.0.0.5",
            "[fe80::abcd]",
            "mymac",
            "mymac.local",
        ]
        assert "aa:bb:cc:dd:ee:ff" not in names

    def test_wildcard_survives_interface_enumeration_failure(self, monkeypatch):
        fake_psutil = types.ModuleType("psutil")

        def _boom():
            raise OSError("no permission")

        fake_psutil.net_if_addrs = _boom
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(socket, "gethostname", lambda: "box.local")
        # Hostname already .local: no second ".local" suffix.
        assert mcp_server._bind_host_names("::") == ["127.0.0.1", "localhost", "[::1]", "box.local"]


class _Security:
    def __init__(self, **kw):
        self.kw = kw


class TestTransportSecurity:
    def test_rebinding_protection_is_enabled_with_bound_names(self, monkeypatch):
        _module(monkeypatch, "mcp", types=None)
        _module(monkeypatch, "mcp.server")
        _module(monkeypatch, "mcp.server.transport_security", TransportSecuritySettings=_Security)
        sec = mcp_server._transport_security("127.0.0.1")
        assert sec.kw["enable_dns_rebinding_protection"] is True
        assert "localhost" in sec.kw["allowed_hosts"]
        assert "localhost:*" in sec.kw["allowed_hosts"]
        assert "http://127.0.0.1:*" in sec.kw["allowed_origins"]
        assert "https://[::1]" in sec.kw["allowed_origins"]
        assert not any("evil" in h for h in sec.kw["allowed_hosts"])

    def test_old_sdk_without_the_setting_returns_none(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "mcp.server.transport_security", None)
        with caplog.at_level(logging.WARNING, logger="hfl.mcp.server"):
            assert mcp_server._transport_security("127.0.0.1") is None
        assert "no DNS-rebinding protection" in caplog.text


class _SseTransport:
    instances: list = []

    def __init__(self, path, security_settings=None):
        self.path, self.security_settings = path, security_settings
        self.connected: list = []
        _SseTransport.instances.append(self)

    def connect_sse(self, scope, receive, send):
        transport = self

        class _CM:
            async def __aenter__(self):
                transport.connected.append(scope)
                return ("read", "write")

            async def __aexit__(self, *_):
                return False

        return _CM()

    async def handle_post_message(self, scope, receive, send):
        return None


def _install_sse(monkeypatch, with_security: bool):
    _SseTransport.instances = []
    _module(monkeypatch, "mcp", types=None)
    _module(monkeypatch, "mcp.server")
    _module(monkeypatch, "mcp.server.sse", SseServerTransport=_SseTransport)
    if with_security:
        _module(monkeypatch, "mcp.server.transport_security", TransportSecuritySettings=_Security)
    else:
        monkeypatch.setitem(sys.modules, "mcp.server.transport_security", None)


class _RunServer:
    def __init__(self):
        self.runs: list = []

    async def run(self, read, write, opts):
        self.runs.append((read, write, opts))

    def create_initialization_options(self):
        return "init-opts"


class TestBuildSseApp:
    async def test_routes_and_session_run(self, monkeypatch):
        _install_sse(monkeypatch, with_security=True)
        server = _RunServer()
        app = mcp_server._build_sse_app(server, "127.0.0.1")
        transport = _SseTransport.instances[-1]
        assert transport.path == "/messages/"
        assert isinstance(transport.security_settings, _Security)

        paths = {getattr(r, "path", None): r for r in app.routes}
        assert set(paths) == {"/sse", "/messages"}
        request = SimpleNamespace(scope={"type": "http"}, receive=object(), _send=object())
        await paths["/sse"].endpoint(request)
        assert server.runs == [("read", "write", "init-opts")]
        assert transport.connected == [{"type": "http"}]

    def test_without_security_support_transport_is_plain(self, monkeypatch):
        _install_sse(monkeypatch, with_security=False)
        mcp_server._build_sse_app(_RunServer(), "127.0.0.1")
        assert _SseTransport.instances[-1].security_settings is None


class TestServeSse:
    async def test_serve_sse_runs_uvicorn_on_host_port(self, monkeypatch):
        _install_sse(monkeypatch, with_security=True)
        # _install_sse made bare "mcp"/"mcp.server"; give build_server its SDK.
        _install_server_sdk_2x(monkeypatch)
        import uvicorn

        served: list = []

        class _UServer:
            def __init__(self, config):
                self.config = config

            async def serve(self):
                served.append(self.config)

        monkeypatch.setattr(uvicorn, "Server", _UServer)
        await mcp_server.serve_sse("127.0.0.1", 8765, ["web_search"])
        assert len(served) == 1
        assert (served[0].host, served[0].port) == ("127.0.0.1", 8765)
        assert served[0].log_level == "info"

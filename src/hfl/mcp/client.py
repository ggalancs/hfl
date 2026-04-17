# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""MCP client — connect to external MCP servers and surface their tools.

Phase 9 P0. Model Context Protocol is how 2026 agent stacks plug
external tool providers (Cline's ``serena``, Goose's memory server,
Anthropic's ``filesystem``, …) into any model host. This module lets
HFL play the host role: on boot we connect to every configured MCP
server, enumerate its tools, and merge them into the ``tools`` array
of any ``/api/chat`` request that didn't already include them.

Transports:
- ``stdio://<command> [args…]`` — subprocess speaking JSON-RPC over
  stdin/stdout. The MCP reference pattern.
- ``sse://<url>`` — HTTP + Server-Sent Events. Used by cloud-hosted
  servers.

The official ``mcp`` Python SDK is an optional dependency; if it's
not installed the module still imports and every public call is a
graceful no-op so unrelated code doesn't crash.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MCPClient",
    "MCPTool",
    "MCPClientUnavailableError",
    "MCPConnectionError",
    "get_client",
    "reset_client",
    "autoload_servers",
]


class MCPClientUnavailableError(RuntimeError):
    """Raised when the ``mcp`` SDK is not installed.

    Surfaced to the CLI; routes never raise this — they log a warning
    and proceed without MCP tools so the agent loop still works.
    """


class MCPConnectionError(RuntimeError):
    """Raised when a specific MCP server cannot be reached.

    Carries ``server_id`` so the route can report which configured
    entry is broken without leaking subprocess / socket details.
    """

    def __init__(self, server_id: str, reason: str) -> None:
        super().__init__(f"MCP server {server_id!r}: {reason}")
        self.server_id = server_id


@dataclass
class MCPTool:
    """Tool exposed by a connected MCP server.

    Converted at the boundary into the Ollama / OpenAI ``tools``
    schema so the model sees a uniform shape regardless of origin.
    """

    server_id: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        """Scope the tool name to its server to prevent collisions.

        Two MCP servers can each define a ``search`` tool; we expose
        them as ``server_id__search`` so the model can address them
        unambiguously and the router knows which connection to use.
        """
        return f"{self.server_id}__{self.name}"

    def to_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class _ServerConnection:
    """Bookkeeping for one connected MCP server."""

    server_id: str
    transport: str
    target: str
    session: Any | None = None  # mcp.ClientSession, typed loosely for optional dep
    tools: list[MCPTool] = field(default_factory=list)


class MCPClient:
    """Singleton holding live connections to every configured server."""

    def __init__(self) -> None:
        self._servers: dict[str, _ServerConnection] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @staticmethod
    def _require_sdk() -> Any:
        """Import ``mcp`` on demand; translate ImportError to our own."""
        try:
            import mcp  # noqa: F401
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.sse import sse_client  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore
        except ImportError as exc:
            raise MCPClientUnavailableError(
                "The MCP SDK is not installed. `pip install 'hfl[mcp]'` adds it."
            ) from exc
        return {
            "ClientSession": ClientSession,
            "StdioServerParameters": StdioServerParameters,
            "sse_client": sse_client,
            "stdio_client": stdio_client,
        }

    async def connect(
        self,
        server_id: str,
        target: str,
        *,
        env: dict[str, str] | None = None,
    ) -> list[MCPTool]:
        """Open a session against ``target`` and enumerate its tools.

        ``target`` is one of:

        - ``stdio://<cmd> <arg1> <arg2>`` — launches a subprocess.
        - ``sse://<url>`` — connects to an HTTP SSE endpoint.

        Returns the list of tools discovered. Failures raise
        ``MCPConnectionError`` with a curated message; the underlying
        exception is logged via ``logger.exception``.
        """
        sdk = self._require_sdk()

        if target.startswith("stdio://"):
            transport = "stdio"
            parts = target[len("stdio://") :].split()
            if not parts:
                raise MCPConnectionError(server_id, "stdio target is empty")
            params = sdk["StdioServerParameters"](
                command=parts[0],
                args=parts[1:],
                env=env or os.environ.copy(),
            )
            transport_cm = sdk["stdio_client"](params)
        elif target.startswith(("sse://", "http://", "https://")):
            transport = "sse"
            url = target.replace("sse://", "https://", 1) if target.startswith("sse://") else target
            transport_cm = sdk["sse_client"](url)
        else:
            raise MCPConnectionError(
                server_id,
                f"unknown transport in {target!r} (expected stdio:// or sse://)",
            )

        conn = _ServerConnection(
            server_id=server_id,
            transport=transport,
            target=target,
        )
        async with self._lock:
            try:
                read, write = await transport_cm.__aenter__()
                session = await sdk["ClientSession"](read, write).__aenter__()
                await session.initialize()
                listing = await session.list_tools()
                tools = [
                    MCPTool(
                        server_id=server_id,
                        name=t.name,
                        description=t.description or "",
                        input_schema=getattr(t, "inputSchema", None) or {"type": "object"},
                    )
                    for t in getattr(listing, "tools", [])
                ]
                conn.session = session
                conn.tools = tools
                self._servers[server_id] = conn
            except MCPClientUnavailableError:
                raise
            except Exception as exc:
                logger.exception("MCP connection failed for %s", server_id)
                raise MCPConnectionError(server_id, "failed to initialize session") from exc
        return tools

    async def disconnect(self, server_id: str) -> None:
        """Close the session and drop bookkeeping for ``server_id``.

        Idempotent — calling twice is safe.
        """
        async with self._lock:
            conn = self._servers.pop(server_id, None)
            if conn is None or conn.session is None:
                return
            try:
                await conn.session.__aexit__(None, None, None)
            except Exception:
                logger.exception("MCP disconnect error for %s", server_id)

    async def disconnect_all(self) -> None:
        ids = list(self._servers.keys())
        for sid in ids:
            await self.disconnect(sid)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_tools(self) -> list[MCPTool]:
        """Return the union of tools across every connected server."""
        out: list[MCPTool] = []
        for conn in self._servers.values():
            out.extend(conn.tools)
        return out

    def tool_by_qualified_name(self, name: str) -> MCPTool | None:
        for tool in self.list_tools():
            if tool.qualified_name == name:
                return tool
        return None

    async def call_tool(
        self,
        qualified_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke a tool by its ``server_id__tool_name`` qualified name.

        Returns the server's raw result object; the caller is
        responsible for serialising it (typically via ``str()`` into
        the tool-result message the model consumes).
        """
        tool = self.tool_by_qualified_name(qualified_name)
        if tool is None:
            raise MCPConnectionError(qualified_name, "tool not found")
        conn = self._servers.get(tool.server_id)
        if conn is None or conn.session is None:
            raise MCPConnectionError(tool.server_id, "no active session")
        try:
            return await conn.session.call_tool(tool.name, arguments or {})
        except Exception as exc:
            logger.exception("MCP call_tool failed: %s", qualified_name)
            raise MCPConnectionError(
                tool.server_id,
                f"tool {tool.name!r} invocation failed",
            ) from exc


# ----------------------------------------------------------------------
# Global singleton + autoload
# ----------------------------------------------------------------------


_client: MCPClient | None = None


def get_client() -> MCPClient:
    global _client
    if _client is None:
        _client = MCPClient()
    return _client


def reset_client() -> None:
    """Test hook — drops the singleton so each test gets a clean one."""
    global _client
    _client = None


async def autoload_servers(config_path: str | os.PathLike | None = None) -> list[str]:
    """Connect every server declared in ``HFL_MCP_AUTOLOAD``.

    The config file is JSON of the shape::

        {
          "servers": [
            {"id": "fs", "target": "stdio://npx @modelcontextprotocol/server-filesystem /tmp"},
            {"id": "web", "target": "sse://localhost:8000/sse"}
          ]
        }

    Returns the list of server-ids that connected successfully.
    Failures are logged but don't abort the server — a broken MCP
    entry must never block ``hfl serve`` from coming up.
    """
    path = config_path or os.environ.get("HFL_MCP_AUTOLOAD")
    if not path:
        return []
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning("HFL_MCP_AUTOLOAD points to missing file: %s", cfg_path)
        return []

    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("HFL_MCP_AUTOLOAD invalid JSON: %s", exc)
        return []

    servers = cfg.get("servers") or []
    client = get_client()
    ok: list[str] = []
    for entry in servers:
        sid = entry.get("id")
        target = entry.get("target")
        if not sid or not target:
            continue
        try:
            await client.connect(sid, target, env=entry.get("env"))
            ok.append(sid)
        except MCPClientUnavailableError:
            logger.warning("MCP SDK not installed — skipping autoload")
            return []
        except MCPConnectionError as exc:
            logger.warning("MCP autoload skipped %s: %s", sid, exc)
    return ok

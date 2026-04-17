# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HFL as an MCP server — expose our built-in tools to external hosts.

Phase 10 P1. Counterpart to ``hfl.mcp.client``: any MCP-speaking
host (Cline, Goose, Claude Desktop, Codex) can now ``Add server →
stdio://hfl mcp serve`` (or ``sse://localhost:8765``) and get
``web_search``, ``web_fetch``, ``model.list``, ``model.show`` as
callable tools without running their own code.

Tool registry is minimal on purpose: we expose only *safe*
operations by default. ``--capabilities`` narrows the set further
— an operator hardening the surface can run
``hfl mcp serve --capabilities web_search`` to publish only the
search tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from hfl.tools.web_fetch import WebFetchError, fetch
from hfl.tools.web_search import WebSearchError, search

logger = logging.getLogger(__name__)

__all__ = [
    "HFL_TOOLS",
    "MCPServerUnavailableError",
    "HFLMCPServer",
    "serve_stdio",
    "serve_sse",
]


class MCPServerUnavailableError(RuntimeError):
    """Raised when the MCP SDK isn't installed.

    Mirrors ``MCPClientUnavailableError`` from the client module —
    allows CLI code to surface a uniform message regardless of which
    side tripped.
    """


@dataclass
class _ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]


# ----------------------------------------------------------------------
# Handlers — adapt HFL internals to MCP's calling convention.
# ----------------------------------------------------------------------


async def _handle_web_search(args: dict[str, Any]) -> list[dict[str, Any]]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query must be a non-empty string")
    max_results = int(args.get("max_results") or 5)
    try:
        payload = await search(query, max_results)
    except WebSearchError as exc:
        raise ValueError(str(exc)) from exc
    return [{"type": "text", "text": str(payload)}]


async def _handle_web_fetch(args: dict[str, Any]) -> list[dict[str, Any]]:
    url = str(args.get("url") or "")
    try:
        payload = await fetch(url)
    except WebFetchError as exc:
        raise ValueError(str(exc)) from exc
    return [{"type": "text", "text": str(payload)}]


async def _handle_model_list(_args: dict[str, Any]) -> list[dict[str, Any]]:
    from hfl.models.registry import get_registry

    models = get_registry().list_all()
    payload = [
        {
            "name": m.name,
            "repo_id": m.repo_id,
            "size_bytes": m.size_bytes,
            "architecture": m.architecture,
        }
        for m in models
    ]
    return [{"type": "text", "text": str(payload)}]


async def _handle_model_show(args: dict[str, Any]) -> list[dict[str, Any]]:
    from hfl.models.registry import get_registry

    name = str(args.get("name") or "")
    manifest = get_registry().get(name)
    if manifest is None:
        raise ValueError(f"unknown model {name!r}")
    return [
        {
            "type": "text",
            "text": str(
                {
                    "name": manifest.name,
                    "local_path": manifest.local_path,
                    "format": manifest.format,
                    "context_length": manifest.context_length,
                    "architecture": manifest.architecture,
                }
            ),
        }
    ]


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


HFL_TOOLS: dict[str, _ToolSpec] = {
    "web_search": _ToolSpec(
        name="web_search",
        description=(
            "Search the web for up-to-date information. Returns ranked "
            "{title,url,content} hits from the configured search backend."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text."},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=_handle_web_search,
    ),
    "web_fetch": _ToolSpec(
        name="web_fetch",
        description=(
            "Fetch a URL and return the extracted title, visible content, "
            "and outbound links. Rejects private / loopback / metadata hosts."
        ),
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
            "required": ["url"],
        },
        handler=_handle_web_fetch,
    ),
    "model_list": _ToolSpec(
        name="model_list",
        description="List every model registered in HFL's local registry.",
        input_schema={"type": "object", "properties": {}},
        handler=_handle_model_list,
    ),
    "model_show": _ToolSpec(
        name="model_show",
        description="Return metadata for a specific model by name/alias/repo_id.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        handler=_handle_model_show,
    ),
}


# ----------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------


class HFLMCPServer:
    """Thin adapter around the MCP SDK's ``Server`` primitive.

    Kept separate from the SDK import so the module is always
    importable. Instantiation (``build_server()``) is what actually
    pulls ``mcp`` in.
    """

    def __init__(self, capabilities: list[str] | None = None) -> None:
        self.capabilities = capabilities
        self._tools = self._filter_tools(capabilities)

    @staticmethod
    def _filter_tools(capabilities: list[str] | None) -> dict[str, _ToolSpec]:
        if not capabilities:
            return dict(HFL_TOOLS)
        wanted = {c.strip() for c in capabilities if c.strip()}
        return {name: spec for name, spec in HFL_TOOLS.items() if name in wanted}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def _require_sdk(self) -> Any:
        try:
            from mcp.server import Server  # type: ignore
            from mcp.server.models import InitializationOptions  # type: ignore
            from mcp.types import TextContent, Tool  # type: ignore
        except ImportError as exc:
            raise MCPServerUnavailableError(
                "The MCP SDK is not installed. `pip install 'hfl[mcp]'` adds it."
            ) from exc
        return {
            "Server": Server,
            "Tool": Tool,
            "TextContent": TextContent,
            "InitializationOptions": InitializationOptions,
        }

    def build_server(self) -> Any:
        """Construct the ``mcp.server.Server`` with tool handlers wired up."""
        sdk = self._require_sdk()
        Server = sdk["Server"]
        Tool = sdk["Tool"]
        TextContent = sdk["TextContent"]

        server = Server("hfl")

        @server.list_tools()  # type: ignore[misc]
        async def _list_tools() -> list[Any]:
            return [
                Tool(
                    name=spec.name,
                    description=spec.description,
                    inputSchema=spec.input_schema,
                )
                for spec in self._tools.values()
            ]

        @server.call_tool()  # type: ignore[misc]
        async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
            spec = self._tools.get(name)
            if spec is None:
                raise ValueError(f"unknown tool {name!r}")
            try:
                payload = await spec.handler(arguments or {})
            except ValueError as exc:
                # ValueError is the contract for "bad input"; MCP
                # surfaces it as an IsError response.
                return [TextContent(type="text", text=f"ERROR: {exc}")]
            except Exception:
                logger.exception("MCP tool handler crashed: %s", name)
                return [TextContent(type="text", text="ERROR: internal server error")]
            # Marshal each dict back through the TextContent constructor
            # so the MCP client sees canonical SDK types.
            out = []
            for part in payload:
                if isinstance(part, dict) and part.get("type") == "text":
                    out.append(TextContent(type="text", text=part.get("text", "")))
            return out

        return server


async def serve_stdio(capabilities: list[str] | None = None) -> None:
    """Run the HFL MCP server over stdio.

    This blocks until the client closes the stream — intended to be
    launched as a subprocess by the MCP host (Cline, etc.).
    """
    server = HFLMCPServer(capabilities).build_server()
    from mcp.server.stdio import stdio_server  # type: ignore

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


async def serve_sse(host: str, port: int, capabilities: list[str] | None = None) -> None:
    """Run the HFL MCP server over SSE on ``host:port``.

    SSE is what networked MCP hosts prefer; ``uvicorn`` hosts the
    Starlette-based transport from the SDK.
    """
    server = HFLMCPServer(capabilities).build_server()
    import uvicorn
    from mcp.server.sse import SseServerTransport  # type: ignore
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):  # type: ignore[no-untyped-def]
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            read, write = streams
            await server.run(read, write, server.create_initialization_options())

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()

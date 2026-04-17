# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Model Context Protocol (MCP) integration (Phase 9+).

Modules:
- ``client``: connect to MCP servers (stdio / SSE) and expose their
  tools to HFL's chat route.
- ``server``: expose HFL's internal tools as an MCP server (Phase 10).

MCP is an optional feature behind the ``[mcp]`` extra; the rest of
HFL never imports this package directly so installations without the
``mcp`` SDK keep working.
"""

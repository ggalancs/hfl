# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl websearch``: the web search HFL's tools use (``web_search``, the
MCP server, Claude Code's WebSearch through ``hfl launch``). Either a SearXNG
the user runs (self-hosted, keyless: HFL connects to it by its port on this
machine or its URL, and never installs or starts it), or a search service
with an API key. The choice is saved in HFL's home, where every server reads
it, the desktop app's included."""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

import httpx

from hfl.tools import web_search as ws

KEYED = {
    "brave": "BRAVE_API_KEY",
    "tavily": "TAVILY_API_KEY",
    "exa": "EXA_API_KEY",
    "serpapi": "SERPAPI_API_KEY",
}
PROBE = "hugging face"


class SetupError(Exception):
    """What stopped the setup, in words for the person who ran it."""


# ----------------------------------------------------------------------
# SearXNG (the user's: HFL connects, never installs or starts it)
# ----------------------------------------------------------------------


def probe_searxng(url: str, timeout: float = 10.0) -> int:
    """How many results the SearXNG at ``url`` gives for a test query."""
    resp = httpx.get(
        url.rstrip("/") + "/search", params={"q": PROBE, "format": "json"}, timeout=timeout
    )
    if resp.status_code == 403:
        raise SetupError(
            f"{url} refused the JSON format HFL reads: its settings.yml needs "
            "`search: formats: [html, json]`"
        )
    resp.raise_for_status()
    return len(resp.json().get("results") or [])


def searxng_url(port: int | None, url: str | None) -> str:
    """The SearXNG to use: ``url``, or the one on this machine at ``port``."""
    if url:
        return url.rstrip("/")
    if port is None:
        raise SetupError(
            "give the port (--port) or the URL (--url) of the SearXNG you run; "
            "HFL does not install it (how to run one: docs/env-vars.md, HFL_SEARXNG_URL)"
        )
    if not 1 <= port <= 65535:
        raise SetupError(f"not a port: {port}")
    return f"http://127.0.0.1:{port}"


def use_searxng(url: str) -> int:
    """The user's SearXNG: checked with one search, then saved."""
    url = url.rstrip("/")
    try:
        count = probe_searxng(url)
    except httpx.HTTPError as exc:
        raise SetupError(f"no SearXNG answered at {url}: {exc}") from exc
    ws.save_settings({"HFL_WEB_SEARCH_BACKEND": "searxng", "HFL_SEARXNG_URL": url})
    return count


# ----------------------------------------------------------------------
# Keyed services
# ----------------------------------------------------------------------


def use_key(provider: str, key: str, check: bool = True) -> int | None:
    """A search service with its API key: tried with one search (unless
    ``check`` is off), then saved. The number of results it gave."""
    provider = provider.lower()
    var = KEYED.get(provider)
    if var is None:
        raise SetupError(f"unknown provider {provider!r}: one of {', '.join(sorted(KEYED))}")
    key = key.strip()
    if not key:
        raise SetupError("the API key is empty")
    count: int | None = None
    if check:
        backend = ws.get_backend(provider)
        saved = os.environ.get(var)
        os.environ[var] = key  # the backend reads its key from the settings
        try:
            results = asyncio.run(backend.search(PROBE, 3))
        except ws.WebSearchError as exc:
            raise SetupError(f"{provider} refused a test search with that key: {exc}") from exc
        finally:
            if saved is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = saved
        count = len(results)
    ws.save_settings({"HFL_WEB_SEARCH_BACKEND": provider, var: key})
    return count


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------


def _masked(value: str) -> str:
    return "…" + value[-4:] if len(value) > 8 else "…"


def status() -> dict[str, Any]:
    """What is configured, where each setting comes from, and one test
    search through it."""
    saved = ws.saved_settings()
    rows: list[tuple[str, str, str]] = []
    for var in ("HFL_WEB_SEARCH_BACKEND", "HFL_SEARXNG_URL", *KEYED.values()):
        if os.environ.get(var):
            value, source = os.environ[var], "environment"
        elif saved.get(var):
            value, source = saved[var], str(ws.settings_path())
        else:
            continue
        rows.append((var, _masked(value) if var.endswith("_API_KEY") else value, source))
    chain = [b.name for b in ws.backend_chain()]
    try:
        found = asyncio.run(ws.search(PROBE, 3))
        test = f"OK: {len(found['results'])} results from {found['backend']}"
        ok = True
    except ws.WebSearchError as exc:
        test, ok = f"failed: {exc}", False
    return {"settings": rows, "chain": chain, "test": test, "ok": ok}


def turn_off() -> bool:
    """Forget the saved choice (the environment still applies); whether
    there was one."""
    path = ws.settings_path()
    existed = path.exists()
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    return existed

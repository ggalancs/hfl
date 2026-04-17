# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Pluggable web-search backends (Phase 9 P0).

Exposes a single ``search(query, max_results)`` async function that
dispatches to a backend selected by ``HFL_WEB_SEARCH_BACKEND``:

- ``duckduckgo`` (default) — scrapes DDG's no-JavaScript HTML
  endpoint. No API key required. Free tier, best-effort parsing.
- ``tavily`` — calls https://api.tavily.com/search, requires
  ``TAVILY_API_KEY``. Best results quality.
- ``brave`` — calls https://api.search.brave.com/res/v1/web/search,
  requires ``BRAVE_API_KEY``.
- ``serpapi`` — https://serpapi.com/search, requires
  ``SERPAPI_API_KEY``.

Every backend returns the same Ollama-compatible shape:

    {
      "results": [
        {"title": str, "url": str, "content": str},
        ...
      ]
    }

so ``/api/web_search`` can forward the payload verbatim.
"""

from __future__ import annotations

import html
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "WebSearchBackend",
    "WebSearchError",
    "DuckDuckGoBackend",
    "TavilyBackend",
    "BraveBackend",
    "SerpAPIBackend",
    "get_backend",
    "search",
]


class WebSearchError(RuntimeError):
    """Raised when a backend can't serve a request.

    Message is safe to surface to the client (no stack-trace info),
    matching the CodeQL policy applied in Phase 7.
    """


# ----------------------------------------------------------------------
# Backend protocol
# ----------------------------------------------------------------------


class WebSearchBackend(ABC):
    """Abstract search backend."""

    name: str = "abstract"

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Return up to ``max_results`` results.

        Each entry is ``{"title", "url", "content"}``. May return
        fewer than ``max_results`` if the engine has no more hits.
        """


# ----------------------------------------------------------------------
# DuckDuckGo HTML scraper (default, no API key)
# ----------------------------------------------------------------------


_DDG_ENDPOINT = "https://html.duckduckgo.com/html/"

# DDG wraps each result in a div.result with a .result__title > a,
# .result__url, and .result__snippet. The markup is stable enough
# that regex extraction is safer than beautifulsoup (no extra dep).
_DDG_RESULT_RE = re.compile(
    r'<a\s+[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a\s+[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)


def _strip_html(text: str) -> str:
    """Remove tags + decode entities without pulling in lxml."""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class DuckDuckGoBackend(WebSearchBackend):
    """Zero-config fallback — scrapes DDG's HTML-only endpoint.

    Accuracy is fine for LLM grounding but the HTML layout can change
    without notice. Prefer a proper API backend in production.
    """

    name = "duckduckgo"

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        payload = {"q": query, "kl": "us-en"}
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.post(
                    _DDG_ENDPOINT,
                    data=payload,
                    headers={
                        "User-Agent": "Mozilla/5.0 (HFL-bot) Python/httpx",
                        "Accept": "text/html",
                    },
                )
                resp.raise_for_status()
                body = resp.text
        except httpx.HTTPError as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            raise WebSearchError("web search backend unreachable") from exc

        results: list[dict[str, str]] = []
        for match in _DDG_RESULT_RE.finditer(body):
            url, title_html, snippet_html = match.groups()
            if len(results) >= max_results:
                break
            # DDG rewrites external URLs through /l/?uddg=… — unwrap.
            if url.startswith("//duckduckgo.com/l/") or "uddg=" in url:
                m = re.search(r"uddg=([^&]+)", url)
                if m:
                    from urllib.parse import unquote

                    url = unquote(m.group(1))
            if url.startswith("//"):
                url = "https:" + url
            results.append(
                {
                    "title": _strip_html(title_html),
                    "url": url,
                    "content": _strip_html(snippet_html),
                }
            )
        return results


# ----------------------------------------------------------------------
# Tavily API backend
# ----------------------------------------------------------------------


class TavilyBackend(WebSearchBackend):
    """https://tavily.com — JSON API, requires ``TAVILY_API_KEY``."""

    name = "tavily"

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            raise WebSearchError("TAVILY_API_KEY not set")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": key,
                        "query": query,
                        "max_results": max_results,
                        "include_raw_content": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Tavily search failed: %s", exc)
            raise WebSearchError("web search backend unreachable") from exc
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in data.get("results", [])[:max_results]
        ]


# ----------------------------------------------------------------------
# Brave Search API
# ----------------------------------------------------------------------


class BraveBackend(WebSearchBackend):
    """https://search.brave.com — JSON API, requires ``BRAVE_API_KEY``."""

    name = "brave"

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        key = os.environ.get("BRAVE_API_KEY")
        if not key:
            raise WebSearchError("BRAVE_API_KEY not set")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={"X-Subscription-Token": key, "Accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Brave search failed: %s", exc)
            raise WebSearchError("web search backend unreachable") from exc
        web = data.get("web", {})
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("description", ""),
            }
            for r in web.get("results", [])[:max_results]
        ]


# ----------------------------------------------------------------------
# SerpAPI
# ----------------------------------------------------------------------


class SerpAPIBackend(WebSearchBackend):
    """https://serpapi.com — requires ``SERPAPI_API_KEY``."""

    name = "serpapi"

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        key = os.environ.get("SERPAPI_API_KEY")
        if not key:
            raise WebSearchError("SERPAPI_API_KEY not set")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "q": query,
                        "num": max_results,
                        "engine": "google",
                        "api_key": key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("SerpAPI search failed: %s", exc)
            raise WebSearchError("web search backend unreachable") from exc
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "content": r.get("snippet", ""),
            }
            for r in data.get("organic_results", [])[:max_results]
        ]


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


_BACKENDS: dict[str, type[WebSearchBackend]] = {
    "duckduckgo": DuckDuckGoBackend,
    "ddg": DuckDuckGoBackend,
    "tavily": TavilyBackend,
    "brave": BraveBackend,
    "serpapi": SerpAPIBackend,
}


def get_backend(name: str | None = None) -> WebSearchBackend:
    """Resolve ``name`` (or ``HFL_WEB_SEARCH_BACKEND``) to an instance.

    Unknown names fall back to ``duckduckgo`` with a warning so the
    server stays functional rather than 500ing on startup.
    """
    raw = (name or os.environ.get("HFL_WEB_SEARCH_BACKEND") or "duckduckgo").lower()
    cls = _BACKENDS.get(raw)
    if cls is None:
        logger.warning("Unknown HFL_WEB_SEARCH_BACKEND=%r, falling back to duckduckgo", raw)
        cls = DuckDuckGoBackend
    return cls()


async def search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Convenience wrapper used by the route handler.

    Bounds ``max_results`` to [1, 10] per the Ollama contract.
    Returns the exact Ollama envelope:
    ``{"results": [{"title","url","content"}, ...]}``.
    """
    if not isinstance(query, str) or not query.strip():
        raise WebSearchError("query must be a non-empty string")
    max_results = max(1, min(10, int(max_results)))
    backend = get_backend()
    results = await backend.search(query.strip(), max_results)
    return {"results": results}

# SPDX-License-Identifier: Apache-2.0
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
    "WebSearchUpstreamError",
    "DuckDuckGoBackend",
    "TavilyBackend",
    "BraveBackend",
    "SerpAPIBackend",
    "get_backend",
    "backend_chain",
    "SearXNGBackend",
    "ExaBackend",
    "search",
]


class WebSearchError(RuntimeError):
    """Raised when a backend can't serve a request.

    Message is safe to surface to the client (no stack-trace info),
    matching the CodeQL policy applied in Phase 7.
    """


class WebSearchUpstreamError(WebSearchError):
    """The search service answered, but with no results page: a refusal,
    an anti-bot check, an outage. Not the caller's fault (a 502)."""


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
# Markers of DuckDuckGo's anti-bot page (the "anomaly" challenge).
_DDG_CHALLENGE_RE = re.compile(r"anomaly-modal|challenge-form|/anomaly\.js", re.IGNORECASE)

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
            logger.warning("DuckDuckGo search failed: %s", _redacted(exc))
            raise WebSearchUpstreamError("web search backend unreachable") from exc
        # DuckDuckGo answers a client it takes for a bot with HTTP 202 and a
        # challenge page — not an error to raise_for_status — which parsed
        # as no results: an empty list with 200 (local audit B38).
        if resp.status_code != 200 or _DDG_CHALLENGE_RE.search(body):
            raise WebSearchUpstreamError(
                f"DuckDuckGo refused the search (HTTP {resp.status_code}, its anti-bot "
                "check: it turns away scripted clients). Point HFL_SEARXNG_URL at a "
                "SearXNG instance (self-hosted, no key), or set EXA_API_KEY, "
                "TAVILY_API_KEY, BRAVE_API_KEY or SERPAPI_API_KEY."
            )

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
            logger.warning("Tavily search failed: %s", _redacted(exc))
            raise _classified(exc, "Tavily") from exc
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
            logger.warning("Brave search failed: %s", _redacted(exc))
            raise _classified(exc, "Brave") from exc
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
            logger.warning("SerpAPI search failed: %s", _redacted(exc))
            raise _classified(exc, "SerpAPI") from exc
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


def _redacted(exc: httpx.HTTPError) -> str:
    """Status + URL without its query: str(HTTPStatusError) carries the full
    URL, and SerpAPI's has ``api_key=`` in it — the key ended up in the log."""
    status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
    try:
        url = exc.request.url
        where = f"{url.scheme}://{url.host}{url.path}"
    except RuntimeError:  # an exception raised before any request was built
        where = "?"
    return f"HTTP {status} from {where}" if status else f"{type(exc).__name__} for {where}"


def _classified(exc: httpx.HTTPError, service: str) -> WebSearchError:
    """A rejected key is the caller's to fix (400); anything else — an
    outage, a refusal, no answer — is the service's (502). All used to be
    "web search backend unreachable", with 400."""
    status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
    if status in (401, 403):
        return WebSearchError(f"{service} rejected the API key (HTTP {status})")
    detail = f"HTTP {status}" if status else type(exc).__name__
    return WebSearchUpstreamError(f"{service} did not answer the search ({detail})")


# ----------------------------------------------------------------------
# SearXNG (self-hosted metasearch, no key)
# ----------------------------------------------------------------------


class SearXNGBackend(WebSearchBackend):
    """A SearXNG instance, at ``HFL_SEARXNG_URL``: open-source metasearch
    that one runs oneself — no account, no key, no third party choosing what
    HFL may search. Its JSON output must be enabled (``search.formats``)."""

    name = "searxng"

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        base = (os.environ.get("HFL_SEARXNG_URL") or "").rstrip("/")
        if not base:
            raise WebSearchError("HFL_SEARXNG_URL not set (the URL of a SearXNG instance)")
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(f"{base}/search", params={"q": query, "format": "json"})
                if resp.status_code == 403:
                    raise WebSearchError(
                        "the SearXNG instance refuses JSON: add json to search.formats "
                        "in its settings.yml"
                    )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("SearXNG search failed: %s", _redacted(exc))
            raise _classified(exc, "SearXNG") from exc
        except ValueError as exc:  # not JSON: an HTML page, a proxy's error
            raise WebSearchUpstreamError("SearXNG answered, but not in JSON") from exc
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "") or "",
            }
            for r in data.get("results", [])[:max_results]
        ]


# ----------------------------------------------------------------------
# Exa
# ----------------------------------------------------------------------


class ExaBackend(WebSearchBackend):
    """https://exa.ai — JSON API, requires ``EXA_API_KEY`` (a free monthly
    allowance, no card)."""

    name = "exa"

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        key = os.environ.get("EXA_API_KEY")
        if not key:
            raise WebSearchError("EXA_API_KEY not set")
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    "https://api.exa.ai/search",
                    headers={"x-api-key": key},
                    json={
                        "query": query,
                        "numResults": max_results,
                        "contents": {"highlights": {"maxCharacters": 400}},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("Exa search failed: %s", _redacted(exc))
            raise _classified(exc, "Exa") from exc
        return [
            {
                "title": r.get("title") or "",
                "url": r.get("url", ""),
                "content": " ".join(r.get("highlights") or []) or (r.get("text") or "")[:400],
            }
            for r in data.get("results", [])[:max_results]
        ]


_BACKENDS: dict[str, type[WebSearchBackend]] = {
    "duckduckgo": DuckDuckGoBackend,
    "ddg": DuckDuckGoBackend,
    "tavily": TavilyBackend,
    "brave": BraveBackend,
    "serpapi": SerpAPIBackend,
    "searxng": SearXNGBackend,
    "exa": ExaBackend,
}

# What each keyed backend needs before it is worth trying.
_NEEDS = {
    "searxng": "HFL_SEARXNG_URL",
    "tavily": "TAVILY_API_KEY",
    "exa": "EXA_API_KEY",
    "brave": "BRAVE_API_KEY",
    "serpapi": "SERPAPI_API_KEY",
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


def backend_chain(setting: str | None = None) -> list[WebSearchBackend]:
    """The backends to try, in order.

    ``HFL_WEB_SEARCH_BACKEND`` names one or a comma-separated list
    (``searxng,exa,duckduckgo``). Unset, the chain is built from what is
    configured: a SearXNG instance first (HFL_SEARXNG_URL), then each
    service whose key is set, and DuckDuckGo last — it needs nothing, but
    turns away clients it takes for bots (every scripted client, measured:
    httpx, curl, even its own Instant Answer API), so alone it was a search
    that did not work.
    """
    raw = setting if setting is not None else os.environ.get("HFL_WEB_SEARCH_BACKEND")
    if raw and raw.strip():
        names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    else:
        names = [n for n, var in _NEEDS.items() if os.environ.get(var)] + ["duckduckgo"]
    chain: list[WebSearchBackend] = []
    for name in names:
        cls = _BACKENDS.get(name)
        if cls is None:
            logger.warning("Unknown web search backend %r, skipped", name)
            continue
        if cls not in [type(b) for b in chain]:
            chain.append(cls())
    return chain or [DuckDuckGoBackend()]


async def search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Convenience wrapper used by the route handler.

    Bounds ``max_results`` to [1, 10] per the Ollama contract.
    Returns the exact Ollama envelope:
    ``{"results": [{"title","url","content"}, ...]}``, plus the backend
    that answered. Each backend of the chain is tried in turn; the error, if
    every one fails, says why each did.
    """
    if not isinstance(query, str) or not query.strip():
        raise WebSearchError("query must be a non-empty string")
    max_results = max(1, min(10, int(max_results)))
    chain = backend_chain()
    reasons: list[str] = []
    only_caller_errors = True
    for backend in chain:
        try:
            results = await backend.search(query.strip(), max_results)
        except WebSearchUpstreamError as exc:
            only_caller_errors = False
            reasons.append(f"{backend.name}: {exc}")
            continue
        except WebSearchError as exc:
            reasons.append(f"{backend.name}: {exc}")
            continue
        return {"results": results, "backend": backend.name}
    message = "; ".join(reasons)
    if len(chain) == 1:
        message = reasons[0].split(": ", 1)[1]
    raise (WebSearchError if only_caller_errors else WebSearchUpstreamError)(message)

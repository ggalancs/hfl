# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.tools.web_search`` backends and the factory (Phase 9)."""

from __future__ import annotations

import httpx
import pytest

from hfl.tools import web_search as ws

DDG_SAMPLE_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">Alpha Title</a>
  <a class="result__snippet" href="//x">Snippet A text.</a>
</div>
<div class="result">
  <a class="result__a" href="https://direct.example.org/b">Beta &amp; Gamma</a>
  <a class="result__snippet" href="#">Snippet B with <b>markup</b>.</a>
</div>
</body></html>
"""


TAVILY_SAMPLE = {
    "results": [
        {"title": "T1", "url": "https://t.example/1", "content": "body 1"},
        {"title": "T2", "url": "https://t.example/2", "content": "body 2"},
    ]
}


BRAVE_SAMPLE = {
    "web": {
        "results": [
            {"title": "B1", "url": "https://b.example/1", "description": "brave 1"},
            {"title": "B2", "url": "https://b.example/2", "description": "brave 2"},
        ]
    }
}


SERPAPI_SAMPLE = {
    "organic_results": [
        {"title": "S1", "link": "https://s.example/1", "snippet": "serp 1"},
        {"title": "S2", "link": "https://s.example/2", "snippet": "serp 2"},
    ]
}


def _make_handler(status_code, content=None, json_body=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if json_body is not None:
            return httpx.Response(status_code, json=json_body)
        return httpx.Response(status_code, text=content or "")

    return handler


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


class TestBackendFactory:
    def test_default_is_duckduckgo(self, monkeypatch):
        monkeypatch.delenv("HFL_WEB_SEARCH_BACKEND", raising=False)
        assert ws.get_backend().name == "duckduckgo"

    def test_env_selects_tavily(self, monkeypatch):
        monkeypatch.setenv("HFL_WEB_SEARCH_BACKEND", "tavily")
        assert ws.get_backend().name == "tavily"

    def test_unknown_falls_back_to_duckduckgo(self, monkeypatch, caplog):
        monkeypatch.setenv("HFL_WEB_SEARCH_BACKEND", "bogus")
        with caplog.at_level("WARNING"):
            backend = ws.get_backend()
        assert backend.name == "duckduckgo"

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("HFL_WEB_SEARCH_BACKEND", "duckduckgo")
        assert ws.get_backend("tavily").name == "tavily"


# ----------------------------------------------------------------------
# DuckDuckGo scraper
# ----------------------------------------------------------------------


class TestDuckDuckGoBackend:
    async def test_parses_titles_urls_snippets(self, monkeypatch):
        transport = httpx.MockTransport(_make_handler(200, DDG_SAMPLE_HTML))

        async def mock_client(*_a, **_k):
            return httpx.AsyncClient(transport=transport, follow_redirects=True)

        # Patch AsyncClient to use our mock transport.
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)

        backend = ws.DuckDuckGoBackend()
        results = await backend.search("q", max_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "Alpha Title"
        assert results[0]["url"] == "https://example.com/a"
        assert results[0]["content"] == "Snippet A text."
        assert results[1]["title"] == "Beta & Gamma"
        assert "markup" in results[1]["content"]

    async def test_respects_max_results(self, monkeypatch):
        transport = httpx.MockTransport(_make_handler(200, DDG_SAMPLE_HTML))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)

        backend = ws.DuckDuckGoBackend()
        results = await backend.search("q", max_results=1)
        assert len(results) == 1

    async def test_http_error_raises_search_error(self, monkeypatch):
        def raise_error(request):
            raise httpx.ConnectTimeout("nope")

        transport = httpx.MockTransport(raise_error)
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)

        backend = ws.DuckDuckGoBackend()
        with pytest.raises(ws.WebSearchError):
            await backend.search("q", max_results=5)


# ----------------------------------------------------------------------
# Tavily / Brave / SerpAPI
# ----------------------------------------------------------------------


class TestKeyedBackends:
    async def test_tavily_requires_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ws.WebSearchError):
            await ws.TavilyBackend().search("q", 5)

    async def test_brave_requires_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(ws.WebSearchError):
            await ws.BraveBackend().search("q", 5)

    async def test_serpapi_requires_key(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        with pytest.raises(ws.WebSearchError):
            await ws.SerpAPIBackend().search("q", 5)

    async def test_tavily_happy_path(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        transport = httpx.MockTransport(_make_handler(200, json_body=TAVILY_SAMPLE))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)

        results = await ws.TavilyBackend().search("q", 10)
        assert [r["title"] for r in results] == ["T1", "T2"]

    async def test_brave_happy_path(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "k")
        transport = httpx.MockTransport(_make_handler(200, json_body=BRAVE_SAMPLE))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)

        results = await ws.BraveBackend().search("q", 10)
        assert results[0]["url"] == "https://b.example/1"
        assert results[0]["content"] == "brave 1"

    async def test_serpapi_happy_path(self, monkeypatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "k")
        transport = httpx.MockTransport(_make_handler(200, json_body=SERPAPI_SAMPLE))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)

        results = await ws.SerpAPIBackend().search("q", 10)
        assert [r["url"] for r in results] == [
            "https://s.example/1",
            "https://s.example/2",
        ]


# ----------------------------------------------------------------------
# search() wrapper
# ----------------------------------------------------------------------


class TestSearchWrapper:
    async def test_rejects_empty_query(self):
        with pytest.raises(ws.WebSearchError):
            await ws.search("", 5)

    async def test_clamps_max_results(self, monkeypatch):
        captured = {}

        class _FakeBackend(ws.WebSearchBackend):
            name = "fake"

            async def search(self, query, max_results):  # noqa: D401
                captured["mr"] = max_results
                return []

        monkeypatch.setattr(ws, "get_backend", lambda: _FakeBackend())
        await ws.search("x", max_results=999)
        assert captured["mr"] == 10

        await ws.search("x", max_results=0)
        assert captured["mr"] == 1

    async def test_returns_ollama_shape(self, monkeypatch):
        class _FakeBackend(ws.WebSearchBackend):
            name = "fake"

            async def search(self, query, max_results):
                return [{"title": "t", "url": "u", "content": "c"}]

        monkeypatch.setattr(ws, "get_backend", lambda: _FakeBackend())
        payload = await ws.search("q", 5)
        assert "results" in payload
        assert payload["results"][0]["title"] == "t"

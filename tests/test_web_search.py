# SPDX-License-Identifier: Apache-2.0
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

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (202, "<html>please wait</html>"),  # what DDG answered the audit
            (200, '<form id="challenge-form"><div class="anomaly-modal"></div></form>'),
        ],
    )
    async def test_a_refusal_is_an_error_not_no_results(self, monkeypatch, status, body):
        """DuckDuckGo's anti-bot answer parsed as zero results: 200 with an
        empty list (local audit B38). It is an upstream error now."""
        transport = httpx.MockTransport(_make_handler(status, body))
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)
        with pytest.raises(ws.WebSearchUpstreamError, match="refused"):
            await ws.DuckDuckGoBackend().search("q", max_results=5)

    async def test_the_route_answers_a_refusal_502(self, monkeypatch):
        from fastapi.testclient import TestClient

        from hfl.api import routes_web
        from hfl.api.server import app

        async def refused(query, max_results):
            raise ws.WebSearchUpstreamError("DuckDuckGo refused the search")

        monkeypatch.setattr(routes_web, "search", refused)
        response = TestClient(app).post("/api/web_search", json={"query": "x"})
        assert response.status_code == 502 and "refused" in response.text


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

        monkeypatch.setattr(ws, "backend_chain", lambda: [_FakeBackend()])
        await ws.search("x", max_results=999)
        assert captured["mr"] == 10

        await ws.search("x", max_results=0)
        assert captured["mr"] == 1

    async def test_returns_ollama_shape(self, monkeypatch):
        class _FakeBackend(ws.WebSearchBackend):
            name = "fake"

            async def search(self, query, max_results):
                return [{"title": "t", "url": "u", "content": "c"}]

        monkeypatch.setattr(ws, "backend_chain", lambda: [_FakeBackend()])
        payload = await ws.search("q", 5)
        assert "results" in payload
        assert payload["results"][0]["title"] == "t"


# ----------------------------------------------------------------------
# The chain: what is configured, in order, with fallback
# ----------------------------------------------------------------------


def _mock_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(ws.httpx, "AsyncClient", _factory)


class TestChain:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        for var in (
            "HFL_WEB_SEARCH_BACKEND",
            "HFL_SEARXNG_URL",
            "TAVILY_API_KEY",
            "EXA_API_KEY",
            "BRAVE_API_KEY",
            "SERPAPI_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_nothing_configured_is_duckduckgo_alone(self):
        assert [b.name for b in ws.backend_chain()] == ["duckduckgo"]

    def test_what_is_configured_comes_first_duckduckgo_last(self, monkeypatch):
        monkeypatch.setenv("HFL_SEARXNG_URL", "http://localhost:8888")
        monkeypatch.setenv("EXA_API_KEY", "k")
        assert [b.name for b in ws.backend_chain()] == ["searxng", "exa", "duckduckgo"]

    def test_an_explicit_list_is_the_order(self, monkeypatch):
        monkeypatch.setenv("HFL_WEB_SEARCH_BACKEND", "exa, searxng, nope")
        assert [b.name for b in ws.backend_chain()] == ["exa", "searxng"]

    async def test_a_refusal_falls_through_to_the_next(self, monkeypatch):
        class Refuses(ws.WebSearchBackend):
            name = "first"

            async def search(self, query, max_results):
                raise ws.WebSearchUpstreamError("turned away")

        class Answers(ws.WebSearchBackend):
            name = "second"

            async def search(self, query, max_results):
                return [{"title": "t", "url": "u", "content": "c"}]

        monkeypatch.setattr(ws, "backend_chain", lambda: [Refuses(), Answers()])
        out = await ws.search("q")
        assert out["backend"] == "second" and out["results"][0]["url"] == "u"

    async def test_when_all_fail_each_reason_is_given(self, monkeypatch):
        class Refuses(ws.WebSearchBackend):
            name = "a"

            async def search(self, query, max_results):
                raise ws.WebSearchUpstreamError("turned away")

        class NoKey(ws.WebSearchBackend):
            name = "b"

            async def search(self, query, max_results):
                raise ws.WebSearchError("B_KEY not set")

        monkeypatch.setattr(ws, "backend_chain", lambda: [Refuses(), NoKey()])
        with pytest.raises(ws.WebSearchUpstreamError) as caught:
            await ws.search("q")
        assert "a: turned away" in str(caught.value) and "b: B_KEY not set" in str(caught.value)


class TestSearXNG:
    async def test_reads_its_json(self, monkeypatch):
        monkeypatch.setenv("HFL_SEARXNG_URL", "http://searx.local/")
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "Python", "url": "https://python.org", "content": "The language"}
                    ]
                },
            )

        _mock_transport(monkeypatch, handler)
        results = await ws.SearXNGBackend().search("python", 5)
        assert results == [
            {"title": "Python", "url": "https://python.org", "content": "The language"}
        ]
        assert seen["url"].startswith("http://searx.local/search?") and "format=json" in seen["url"]

    async def test_json_switched_off_says_how_to_switch_it_on(self, monkeypatch):
        monkeypatch.setenv("HFL_SEARXNG_URL", "http://searx.local")
        _mock_transport(monkeypatch, lambda r: httpx.Response(403, text="Forbidden"))
        with pytest.raises(ws.WebSearchError, match="search.formats"):
            await ws.SearXNGBackend().search("python", 5)


class TestExa:
    async def test_reads_highlights(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "k")
        seen = {}

        def handler(request):
            seen["key"] = request.headers.get("x-api-key")
            seen["body"] = request.read()
            return httpx.Response(
                200,
                json={
                    "results": [{"title": "Py", "url": "https://py", "highlights": ["one", "two"]}]
                },
            )

        _mock_transport(monkeypatch, handler)
        results = await ws.ExaBackend().search("python", 3)
        assert results == [{"title": "Py", "url": "https://py", "content": "one two"}]
        assert seen["key"] == "k" and b'"numResults":3' in seen["body"].replace(b" ", b"")

    async def test_a_rejected_key_is_the_callers_error(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "bad")
        _mock_transport(monkeypatch, lambda r: httpx.Response(401, json={"error": "unauthorized"}))
        with pytest.raises(ws.WebSearchError, match="rejected the API key") as caught:
            await ws.ExaBackend().search("python", 3)
        assert not isinstance(caught.value, ws.WebSearchUpstreamError)

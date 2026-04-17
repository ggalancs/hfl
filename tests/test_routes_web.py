# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HTTP contract tests for ``/api/web_search`` and ``/api/web_fetch``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state
from hfl.tools import web_fetch as wf
from hfl.tools import web_search as ws


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


# ----------------------------------------------------------------------
# /api/web_search
# ----------------------------------------------------------------------


class TestWebSearchRoute:
    def test_200_with_ollama_envelope(self, client, monkeypatch):
        class _FakeBackend(ws.WebSearchBackend):
            name = "fake"

            async def search(self, query, max_results):
                return [
                    {"title": "A", "url": "https://a.example", "content": "Ainfo"},
                    {"title": "B", "url": "https://b.example", "content": "Binfo"},
                ]

        monkeypatch.setattr(ws, "get_backend", lambda: _FakeBackend())

        resp = client.post(
            "/api/web_search",
            json={"query": "hfl", "max_results": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert body["results"][0]["title"] == "A"
        assert body["results"][0]["url"] == "https://a.example"

    def test_rejects_empty_query(self, client):
        resp = client.post("/api/web_search", json={"query": ""})
        assert resp.status_code == 422

    def test_max_results_out_of_range(self, client):
        resp = client.post("/api/web_search", json={"query": "q", "max_results": 100})
        assert resp.status_code == 422

    def test_backend_error_becomes_400(self, client, monkeypatch):
        class _BrokenBackend(ws.WebSearchBackend):
            name = "broken"

            async def search(self, query, max_results):
                raise ws.WebSearchError("backend unreachable")

        monkeypatch.setattr(ws, "get_backend", lambda: _BrokenBackend())
        resp = client.post("/api/web_search", json={"query": "q"})
        assert resp.status_code == 400

    def test_unexpected_error_becomes_500_generic(self, client, monkeypatch):
        class _ExplodingBackend(ws.WebSearchBackend):
            name = "boom"

            async def search(self, query, max_results):
                raise RuntimeError("leaked details here")

        monkeypatch.setattr(ws, "get_backend", lambda: _ExplodingBackend())
        resp = client.post("/api/web_search", json={"query": "q"})
        assert resp.status_code == 500
        # Traceback must not leak.
        assert "leaked details" not in resp.text


# ----------------------------------------------------------------------
# /api/web_fetch
# ----------------------------------------------------------------------


class TestWebFetchRoute:
    def test_200_happy_path(self, client, monkeypatch):
        async def _fake(url, **_k):
            return {
                "title": "T",
                "content": "body",
                "links": ["https://example.com/x"],
                "url": url,
            }

        monkeypatch.setattr(
            "hfl.api.routes_web.fetch",
            _fake,
        )

        resp = client.post("/api/web_fetch", json={"url": "https://example.com/page"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "T"
        assert body["links"] == ["https://example.com/x"]

    def test_bad_url_returns_400(self, client, monkeypatch):
        async def _reject(url, **_k):
            raise wf.WebFetchError("URL scheme must be http or https")

        monkeypatch.setattr("hfl.api.routes_web.fetch", _reject)
        resp = client.post("/api/web_fetch", json={"url": "ftp://x"})
        assert resp.status_code == 400

    def test_missing_body_422(self, client):
        resp = client.post("/api/web_fetch", json={})
        assert resp.status_code == 422

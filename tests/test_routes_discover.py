# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``GET /api/discover``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state


@dataclass
class _FakeCardData:
    license: str | None = None


@dataclass
class _FakeModelInfo:
    id: str
    likes: int = 0
    downloads: int = 0
    last_modified: datetime | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    gated: bool = False
    tags: list[str] = field(default_factory=list)
    card_data: _FakeCardData | None = None


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def fake_hf(monkeypatch):
    """Replace ``HfApi`` so no network hit happens."""
    api = MagicMock()
    api.list_models.return_value = iter(
        [
            _FakeModelInfo(
                id="meta-llama/Llama-3.1-8B-Instruct",
                likes=12000,
                downloads=5_000_000,
                tags=["llama", "instruct"],
                card_data=_FakeCardData(license="llama3.1"),
            ),
            _FakeModelInfo(
                id="Qwen/Qwen2.5-7B-Instruct",
                likes=8000,
                downloads=3_000_000,
                tags=["qwen", "instruct"],
                card_data=_FakeCardData(license="apache-2.0"),
            ),
            _FakeModelInfo(
                id="mlx-community/Llama-3.1-8B-Instruct-4bit",
                likes=200,
                downloads=50_000,
                tags=["mlx", "llama"],
            ),
            _FakeModelInfo(
                id="bartowski/Qwen2.5-7B-GGUF",
                likes=300,
                downloads=80_000,
                tags=["gguf", "qwen"],
            ),
        ]
    )
    monkeypatch.setattr("hfl.hub.discovery.HfApi", lambda: api, raising=False)

    # ``search_hub`` builds the api inside; we monkeypatch the import
    # path used by the function instead.
    from hfl.hub import discovery as discovery_module

    real_search = discovery_module.search_hub

    def patched_search(query, *, api=None):
        # Re-use the shared fake regardless of caller.
        return real_search(query, api=api or MagicMock(list_models=lambda **kw: iter([])))

    # Easier: replace HfApi at import site so search_hub picks the fake.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)

    # Reset list_models per test (iter exhausts after first use).
    yield api


class TestDiscoverHappyPath:
    def test_returns_typed_envelope(self, client, fake_hf):
        response = client.get("/api/discover")
        assert response.status_code == 200
        body = response.json()
        assert "query" in body
        assert "entries" in body
        assert "cached" in body
        assert "fetched_at" in body
        assert body["cached"] is False
        assert body["total"] == len(body["entries"])

    def test_filters_by_family(self, client, fake_hf):
        response = client.get("/api/discover?family=qwen")
        body = response.json()
        ids = [e["repo_id"] for e in body["entries"]]
        assert all("qwen" in i.lower() for i in ids)

    def test_filters_by_quantization(self, client, fake_hf):
        response = client.get("/api/discover?quantization=mlx")
        body = response.json()
        ids = [e["repo_id"] for e in body["entries"]]
        assert all("mlx" in i.lower() for i in ids)

    def test_min_likes_excludes_low_popularity(self, client, fake_hf):
        response = client.get("/api/discover?min_likes=1000")
        body = response.json()
        # The MLX/GGUF re-quants only have 200/300 likes — must be
        # excluded.
        ids = [e["repo_id"] for e in body["entries"]]
        for i in ids:
            assert "mlx-community" not in i
            assert "bartowski" not in i

    def test_locally_available_flag_present(self, client, fake_hf):
        response = client.get("/api/discover")
        body = response.json()
        for entry in body["entries"]:
            assert "locally_available" in entry
            # No models registered locally in this test, so all False.
            assert entry["locally_available"] is False


class TestDiscoverValidation:
    def test_negative_min_likes_is_400(self, client, fake_hf):
        response = client.get("/api/discover?min_likes=-1")
        assert response.status_code in (400, 422)

    def test_excessive_page_size_is_400(self, client, fake_hf):
        response = client.get("/api/discover?page_size=1000")
        assert response.status_code in (400, 422)


class TestDiscoverCaching:
    def test_second_call_marks_cached(self, client, fake_hf):
        first = client.get("/api/discover?q=qwen").json()
        assert first["cached"] is False

        # The first call exhausted the iterator; rebuild it for round 2.
        fake_hf.list_models.return_value = iter([])
        second = client.get("/api/discover?q=qwen").json()
        assert second["cached"] is True
        # And the entries match the first call (came from cache, not
        # the empty iterator we just installed).
        assert second["total"] == first["total"]

    def test_refresh_bypasses_cache(self, client, fake_hf):
        client.get("/api/discover?q=qwen").json()
        # Empty iterator on the second call → refresh forces the new
        # (empty) result, so cached=False AND total=0.
        fake_hf.list_models.return_value = iter([])
        second = client.get("/api/discover?q=qwen&refresh=true").json()
        assert second["cached"] is False
        assert second["total"] == 0


class TestDiscoverHubFailureSurfaces:
    def test_hub_exception_returns_503(self, client, monkeypatch):
        # Patch the symbol in the route module — that's where the
        # ``from ... import search_hub`` lives, so mutating the
        # source module's attribute would not be observable.
        from hfl.api import routes_discover as module

        def _boom(*args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr(module, "search_hub", _boom)

        response = client.get("/api/discover?q=different-key&refresh=true")
        assert response.status_code == 503

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``POST /api/batch`` (Phase 15 P2 — V2 row 8)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import GenerationResult
from hfl.models.manifest import ModelManifest
from hfl.models.registry import get_registry, reset_registry


@pytest.fixture
def client(temp_config):
    reset_state()
    reset_registry()
    yield TestClient(app)
    reset_state()


def _install(manifest, *, results: list[GenerationResult]):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    engine.generate = MagicMock(side_effect=results)
    state.engine = engine
    state.current_model = manifest
    return engine


def _register(name="m"):
    manifest = ModelManifest(
        name=name,
        repo_id=f"org/{name}",
        local_path="/tmp/x.gguf",
        format="gguf",
    )
    get_registry().add(manifest)
    return manifest


class TestBatchRoute:
    def test_sequential_execution_returns_both(self, client):
        manifest = _register()
        _install(
            manifest,
            results=[
                GenerationResult(text="A", tokens_generated=1, tokens_prompt=1),
                GenerationResult(text="B", tokens_generated=1, tokens_prompt=1),
            ],
        )
        with patch("hfl.api.routes_batch.load_llm") as load_mock:

            async def _noop(_name):
                return None, manifest

            load_mock.side_effect = _noop

            resp = client.post(
                "/api/batch",
                json={
                    "model": "m",
                    "requests": [
                        {"prompt": "first"},
                        {"prompt": "second"},
                    ],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "m"
        assert len(body["results"]) == 2
        assert body["results"][0]["response"] == "A"
        assert body["results"][1]["response"] == "B"

    def test_partial_failure_isolated(self, client):
        manifest = _register()
        engine = _install(
            manifest,
            results=[
                GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1),
                RuntimeError("boom"),
            ],
        )

        with patch("hfl.api.routes_batch.load_llm") as load_mock:

            async def _noop(_name):
                return None, manifest

            load_mock.side_effect = _noop

            resp = client.post(
                "/api/batch",
                json={
                    "model": "m",
                    "requests": [
                        {"prompt": "good"},
                        {"prompt": "bad"},
                    ],
                },
            )
        body = resp.json()
        assert body["results"][0]["response"] == "ok"
        assert "error" in body["results"][1]
        # ``engine.generate`` called twice even though the second crashed.
        assert engine.generate.call_count == 2

    def test_empty_batch_rejected(self, client):
        resp = client.post("/api/batch", json={"model": "m", "requests": []})
        assert resp.status_code == 422

    def test_missing_prompt_rejected(self, client):
        resp = client.post(
            "/api/batch",
            json={"model": "m", "requests": [{}]},
        )
        assert resp.status_code == 422

    def test_too_many_requests_rejected(self, client):
        resp = client.post(
            "/api/batch",
            json={
                "model": "m",
                "requests": [{"prompt": "x"}] * 257,
            },
        )
        assert resp.status_code == 422

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for ``GET /api/draft/recommend`` (V4 F5)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def patched_picker(monkeypatch):
    from hfl.hub.draft_picker import DraftPick

    pick = DraftPick(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        family="llama",
        parameter_estimate_b=1.0,
        quantization=None,
        rationale="test pick",
    )

    def _ret(*args, **kwargs):
        return pick

    from hfl.api import routes_draft as module

    monkeypatch.setattr(module, "pick_draft_for", _ret)
    return pick


class TestDraftRecommendShape:
    def test_returns_target_and_pick(self, client, patched_picker):
        response = client.get("/api/draft/recommend?model=meta-llama/Llama-3.1-70B-Instruct")
        assert response.status_code == 200
        body = response.json()
        assert body["target"] == "meta-llama/Llama-3.1-70B-Instruct"
        assert body["pick"] == {
            "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
            "family": "llama",
            "parameter_estimate_b": 1.0,
            "quantization": None,
            "rationale": "test pick",
        }

    def test_null_pick_when_picker_returns_none(self, client, monkeypatch):
        from hfl.api import routes_draft as module

        monkeypatch.setattr(module, "pick_draft_for", lambda *a, **k: None)

        body = client.get("/api/draft/recommend?model=anthropic/notamodel").json()
        assert body["pick"] is None
        assert body["target"] == "anthropic/notamodel"


class TestDraftRecommendValidation:
    def test_missing_model_param_is_422(self, client):
        response = client.get("/api/draft/recommend")
        assert response.status_code in (400, 422)

    def test_invalid_max_ratio_is_400(self, client):
        response = client.get("/api/draft/recommend?model=meta-llama/foo&max_ratio=2.0")
        assert response.status_code in (400, 422)

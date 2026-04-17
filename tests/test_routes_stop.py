# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``POST /api/stop``.

Contract:
- model name matches resident LLM → status=stopped, background
  unload scheduled.
- model name matches resident TTS → status=stopped, TTS unload
  scheduled.
- model name given but no match → status=not_loaded, no-op.
- model omitted / null, something is loaded → all loaded engines
  scheduled for unload, status=stopped.
- model omitted / null, nothing loaded → status=nothing_loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


class TestStopLLMByName:
    def test_stop_named_resident_llm(self, client, sample_manifest):
        """The resident LLM matching ``model`` is scheduled for unload."""
        state = get_state()
        engine = MagicMock(is_loaded=True)
        state.engine = engine
        state.current_model = sample_manifest

        # Intercept the private unload path so we can verify scheduling
        # without actually tearing down the mock engine via to_thread.
        original = state._llm_lock  # keep handle just in case

        response = client.post("/api/stop", json={"model": sample_manifest.name})
        assert response.status_code == 200
        body = response.json()
        assert body == {"status": "stopped", "model": sample_manifest.name}

        # Background task fired → engine.unload was called (TestClient
        # runs background tasks synchronously at response-commit time).
        engine.unload.assert_called_once()
        assert state._engine is None
        assert state._current_model is None
        assert state._llm_lock is original  # lock object unchanged

    def test_stop_named_nonresident_is_noop(self, client, sample_manifest):
        """A model name that isn't loaded just reports not_loaded."""
        state = get_state()
        state.engine = None
        state.current_model = None

        response = client.post("/api/stop", json={"model": "phantom:7b"})
        assert response.status_code == 200
        assert response.json() == {"status": "not_loaded", "model": "phantom:7b"}

    def test_stop_named_clears_keep_alive_deadline(self, client, sample_manifest):
        """Stopping a model should erase its keep_alive deadline so
        /api/ps stops showing a dangling expires_at.
        """
        from datetime import datetime, timedelta, timezone

        state = get_state()
        state.engine = MagicMock(is_loaded=True)
        state.current_model = sample_manifest
        state.set_keep_alive_deadline(
            sample_manifest.name, datetime.now(timezone.utc) + timedelta(minutes=5)
        )

        client.post("/api/stop", json={"model": sample_manifest.name})

        assert state.keep_alive_deadline_for(sample_manifest.name) is None


class TestStopAll:
    def test_stop_all_with_nothing_loaded(self, client):
        """Empty state → status=nothing_loaded, no side effects."""
        state = get_state()
        state.engine = None
        state.tts_engine = None
        state.current_model = None
        state.current_tts_model = None

        response = client.post("/api/stop", json={})
        assert response.status_code == 200
        assert response.json() == {"status": "nothing_loaded", "model": None}

    def test_stop_all_evicts_llm_and_tts(self, client, sample_manifest):
        """No model name + both loaded → both get unloaded."""
        from hfl.models.manifest import ModelManifest

        tts_manifest = ModelManifest(
            name="bark-small",
            repo_id="suno/bark-small",
            local_path="/tmp/bark",
            format="safetensors",
        )

        state = get_state()
        llm_engine = MagicMock(is_loaded=True)
        tts_engine = MagicMock(is_loaded=True)
        state.engine = llm_engine
        state.current_model = sample_manifest
        state.tts_engine = tts_engine
        state.current_tts_model = tts_manifest

        response = client.post("/api/stop", json={})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "stopped"
        # The "model" field concatenates evicted names for readability.
        assert sample_manifest.name in body["model"]
        assert "bark-small" in body["model"]

        # Both unloads ran.
        llm_engine.unload.assert_called_once()
        tts_engine.unload.assert_called_once()
        assert state._engine is None
        assert state._tts_engine is None


class TestStopTTSByName:
    def test_stop_matches_tts_engine(self, client):
        """Named model that matches the resident TTS engine."""
        from hfl.models.manifest import ModelManifest

        tts_manifest = ModelManifest(
            name="bark-small",
            repo_id="suno/bark-small",
            local_path="/tmp/bark",
            format="safetensors",
        )

        state = get_state()
        state.engine = None
        state.current_model = None
        tts_engine = MagicMock(is_loaded=True)
        state.tts_engine = tts_engine
        state.current_tts_model = tts_manifest

        response = client.post("/api/stop", json={"model": "bark-small"})
        assert response.status_code == 200
        assert response.json() == {"status": "stopped", "model": "bark-small"}

        tts_engine.unload.assert_called_once()
        assert state._tts_engine is None


class TestStopValidation:
    def test_empty_body_treated_as_stop_all(self, client):
        """Missing body is equivalent to ``{}`` per Ollama convention."""
        response = client.post("/api/stop", json={})
        assert response.status_code == 200

    def test_rejects_oversized_model_name(self, client):
        """Pydantic max_length=256 guards the field."""
        response = client.post("/api/stop", json={"model": "x" * 1024})
        assert response.status_code == 422


class TestStopReturns200AlwaysForIdempotency:
    """Clients expect /api/stop to be idempotent — repeated stops of
    the same name must not escalate to 4xx / 5xx. The router returns
    200 in every case the server can parse the body."""

    def test_stopping_twice_still_200(self, client, sample_manifest):
        state = get_state()
        state.engine = MagicMock(is_loaded=True)
        state.current_model = sample_manifest

        first = client.post("/api/stop", json={"model": sample_manifest.name})
        assert first.status_code == 200
        assert first.json()["status"] == "stopped"

        # After the first stop, the LLM is gone — second stop is
        # ``not_loaded``, still 200.
        second = client.post("/api/stop", json={"model": sample_manifest.name})
        assert second.status_code == 200
        assert second.json()["status"] == "not_loaded"

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for health check probe feature."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.engine.base import GenerationResult


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthDeepProbe:
    def test_deep_without_probe(self, client):
        """Deep health check without probe returns normal response."""
        resp = client.get("/health/deep")
        assert resp.status_code == 200
        data = resp.json()
        assert "llm" in data
        assert "probe" not in data["llm"]

    def test_deep_with_probe_no_model(self, client):
        """Probe with no model loaded skips inference test."""
        resp = client.get("/health/deep?probe=true")
        assert resp.status_code == 200
        data = resp.json()
        # No probe result when no model loaded
        assert "probe" not in data["llm"]

    @patch("hfl.api.routes_health.get_state")
    def test_deep_with_probe_model_loaded(self, mock_get_state, client):
        """Probe with model loaded runs inference test."""
        mock_state = MagicMock()
        mock_state.is_llm_loaded.return_value = True
        mock_state.is_tts_loaded.return_value = False
        mock_state.current_model = MagicMock(name="test-model")
        mock_state.current_tts_model = None
        mock_state.engine = MagicMock()
        mock_state.engine.generate.return_value = GenerationResult(
            text="ok", tokens_generated=1
        )
        mock_get_state.return_value = mock_state

        resp = client.get("/health/deep?probe=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm"]["probe"] == "ok"

    @patch("hfl.api.routes_health.get_state")
    def test_deep_probe_failure_reports_degraded(self, mock_get_state, client):
        """Probe failure reports degraded status."""
        mock_state = MagicMock()
        mock_state.is_llm_loaded.return_value = True
        mock_state.is_tts_loaded.return_value = False
        mock_state.current_model = MagicMock(name="test-model")
        mock_state.current_tts_model = None
        mock_state.engine = MagicMock()
        mock_state.engine.generate.side_effect = RuntimeError("GPU error")
        mock_get_state.return_value = mock_state

        resp = client.get("/health/deep?probe=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "failed" in data["llm"]["probe"]
        assert data["status"] == "degraded"

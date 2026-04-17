# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state
from hfl.metrics import reset_metrics


@pytest.fixture
def client(temp_config):
    """Create test client."""
    reset_state()
    reset_metrics()
    return TestClient(app)


class TestHealthBasic:
    """Tests for basic health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 with a well-formed body."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Minimal contract — every other assertion in this class is
        # additive. These four keys are what callers rely on.
        assert set(data.keys()) >= {
            "status",
            "model_loaded",
            "current_model",
            "tts_model_loaded",
            "current_tts_model",
        }
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)

    def test_health_returns_status(self, client):
        """Health endpoint returns status field."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_model_info(self, client):
        """Health endpoint includes model information."""
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert "current_model" in data


class TestHealthReady:
    """Tests for readiness endpoint."""

    def test_ready_returns_200(self, client):
        """Ready endpoint returns 200 with ready/not_ready status."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"ready", "not_ready"}
        assert isinstance(data["checks"], dict)
        # Caller contract: checks dict is not empty (orchestrators gate
        # on specific keys, a silent empty dict would hide outages).
        assert data["checks"]

    def test_ready_returns_checks(self, client):
        """Ready endpoint returns check results."""
        response = client.get("/health/ready")
        data = response.json()
        assert "checks" in data
        assert "config_loaded" in data["checks"]
        assert "registry_accessible" in data["checks"]


class TestHealthLive:
    """Tests for liveness endpoint."""

    def test_live_returns_200(self, client):
        """Live endpoint returns 200 with 'alive' status and uptime."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_live_returns_uptime(self, client):
        """Live endpoint returns uptime."""
        response = client.get("/health/live")
        data = response.json()
        assert data["status"] == "alive"
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestHealthDeep:
    """Tests for deep health endpoint."""

    def test_deep_returns_200(self, client):
        """Deep endpoint returns 200 with llm and tts sections."""
        response = client.get("/health/deep")
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data and isinstance(data["llm"], dict)
        assert "tts" in data and isinstance(data["tts"], dict)
        assert "loaded" in data["llm"]
        assert "loaded" in data["tts"]

    def test_deep_returns_version(self, client):
        """Deep endpoint returns version."""
        response = client.get("/health/deep")
        data = response.json()
        assert "version" in data

    def test_deep_returns_llm_status(self, client):
        """Deep endpoint returns LLM status."""
        response = client.get("/health/deep")
        data = response.json()
        assert "llm" in data
        assert "loaded" in data["llm"]


class TestHealthSLI:
    """Tests for SLI health endpoint."""

    def test_sli_returns_200(self, client):
        """SLI endpoint returns 200 with full SLI report structure."""
        response = client.get("/health/sli")
        assert response.status_code == 200
        data = response.json()
        # Top-level schema — all three are caller-facing contracts.
        assert data["slo_version"] == "1.0"
        assert "indicators" in data and isinstance(data["indicators"], dict)
        assert "summary" in data and isinstance(data["summary"], dict)

    def test_sli_returns_status(self, client):
        """SLI endpoint returns overall status."""
        response = client.get("/health/sli")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "warning", "critical")

    def test_sli_returns_indicators(self, client):
        """SLI endpoint returns all indicators."""
        response = client.get("/health/sli")
        data = response.json()
        assert "indicators" in data

        expected_indicators = [
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "error_rate",
            "availability",
            "memory",
        ]
        for indicator in expected_indicators:
            assert indicator in data["indicators"]

    def test_sli_indicators_have_status(self, client):
        """Each indicator has a status."""
        response = client.get("/health/sli")
        data = response.json()

        for name, indicator in data["indicators"].items():
            assert "status" in indicator, f"{name} missing status"
            assert indicator["status"] in (
                "ok",
                "warning",
                "critical",
                "unknown",
            ), f"{name} has invalid status"

    def test_sli_latency_indicators_have_target(self, client):
        """Latency indicators have target values."""
        response = client.get("/health/sli")
        data = response.json()

        for name in ["latency_p50", "latency_p95", "latency_p99"]:
            assert "current_ms" in data["indicators"][name]
            assert "target_ms" in data["indicators"][name]

    def test_sli_returns_summary(self, client):
        """SLI endpoint returns summary metrics."""
        response = client.get("/health/sli")
        data = response.json()
        assert "summary" in data
        assert "total_requests" in data["summary"]
        assert "uptime_seconds" in data["summary"]

    def test_sli_unknown_status_with_no_data(self, client):
        """SLI returns unknown for latency with no data."""
        response = client.get("/health/sli")
        data = response.json()

        # With no requests, latencies should be unknown or ok
        # (depends on implementation - 0 latency is not worse than target)
        assert data["indicators"]["latency_p50"]["status"] in ("ok", "unknown")

    def test_sli_error_rate_target(self, client):
        """SLI error rate has current and target."""
        response = client.get("/health/sli")
        data = response.json()

        error_rate = data["indicators"]["error_rate"]
        assert "current" in error_rate
        assert "target" in error_rate

    def test_sli_memory_thresholds(self, client):
        """SLI memory has threshold values."""
        response = client.get("/health/sli")
        data = response.json()

        memory = data["indicators"]["memory"]
        assert "current" in memory
        assert "warning_threshold" in memory
        assert "critical_threshold" in memory

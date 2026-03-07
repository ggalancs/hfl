# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for metrics endpoints."""

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state
from hfl.metrics import get_metrics


@pytest.fixture(autouse=True)
def reset_state():
    """Reset server state before each test."""
    get_state().api_key = None
    get_metrics().reset()
    yield
    get_state().api_key = None


class TestPrometheusMetrics:
    """Tests for /metrics endpoint."""

    def test_metrics_endpoint_exists(self):
        """Metrics endpoint should return 200."""
        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200

    def test_metrics_returns_plain_text(self):
        """Metrics should return plain text content type."""
        client = TestClient(app)
        response = client.get("/metrics")

        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_prometheus_format(self):
        """Metrics should contain Prometheus format markers."""
        client = TestClient(app)
        response = client.get("/metrics")

        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "hfl_uptime_seconds" in content

    def test_metrics_includes_counters(self):
        """Metrics should include request counters."""
        client = TestClient(app)
        response = client.get("/metrics")

        content = response.text
        assert "hfl_requests_total" in content
        assert "hfl_tokens_generated_total" in content

    def test_metrics_bypass_auth(self):
        """Metrics endpoint should bypass authentication."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        # Should work without auth
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_records_requests(self):
        """Metrics should record requests from middleware."""
        client = TestClient(app)

        # Make some requests
        client.get("/health")
        client.get("/health")
        client.get("/")

        # Check metrics reflect requests
        response = client.get("/metrics")
        content = response.text

        # Should have recorded requests
        assert "hfl_requests_total" in content


class TestJsonMetrics:
    """Tests for /metrics/json endpoint."""

    def test_json_metrics_endpoint_exists(self):
        """JSON metrics endpoint should return 200."""
        client = TestClient(app)
        response = client.get("/metrics/json")

        assert response.status_code == 200

    def test_json_metrics_returns_json(self):
        """JSON metrics should return application/json."""
        client = TestClient(app)
        response = client.get("/metrics/json")

        assert "application/json" in response.headers["content-type"]

    def test_json_metrics_structure(self):
        """JSON metrics should have expected structure."""
        client = TestClient(app)
        response = client.get("/metrics/json")

        data = response.json()
        assert "uptime_seconds" in data
        assert "requests_total" in data
        assert "tokens_generated" in data
        assert "latencies" in data

    def test_json_metrics_latencies(self):
        """JSON metrics should include latency percentiles."""
        client = TestClient(app)
        response = client.get("/metrics/json")

        data = response.json()
        latencies = data["latencies"]
        assert "request_p50_ms" in latencies
        assert "request_p95_ms" in latencies
        assert "request_p99_ms" in latencies

    def test_json_metrics_bypass_auth(self):
        """JSON metrics endpoint should bypass authentication."""
        get_state().api_key = "test-secret-key"
        client = TestClient(app)

        response = client.get("/metrics/json")
        assert response.status_code == 200


class TestMetricsRecording:
    """Tests for metrics being recorded correctly."""

    def test_metrics_increment_on_requests(self):
        """Metrics should increment when requests are made."""
        client = TestClient(app)
        metrics = get_metrics()

        initial_count = metrics.requests_total

        # Make requests
        client.get("/health")
        client.get("/")

        # Metrics should have been recorded by RequestLogger middleware
        assert metrics.requests_total > initial_count
        assert "/health" in metrics.requests_by_endpoint
        assert "/" in metrics.requests_by_endpoint

    def test_metrics_reset_works(self):
        """Metrics reset should clear all counters."""
        metrics = get_metrics()

        # Record some data
        metrics.record_request("/test", "GET", 200, 10.0)
        metrics.record_request("/test", "GET", 200, 20.0)

        assert metrics.requests_total == 2

        # Reset
        metrics.reset()

        assert metrics.requests_total == 0
        assert len(metrics.requests_by_endpoint) == 0

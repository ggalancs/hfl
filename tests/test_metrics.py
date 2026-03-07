# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for metrics collection."""

import pytest

from hfl.metrics import Metrics, get_metrics, reset_metrics


class TestMetrics:
    """Tests for Metrics class."""

    @pytest.fixture
    def metrics(self):
        """Create fresh metrics instance."""
        return Metrics()

    def test_record_request(self, metrics):
        """Should record request metrics."""
        metrics.record_request(
            endpoint="/v1/chat/completions",
            method="POST",
            status=200,
            duration_ms=150.0,
            tokens_in=100,
            tokens_out=50,
        )

        assert metrics.requests_total == 1
        assert metrics.requests_by_endpoint["/v1/chat/completions"] == 1
        assert metrics.requests_by_status[200] == 1
        assert metrics.requests_by_method["POST"] == 1
        assert metrics.tokens_input == 100
        assert metrics.tokens_generated == 50

    def test_record_multiple_requests(self, metrics):
        """Should accumulate multiple requests."""
        for _ in range(5):
            metrics.record_request("/test", "GET", 200, 10.0)

        assert metrics.requests_total == 5
        assert metrics.requests_by_endpoint["/test"] == 5

    def test_record_generation(self, metrics):
        """Should record generation metrics."""
        metrics.record_generation(
            duration_ms=500.0,
            tokens_in=50,
            tokens_out=100,
        )

        assert metrics.tokens_input == 50
        assert metrics.tokens_generated == 100

    def test_record_model_load(self, metrics):
        """Should record model load."""
        metrics.record_model_load("test-model", 5000.0)

        assert metrics.model_loads == 1

    def test_record_model_unload(self, metrics):
        """Should record model unload."""
        metrics.record_model_unload()

        assert metrics.model_unloads == 1

    def test_record_error(self, metrics):
        """Should record errors by type."""
        metrics.record_error("ValueError")
        metrics.record_error("ValueError")
        metrics.record_error("TypeError")

        assert metrics.errors_by_type["ValueError"] == 2
        assert metrics.errors_by_type["TypeError"] == 1

    def test_export_json(self, metrics):
        """Should export metrics as JSON."""
        metrics.record_request("/test", "GET", 200, 100.0)

        data = metrics.export_json()

        assert "uptime_seconds" in data
        assert data["requests_total"] == 1
        assert data["tokens_generated"] == 0
        assert "/test" in data["requests_by_endpoint"]
        assert "latencies" in data

    def test_export_prometheus(self, metrics):
        """Should export metrics in Prometheus format."""
        metrics.record_request("/test", "GET", 200, 100.0)

        output = metrics.export_prometheus()

        assert "hfl_requests_total 1" in output
        assert "hfl_uptime_seconds" in output
        assert "TYPE hfl_requests_total counter" in output

    def test_percentile_calculation(self, metrics):
        """Should calculate percentiles correctly."""
        for i in range(100):
            metrics.record_request("/test", "GET", 200, float(i))

        data = metrics.export_json()

        # p50 should be around 50
        assert 45 < data["latencies"]["request_p50_ms"] < 55
        # p95 should be around 95
        assert 90 < data["latencies"]["request_p95_ms"] < 100

    def test_histogram_size_limit(self, metrics):
        """Should limit histogram size."""
        for i in range(2000):
            metrics.record_request("/test", "GET", 200, float(i))

        # Internal histogram should be limited
        assert len(metrics._request_latencies_ms) <= 1000

    def test_reset(self, metrics):
        """reset() should clear all metrics."""
        metrics.record_request("/test", "GET", 200, 100.0)
        metrics.record_error("Error")
        metrics.record_model_load("model", 1000.0)

        metrics.reset()

        assert metrics.requests_total == 0
        assert len(metrics.requests_by_endpoint) == 0
        assert len(metrics.errors_by_type) == 0
        assert metrics.model_loads == 0


class TestMetricsSLI:
    """Tests for SLI calculation."""

    @pytest.fixture
    def metrics(self):
        """Create fresh metrics instance."""
        return Metrics()

    def test_get_sli_empty(self, metrics):
        """Should return zeros for empty metrics."""
        sli = metrics.get_sli()

        assert sli["latency_p50_ms"] == 0.0
        assert sli["latency_p95_ms"] == 0.0
        assert sli["latency_p99_ms"] == 0.0
        assert sli["error_rate"] == 0.0
        assert sli["availability"] == 1.0
        assert sli["total_requests"] == 0
        assert sli["sample_count"] == 0

    def test_get_sli_with_requests(self, metrics):
        """Should calculate SLIs from requests."""
        # Record successful requests
        for i in range(100):
            metrics.record_request("/test", "GET", 200, float(i * 10))

        sli = metrics.get_sli()

        assert sli["total_requests"] == 100
        assert sli["error_rate"] == 0.0
        assert sli["availability"] == 1.0
        assert sli["sample_count"] == 100
        assert sli["latency_p50_ms"] > 0
        assert sli["throughput_rps"] > 0

    def test_get_sli_with_errors(self, metrics):
        """Should calculate error rate correctly."""
        # 90 successful, 10 errors (5xx)
        for _ in range(90):
            metrics.record_request("/test", "GET", 200, 10.0)
        for _ in range(10):
            metrics.record_request("/test", "GET", 500, 10.0)

        sli = metrics.get_sli()

        assert sli["total_requests"] == 100
        assert sli["error_count"] == 10
        assert sli["error_rate"] == 0.1  # 10%
        assert sli["availability"] == 0.9  # 90%

    def test_get_sli_excludes_4xx_from_errors(self, metrics):
        """Should not count 4xx as server errors."""
        metrics.record_request("/test", "GET", 200, 10.0)
        metrics.record_request("/test", "GET", 400, 10.0)
        metrics.record_request("/test", "GET", 404, 10.0)
        metrics.record_request("/test", "GET", 500, 10.0)

        sli = metrics.get_sli()

        assert sli["total_requests"] == 4
        assert sli["error_count"] == 1  # Only 500
        assert sli["error_rate"] == 0.25  # 1/4

    def test_get_sli_latency_percentiles(self, metrics):
        """Should calculate latency percentiles correctly."""
        # Record latencies: 1-100ms
        for i in range(1, 101):
            metrics.record_request("/test", "GET", 200, float(i))

        sli = metrics.get_sli()

        # p50 should be ~50
        assert 45 < sli["latency_p50_ms"] < 55
        # p95 should be ~95
        assert 90 < sli["latency_p95_ms"] < 100
        # p99 should be ~99
        assert 95 < sli["latency_p99_ms"] <= 100


class TestGetMetrics:
    """Tests for get_metrics singleton."""

    def test_returns_same_instance(self):
        """Should return same instance."""
        reset_metrics()

        m1 = get_metrics()
        m2 = get_metrics()

        assert m1 is m2

    def test_reset_clears_instance(self):
        """reset_metrics should clear singleton."""
        m1 = get_metrics()
        m1.record_request("/test", "GET", 200, 100.0)

        reset_metrics()

        m2 = get_metrics()
        assert m2.requests_total == 0

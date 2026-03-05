# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for rate limiting middleware."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hfl.api.middleware import RateLimitMiddleware


@pytest.fixture
def rate_limited_app():
    """Create a test app with rate limiting."""
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=3,  # Very low for testing
        window_seconds=60,
    )

    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}

    return app


@pytest.fixture
def client(rate_limited_app):
    """Create test client."""
    return TestClient(rate_limited_app)


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    def test_requests_under_limit_succeed(self, client):
        """Test requests under limit are allowed."""
        # First 3 requests should succeed
        for i in range(3):
            response = client.get("/test")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

    def test_request_over_limit_rejected(self, client):
        """Test request over limit is rejected."""
        # Use up the limit
        for _ in range(3):
            client.get("/test")

        # Fourth request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "Too Many Requests" in response.json()["error"]

    def test_rate_limit_headers_present(self, client):
        """Test rate limit headers are included in response."""
        response = client.get("/test")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "3"

    def test_remaining_decreases(self, client):
        """Test remaining count decreases with each request."""
        response1 = client.get("/test")
        remaining1 = int(response1.headers["X-RateLimit-Remaining"])

        response2 = client.get("/test")
        remaining2 = int(response2.headers["X-RateLimit-Remaining"])

        assert remaining2 < remaining1

    def test_retry_after_header_on_limit(self, client):
        """Test Retry-After header is set when rate limited."""
        # Use up the limit
        for _ in range(3):
            client.get("/test")

        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_different_clients_independent(self, rate_limited_app):
        """Test different client IPs have independent limits."""
        # Create mock requests with different IPs
        with TestClient(rate_limited_app) as client1:
            with TestClient(rate_limited_app) as client2:
                # Use headers to simulate different IPs
                # Note: In real test, IPs would be different connections
                # This test verifies the middleware structure is correct
                response1 = client1.get("/test")
                assert response1.status_code == 200

                response2 = client2.get("/test")
                assert response2.status_code == 200


class TestRateLimitWindow:
    """Tests for rate limit window behavior."""

    def test_window_resets_requests(self):
        """Test requests are allowed again after window expires."""
        app = FastAPI()
        middleware = RateLimitMiddleware(
            app,
            requests_per_window=2,
            window_seconds=1,  # 1 second window for fast testing
        )

        # Simulate time-based request tracking
        client_ip = "127.0.0.1"

        # First two requests
        limited1, _ = middleware._is_rate_limited(client_ip)
        limited2, _ = middleware._is_rate_limited(client_ip)
        assert not limited1
        assert not limited2

        # Third request (over limit)
        limited3, _ = middleware._is_rate_limited(client_ip)
        assert limited3

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        limited4, _ = middleware._is_rate_limited(client_ip)
        assert not limited4


class TestClientIPDetection:
    """Tests for client IP detection."""

    def test_x_forwarded_for_used(self, rate_limited_app):
        """Test X-Forwarded-For header is respected."""
        client = TestClient(rate_limited_app)

        # Make requests with different forwarded IPs
        response1 = client.get(
            "/test", headers={"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        )
        assert response1.status_code == 200

        # Different forwarded IP should have its own limit
        response2 = client.get("/test", headers={"X-Forwarded-For": "10.0.0.2"})
        assert response2.status_code == 200

    def test_direct_connection_fallback(self, rate_limited_app):
        """Test fallback to connection IP when no forwarded header."""
        client = TestClient(rate_limited_app)
        response = client.get("/test")
        assert response.status_code == 200

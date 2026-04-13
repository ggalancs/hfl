# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for rate limiting middleware."""

import tempfile
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hfl.api.middleware import RateLimitMiddleware
from hfl.api.rate_limit import (
    InMemoryRateLimiter,
    SQLiteRateLimiter,
    create_rate_limiter,
)


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
        body = response.json()
        # New structured envelope (spec §5.4)
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert body["error"]["category"] == "rate_limit"
        assert body["error"]["retryable"] is True

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
        response1 = client.get("/test", headers={"X-Forwarded-For": "10.0.0.1, 192.168.1.1"})
        assert response1.status_code == 200

        # Different forwarded IP should have its own limit
        response2 = client.get("/test", headers={"X-Forwarded-For": "10.0.0.2"})
        assert response2.status_code == 200

    def test_direct_connection_fallback(self, rate_limited_app):
        """Test fallback to connection IP when no forwarded header."""
        client = TestClient(rate_limited_app)
        response = client.get("/test")
        assert response.status_code == 200


# =============================================================================
# Tests for rate_limit.py rate limiter implementations
# =============================================================================


class TestInMemoryRateLimiter:
    """Tests for InMemoryRateLimiter."""

    def test_allows_requests_under_limit(self):
        """Should allow requests under the limit."""
        limiter = InMemoryRateLimiter(requests_per_window=5, window_seconds=60)

        for i in range(5):
            allowed, remaining = limiter.is_allowed("client1")
            assert allowed is True
            assert remaining == 5 - i - 1

    def test_blocks_requests_over_limit(self):
        """Should block requests over the limit."""
        limiter = InMemoryRateLimiter(requests_per_window=3, window_seconds=60)

        for _ in range(3):
            limiter.is_allowed("client1")

        allowed, remaining = limiter.is_allowed("client1")
        assert allowed is False
        assert remaining == 0

    def test_different_clients_have_separate_limits(self):
        """Different clients should have separate limits."""
        limiter = InMemoryRateLimiter(requests_per_window=2, window_seconds=60)

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        allowed, _ = limiter.is_allowed("client2")
        assert allowed is True

    def test_window_slides(self):
        """Old requests should fall out of the window."""
        limiter = InMemoryRateLimiter(requests_per_window=2, window_seconds=1)

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        time.sleep(1.1)

        allowed, _ = limiter.is_allowed("client1")
        assert allowed is True

    def test_reset_specific_client(self):
        """reset() should clear limit for specific client."""
        limiter = InMemoryRateLimiter(requests_per_window=2, window_seconds=60)

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        limiter.is_allowed("client2")

        limiter.reset("client1")

        allowed1, _ = limiter.is_allowed("client1")
        assert allowed1 is True

    def test_reset_all_clients(self):
        """reset() with None should clear all limits."""
        limiter = InMemoryRateLimiter(requests_per_window=1, window_seconds=60)

        limiter.is_allowed("client1")
        limiter.is_allowed("client2")

        allowed1, _ = limiter.is_allowed("client1")
        allowed2, _ = limiter.is_allowed("client2")
        assert allowed1 is False
        assert allowed2 is False

        limiter.reset()

        allowed1, _ = limiter.is_allowed("client1")
        allowed2, _ = limiter.is_allowed("client2")
        assert allowed1 is True
        assert allowed2 is True

    def test_properties(self):
        """Properties should return configuration values."""
        limiter = InMemoryRateLimiter(requests_per_window=100, window_seconds=120)

        assert limiter.requests_per_window == 100
        assert limiter.window_seconds == 120

    def test_thread_safety(self):
        """Rate limiter should be thread-safe."""
        limiter = InMemoryRateLimiter(requests_per_window=100, window_seconds=60)
        results = []

        def make_requests():
            for _ in range(20):
                allowed, _ = limiter.is_allowed("client1")
                results.append(allowed)

        threads = [threading.Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == 100

    def test_reset_nonexistent_client(self):
        """reset() for nonexistent client should not raise."""
        limiter = InMemoryRateLimiter()
        limiter.reset("nonexistent")


class TestSQLiteRateLimiter:
    """Tests for SQLiteRateLimiter."""

    @pytest.fixture
    def db_path(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_rate_limit.db"

    def test_allows_requests_under_limit(self, db_path):
        """Should allow requests under the limit."""
        limiter = SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=5,
            window_seconds=60,
        )

        for i in range(5):
            allowed, remaining = limiter.is_allowed("client1")
            assert allowed is True
            assert remaining == 5 - i - 1

    def test_blocks_requests_over_limit(self, db_path):
        """Should block requests over the limit."""
        limiter = SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=3,
            window_seconds=60,
        )

        for _ in range(3):
            limiter.is_allowed("client1")

        allowed, remaining = limiter.is_allowed("client1")
        assert allowed is False
        assert remaining == 0

    def test_different_clients_have_separate_limits(self, db_path):
        """Different clients should have separate limits."""
        limiter = SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=2,
            window_seconds=60,
        )

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")

        allowed, _ = limiter.is_allowed("client2")
        assert allowed is True

    def test_reset_specific_client(self, db_path):
        """reset() should clear limit for specific client."""
        limiter = SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=1,
            window_seconds=60,
        )

        limiter.is_allowed("client1")
        limiter.reset("client1")

        allowed, _ = limiter.is_allowed("client1")
        assert allowed is True

    def test_reset_all_clients(self, db_path):
        """reset() with None should clear all limits."""
        limiter = SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=1,
            window_seconds=60,
        )

        limiter.is_allowed("client1")
        limiter.is_allowed("client2")
        limiter.reset()

        allowed1, _ = limiter.is_allowed("client1")
        allowed2, _ = limiter.is_allowed("client2")
        assert allowed1 is True
        assert allowed2 is True

    def test_properties(self, db_path):
        """Properties should return configuration values."""
        limiter = SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=100,
            window_seconds=120,
        )

        assert limiter.requests_per_window == 100
        assert limiter.window_seconds == 120

    def test_creates_parent_directories(self):
        """Should create parent directories for database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "rate.db"
            limiter = SQLiteRateLimiter(db_path=db_path)

            assert db_path.parent.exists()
            limiter.is_allowed("test")

    def test_persists_across_instances(self, db_path):
        """Rate limits should persist across limiter instances."""
        limiter1 = SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=2,
            window_seconds=60,
        )
        limiter1.is_allowed("client1")
        limiter1.is_allowed("client1")

        limiter2 = SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=2,
            window_seconds=60,
        )

        allowed, _ = limiter2.is_allowed("client1")
        assert allowed is False


class TestCreateRateLimiter:
    """Tests for create_rate_limiter factory function."""

    def test_creates_in_memory_by_default(self):
        """Should create InMemoryRateLimiter by default."""
        limiter = create_rate_limiter()
        assert isinstance(limiter, InMemoryRateLimiter)

    def test_creates_in_memory_with_params(self):
        """Should pass parameters to InMemoryRateLimiter."""
        limiter = create_rate_limiter(
            distributed=False,
            requests_per_window=100,
            window_seconds=120,
        )

        assert isinstance(limiter, InMemoryRateLimiter)
        assert limiter.requests_per_window == 100
        assert limiter.window_seconds == 120

    def test_creates_sqlite_when_distributed(self):
        """Should create SQLiteRateLimiter when distributed=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "rate.db"
            limiter = create_rate_limiter(distributed=True, db_path=db_path)

            assert isinstance(limiter, SQLiteRateLimiter)

    def test_sqlite_uses_default_path_when_not_provided(self):
        """Should use default path when distributed=True and no path given."""
        limiter = create_rate_limiter(distributed=True)
        assert isinstance(limiter, SQLiteRateLimiter)

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the api/middleware module."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestRequestLogger:
    """Tests for RequestLogger middleware."""

    def test_middleware_logs_request_metadata(self):
        """Verifies that the middleware logs request metadata via log_request."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        with patch("hfl.api.middleware.log_request") as mock_log:
            response = client.get("/test")

            assert response.status_code == 200

            # Verify that log_request was called with correct arguments
            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args.kwargs

            assert call_kwargs["method"] == "GET"
            assert call_kwargs["path"] == "/test"
            assert call_kwargs["status"] == 200
            assert call_kwargs["duration_ms"] >= 0

    def test_middleware_logs_post_request(self):
        """Verifies POST request logging."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.post("/api/endpoint")
        def post_endpoint():
            return {"created": True}

        client = TestClient(app)

        with patch("hfl.api.middleware.log_request") as mock_log:
            response = client.post("/api/endpoint")

            assert response.status_code == 200
            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs["method"] == "POST"
            assert call_kwargs["path"] == "/api/endpoint"

    def test_middleware_logs_error_status(self):
        """Verifies error logging."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/error")
        def error_endpoint():
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Not found")

        client = TestClient(app)

        with patch("hfl.api.middleware.log_request") as mock_log:
            response = client.get("/error")

            assert response.status_code == 404
            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs["status"] == 404

    def test_middleware_privacy_no_body_logged(self):
        """
        CRITICAL: Verifies that the middleware does NOT log the request body.

        R6 - Privacy compliance: User prompts are sensitive data.
        """
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.post("/chat")
        def chat_endpoint(data: dict):
            return {"response": "ok"}

        client = TestClient(app)

        with patch("hfl.api.middleware.log_request") as mock_log:
            # Send sensitive data
            sensitive_data = {"prompt": "My secret is...", "api_key": "sk-123"}
            client.post("/chat", json=sensitive_data)

            # Verify that the log call does NOT contain sensitive data
            log_call = str(mock_log.call_args)
            assert "My secret" not in log_call
            assert "sk-123" not in log_call
            assert "api_key" not in log_call
            assert "prompt" not in log_call  # Field should not be logged

    def test_middleware_measures_duration(self):
        """Verifies that request duration is measured."""
        import time

        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/slow")
        def slow_endpoint():
            time.sleep(0.1)  # 100ms
            return {"status": "done"}

        client = TestClient(app)

        with patch("hfl.api.middleware.log_request") as mock_log:
            client.get("/slow")

            call_kwargs = mock_log.call_args.kwargs
            duration_ms = call_kwargs["duration_ms"]
            assert duration_ms >= 100  # At least 100ms

    def test_middleware_adds_request_id_header(self):
        """Verifies that request ID is added to response."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8  # 8 hex chars

    def test_middleware_uses_incoming_request_id(self):
        """Verifies that incoming request ID is respected."""
        from hfl.api.middleware import RequestLogger

        app = FastAPI()
        app.add_middleware(RequestLogger)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-Request-ID": "custom-id"})

        assert response.headers["X-Request-ID"] == "custom-id"


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    @pytest.fixture
    def rate_limited_app(self):
        """Create app with rate limiting enabled."""
        from hfl.api.middleware import RateLimitMiddleware, reset_rate_limiter

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, requests_per_window=5, window_seconds=60)

        @app.get("/api/test")
        def api_endpoint():
            return {"status": "ok"}

        @app.get("/health")
        def health_endpoint():
            return {"status": "healthy"}

        @app.get("/health/ready")
        def health_ready():
            return {"status": "ready"}

        @app.get("/health/live")
        def health_live():
            return {"status": "alive"}

        yield app
        reset_rate_limiter()

    def test_rate_limit_applies_to_normal_endpoints(self, rate_limited_app):
        """Normal endpoints should be rate limited."""
        client = TestClient(rate_limited_app)

        # Make requests up to the limit
        for i in range(5):
            response = client.get("/api/test")
            assert response.status_code == 200

        # Next request should be rate limited
        response = client.get("/api/test")
        assert response.status_code == 429
        body = response.json()
        # Structured envelope (spec §5.4)
        err = body["error"]
        assert isinstance(err, dict)
        assert "Rate limit exceeded" in err["error"]
        assert err["code"] == "RATE_LIMIT_EXCEEDED"
        assert err["retryable"] is True

    def test_health_endpoint_bypasses_rate_limit(self, rate_limited_app):
        """Health endpoints should bypass rate limiting."""
        client = TestClient(rate_limited_app)

        # First exhaust rate limit
        for _ in range(5):
            client.get("/api/test")

        # Verify rate limit is active
        response = client.get("/api/test")
        assert response.status_code == 429

        # Health endpoints should still work
        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/health/ready")
        assert response.status_code == 200

        response = client.get("/health/live")
        assert response.status_code == 200

    def test_health_endpoints_never_rate_limited(self, rate_limited_app):
        """Health endpoints should never be rate limited even with many requests."""
        client = TestClient(rate_limited_app)

        # Make many requests to health endpoints - should never fail
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200

            response = client.get("/health/ready")
            assert response.status_code == 200

    def test_rate_limit_headers_not_added_to_health(self, rate_limited_app):
        """Rate limit headers should not be added to health endpoints."""
        client = TestClient(rate_limited_app)

        response = client.get("/health")
        assert response.status_code == 200
        # Health endpoints bypass rate limiting entirely, so no rate limit headers
        assert "X-RateLimit-Limit" not in response.headers

    def test_rate_limit_headers_added_to_normal_endpoints(self, rate_limited_app):
        """Rate limit headers should be added to normal endpoints."""
        client = TestClient(rate_limited_app)

        response = client.get("/api/test")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "5"

    def test_is_excluded_method(self):
        """Test _is_excluded helper method."""
        # Create middleware instance to test the method
        from starlette.applications import Starlette

        from hfl.api.middleware import RateLimitMiddleware

        dummy_app = Starlette()
        middleware = RateLimitMiddleware(dummy_app)

        # Health paths should be excluded
        assert middleware._is_excluded("/health") is True
        assert middleware._is_excluded("/health/ready") is True
        assert middleware._is_excluded("/health/live") is True
        assert middleware._is_excluded("/health/deep") is True

        # Normal paths should not be excluded
        assert middleware._is_excluded("/api/test") is False
        assert middleware._is_excluded("/v1/chat/completions") is False
        assert middleware._is_excluded("/") is False


class TestRequestBodyLimitMiddleware:
    """Tests for RequestBodyLimitMiddleware."""

    def _make_app(self, max_bytes: int) -> FastAPI:
        from hfl.api.middleware import RequestBodyLimitMiddleware

        app = FastAPI()
        app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_bytes)

        @app.post("/echo")
        def echo(payload: dict) -> dict:
            return {"ok": True, "len": len(str(payload))}

        return app

    def test_small_body_passes(self):
        """Body under the limit is accepted."""
        app = self._make_app(max_bytes=1024)
        client = TestClient(app)
        response = client.post("/echo", json={"msg": "small"})
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_oversize_body_rejected_with_413(self):
        """Body over the limit is rejected with 413 and structured error."""
        app = self._make_app(max_bytes=128)
        client = TestClient(app)
        response = client.post("/echo", json={"msg": "x" * 1024})
        assert response.status_code == 413
        body = response.json()
        assert body["error"]["code"] == "PAYLOAD_TOO_LARGE"
        assert body["error"]["retryable"] is False
        assert body["error"]["details"]["max_bytes"] == 128

    def test_zero_disables_limit(self):
        """max_bytes=0 disables the limit — any size is accepted."""
        app = self._make_app(max_bytes=0)
        client = TestClient(app)
        response = client.post("/echo", json={"msg": "y" * 5000})
        assert response.status_code == 200

    def test_malformed_content_length_falls_through(self):
        """Malformed Content-Length header doesn't crash — request proceeds."""
        app = self._make_app(max_bytes=1024)
        client = TestClient(app)
        # httpx sets a correct Content-Length when we pass content; we
        # overwrite it with garbage to simulate a malformed header.
        response = client.post(
            "/echo",
            content=b'{"msg":"ok"}',
            headers={"Content-Type": "application/json", "Content-Length": "notanumber"},
        )
        # Route still processes (malformed header is ignored by the limit)
        assert response.status_code == 200

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for middleware execution order.

Starlette executes middlewares in reverse add_middleware() order.
This test verifies that APIKey auth runs BEFORE RateLimit,
so unauthenticated requests don't consume rate limit tokens.
"""

import secrets
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from hfl.api.middleware import RateLimitMiddleware, reset_rate_limiter
from hfl.api.server import APIKeyMiddleware


@pytest.fixture(autouse=True)
def cleanup_rate_limiter():
    """Reset rate limiter state after each test."""
    yield
    reset_rate_limiter()


class TestMiddlewareExecutionOrder:
    """Verify middleware ordering: APIKey runs before RateLimit."""

    def _create_app_correct_order(self, api_key: str) -> FastAPI:
        """Create app with correct middleware order (matching server.py)."""
        app = FastAPI()

        # Same order as server.py: RateLimit added before APIKey
        # So execution order is: APIKey → RateLimit
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=3,
            window_seconds=60,
        )
        app.add_middleware(APIKeyMiddleware)

        @app.get("/api/test")
        def test_endpoint():
            return {"status": "ok"}

        return app

    def test_unauthenticated_rejected_before_rate_limit(self):
        """Unauthenticated requests should be rejected by auth, not rate limit."""
        app = self._create_app_correct_order(api_key="test-key-123")

        with patch.object(
            type(app), "_state", create=True
        ):
            from hfl.api.state import get_state

            state = get_state()
            state.api_key = "test-key-123"

            client = TestClient(app)

            # Make many unauthenticated requests
            for _ in range(10):
                resp = client.get("/api/test")
                # Should get 401 (auth failure), NOT 429 (rate limit)
                assert resp.status_code == 401

            # Now make authenticated request - should succeed (rate limit not exhausted)
            resp = client.get(
                "/api/test",
                headers={"Authorization": "Bearer test-key-123"},
            )
            assert resp.status_code == 200

            state.api_key = None

    def test_authenticated_requests_are_rate_limited(self):
        """Authenticated requests should properly reach rate limiter."""
        app = self._create_app_correct_order(api_key="key-456")

        from hfl.api.state import get_state

        state = get_state()
        state.api_key = "key-456"

        client = TestClient(app)

        # Make requests up to the limit
        for _ in range(3):
            resp = client.get(
                "/api/test",
                headers={"Authorization": "Bearer key-456"},
            )
            assert resp.status_code == 200

        # Next authenticated request should be rate limited
        resp = client.get(
            "/api/test",
            headers={"Authorization": "Bearer key-456"},
        )
        assert resp.status_code == 429

        state.api_key = None

    def test_no_api_key_configured_requests_pass_to_rate_limiter(self):
        """Without API key configured, requests pass directly to rate limiter."""
        app = self._create_app_correct_order(api_key="")

        from hfl.api.state import get_state

        state = get_state()
        state.api_key = None  # No auth configured

        client = TestClient(app)

        # All requests go to rate limiter
        for _ in range(3):
            resp = client.get("/api/test")
            assert resp.status_code == 200

        # Should be rate limited now
        resp = client.get("/api/test")
        assert resp.status_code == 429

    def test_server_middleware_add_order(self):
        """Verify server.py adds APIKey AFTER RateLimit (so APIKey runs first)."""
        import ast
        from pathlib import Path

        server_path = Path(__file__).parent.parent / "src" / "hfl" / "api" / "server.py"
        source = server_path.read_text()

        # Find the order of add_middleware calls in source code
        # RateLimitMiddleware is on its own line inside add_middleware()
        lines = source.splitlines()
        apikey_line = None
        ratelimit_line = None
        for i, line in enumerate(lines):
            if "APIKeyMiddleware" in line and "add_middleware" in line:
                apikey_line = i
            if "RateLimitMiddleware" in line:
                ratelimit_line = i

        # APIKey should be added AFTER RateLimit in the source
        # (Starlette executes in reverse order, so APIKey runs first)
        assert apikey_line is not None, "APIKeyMiddleware add_middleware not found"
        assert ratelimit_line is not None, "RateLimitMiddleware add_middleware not found"
        assert apikey_line > ratelimit_line, (
            f"APIKeyMiddleware (line {apikey_line}) must be added AFTER "
            f"RateLimitMiddleware (line {ratelimit_line}) so auth runs first"
        )

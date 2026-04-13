# SPDX-License-Identifier: HRUL-1.0
"""Tests for the operational §5 contract in the tool-calling spec.

Covers:

- Rate-limit response headers (``X-RateLimit-Reset``, ``X-RateLimit-Window``)
- Structured error envelope with ``category`` + ``retryable``
- ``/healthz`` endpoint shape
- Deterministic auth policy on ``/api/tags`` and ``/api/chat``
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.errors import ErrorDetail
from hfl.api.middleware import reset_rate_limiter
from hfl.api.server import app
from hfl.api.state import get_state


@pytest.fixture(autouse=True)
def _reset_state():
    get_state().api_key = None
    get_state().engine = None
    get_state().current_model = None
    reset_rate_limiter()
    yield
    get_state().api_key = None
    get_state().engine = None
    get_state().current_model = None
    reset_rate_limiter()


class TestRateLimitHeaders:
    def test_standard_response_has_full_rate_limit_headers(self):
        client = TestClient(app)
        resp = client.get("/api/version")
        # Spec §5.2: all four headers must be present.
        for h in (
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
            "x-ratelimit-window",
        ):
            assert h in {k.lower() for k in resp.headers.keys()}, f"missing {h}"
        # reset must be an epoch in the future
        import time

        reset = int(resp.headers["x-ratelimit-reset"])
        assert reset > int(time.time()) - 5


class TestErrorEnvelope:
    def test_error_detail_has_category_and_retryable(self):
        d = ErrorDetail(
            error="nope",
            code="UNAUTHORIZED",
            category="auth",
            retryable=False,
        )
        payload = d.model_dump()
        assert payload["category"] == "auth"
        assert payload["retryable"] is False

    def test_unauthorized_response_is_structured(self):
        """Spec §5.4 / §1: /api/chat with bad key returns structured
        envelope the client can retry-decide on."""
        get_state().api_key = "secret"
        client = TestClient(app)

        resp = client.post(
            "/api/chat",
            headers={"Authorization": "Bearer wrong"},
            json={
                "model": "qwen3-32b-q4_k_m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        err = body["error"]
        # Allow either the legacy string form OR the structured form:
        # we only assert presence of category/retryable in the structured one.
        assert isinstance(err, dict), f"error must be structured, got {err!r}"
        assert err["code"] == "UNAUTHORIZED"
        assert err["category"] == "auth"
        assert err["retryable"] is False


class TestHealthz:
    def test_healthz_ok_when_no_model(self):
        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "models_loaded" in body
        assert isinstance(body["models_loaded"], list)
        assert "queue_depth" in body
        assert "uptime_seconds" in body

    def test_healthz_is_public(self):
        """Spec §5.1/§5.5: healthz must not require auth."""
        get_state().api_key = "secret"
        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200


class TestDeterministicAuth:
    def test_tags_requires_auth_when_key_configured(self):
        """Spec §3: /api/tags auth must be deterministic."""
        get_state().api_key = "secret"
        client = TestClient(app)

        # Without header -> 401
        resp = client.get("/api/tags")
        assert resp.status_code == 401

        # With correct header -> 200 (or 500 if registry empty, but not 401)
        resp = client.get(
            "/api/tags", headers={"Authorization": "Bearer secret"}
        )
        assert resp.status_code != 401

    def test_tags_is_open_when_no_key(self):
        get_state().api_key = None
        client = TestClient(app)
        resp = client.get("/api/tags")
        assert resp.status_code == 200

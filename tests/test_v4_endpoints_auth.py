# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V5 α3 — confirm every V4 REST endpoint requires API key when configured.

The previous V4 audit explicitly called this out as missing. We
exhaustively probe every V4 path and ensure none of them accidentally
falls into ``APIKeyMiddleware.PUBLIC_ENDPOINTS`` / ``PUBLIC_PREFIXES``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state

# Every V4 (post-paridad) REST endpoint with a representative method.
# The body / query is irrelevant — we only care about the auth gate.
V4_ROUTES = [
    ("GET", "/api/discover"),
    ("GET", "/api/recommend"),
    ("POST", "/api/pull/smart"),
    ("POST", "/api/verify/some-model"),
    ("POST", "/api/benchmark/some-model"),
    ("POST", "/api/lora/apply"),
    ("POST", "/api/lora/remove"),
    ("GET", "/api/lora"),
    ("GET", "/api/lora/some-model"),
    ("POST", "/api/snapshot/save"),
    ("POST", "/api/snapshot/load"),
    ("GET", "/api/snapshot"),
    ("DELETE", "/api/snapshot/foo"),
    ("GET", "/api/compliance/dashboard"),
    ("GET", "/api/draft/recommend"),
    ("POST", "/v1/responses"),
    ("POST", "/api/push"),
]


@pytest.fixture
def client_with_api_key(temp_config):
    reset_state()
    state = get_state()
    state.api_key = "production-secret-token-1234567890"
    yield TestClient(app)
    state.api_key = None
    reset_state()


@pytest.mark.parametrize("method,path", V4_ROUTES)
def test_v4_endpoint_requires_auth_when_key_configured(client_with_api_key, method, path):
    """Every V4 route returns 401 when no API key is presented."""
    response = client_with_api_key.request(method, path, json={"model": "x"})
    assert response.status_code == 401, (
        f"{method} {path} did not require auth — currently returns {response.status_code}"
    )


@pytest.fixture
def client_no_auth(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def test_v4_endpoints_are_metric_tracked(client_no_auth):
    """V6 ν3 — every V4 GET endpoint flows through RequestLogger
    middleware and lands in ``Metrics.requests_by_endpoint``. This
    test pins the contract so a future routing change doesn't
    silently lose observability for the new surface.
    """
    from hfl.metrics import get_metrics, reset_metrics

    reset_metrics()

    # Hit the read-only endpoints that don't need auth setup or
    # complex bodies.
    safe_gets = [
        "/api/discover",
        "/api/recommend",
        "/api/compliance/dashboard",
        "/api/snapshot",
        "/api/lora",
        "/api/draft/recommend?model=meta-llama/Llama-3.1-8B-Instruct",
    ]
    for path in safe_gets:
        client_no_auth.get(path)

    recorded = get_metrics().requests_by_endpoint
    for path in safe_gets:
        canonical = path.split("?")[0]
        assert canonical in recorded, f"{canonical} not tracked in /metrics"


@pytest.mark.parametrize("method,path", V4_ROUTES)
def test_v4_endpoint_accepts_with_valid_bearer(client_with_api_key, method, path):
    """The same routes admit any well-formed request once auth is OK
    — beyond the auth gate they may still 4xx for missing payload, but
    they must NOT 401.

    Some endpoints (``/api/benchmark/{model}``) trigger a streaming
    response that surfaces engine errors mid-stream rather than as
    HTTP status — those raise from the test client. We treat that as
    "auth passed" since the auth gate is well before the stream
    starts.
    """
    body = {
        "model": "x",
        "lora_path": "/tmp/x",
        "adapter_id": "x",
        "name": "x",
        "destination": "u/x",
        "input": "hi",
        "stream": False,
    }
    try:
        response = client_with_api_key.request(
            method,
            path,
            json=body,
            headers={"Authorization": "Bearer production-secret-token-1234567890"},
        )
    except RuntimeError as exc:
        # FastAPI raises RuntimeError when a streaming handler hits a
        # downstream engine error after the response started — by that
        # point auth has already been validated.
        assert "Caught handled exception" in str(exc)
        return

    assert response.status_code != 401

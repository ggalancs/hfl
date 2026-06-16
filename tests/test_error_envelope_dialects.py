# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Per-dialect error-envelope shapes (API-3 / API-4).

Real OpenAI/Anthropic SDKs branch on a specific error shape; HFL's flat
``{"error","code",...}`` matched none, and the Anthropic route was internally
inconsistent (its service path used the flat shape). These tests pin the
OpenAI-nested and Anthropic-native envelopes on the ``/v1/*`` surfaces and
confirm the Ollama (``/api/*``) surface keeps its native shape.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


class TestValidationDialects:
    """API-4: invalid request bodies map to the provider's native 400 shape on
    ``/v1/*`` (not FastAPI's 422 ``{"detail": [...]}``)."""

    def test_openai_validation_is_400_nested(self, client):
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 5.0,  # > 2.0
            },
        )
        assert r.status_code == 400
        body = r.json()
        # OpenAI SDKs read err.error.message / .type — only "error" at top.
        assert set(body.keys()) == {"error"}
        err = body["error"]
        assert {"message", "type", "param", "code"} <= set(err.keys())
        assert err["type"] == "invalid_request_error"
        assert err["param"] is None
        # HFL diagnostics survive as SDK-ignored nested extensions.
        assert err["category"] == "validation"
        assert err["retryable"] is False

    def test_anthropic_validation_is_400_native(self, client):
        r = client.post(
            "/v1/messages",
            json={"model": "m", "max_tokens": 8, "messages": []},
        )
        assert r.status_code == 400
        body = r.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert "message" in body["error"]

    def test_ollama_native_validation_stays_422(self, client):
        r = client.post("/api/chat", json={"model": "m", "messages": []})
        # Native FastAPI shape preserved for the Ollama surface.
        assert r.status_code == 422
        assert "detail" in r.json()


class TestErrorPathDialects:
    """API-3: the error/service paths (model-load failure) are dialect-rendered
    too — the Anthropic route previously leaked the flat shape here."""

    def test_openai_error_path_is_nested(self, client):
        state = get_state()
        state.engine = None
        state.current_model = None
        r = client.post(
            "/v1/chat/completions",
            json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code in (404, 503)
        body = r.json()
        assert set(body.keys()) == {"error"}
        assert isinstance(body["error"], dict)
        assert "message" in body["error"] and "type" in body["error"]

    def test_anthropic_error_path_is_native(self, client):
        state = get_state()
        state.engine = None
        state.current_model = None
        r = client.post(
            "/v1/messages",
            json={
                "model": "nonexistent",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code in (404, 503)
        body = r.json()
        assert body["type"] == "error"
        assert isinstance(body["error"], dict)
        assert "message" in body["error"] and "type" in body["error"]
        assert "code" not in body  # no flat top-level leakage


class TestQueueRejectionFactoryDialect:
    """The dispatcher queue-rejection factories are path-aware so streaming
    /v1/* rejections (routed through acquire_stream_slot) nest too, not just
    the non-streaming path."""

    def test_queue_full_flat_by_default_nested_on_v1(self):
        import json

        from hfl.api.errors import queue_full

        flat = json.loads(bytes(queue_full(retry_after=1, depth=2, max_queued=3).body))
        assert flat["code"] == "QUEUE_FULL"  # flat (Ollama / default)

        resp = queue_full(retry_after=1, depth=2, max_queued=3, path="/v1/chat/completions")
        nested = json.loads(bytes(resp.body))
        assert set(nested.keys()) == {"error"}
        assert nested["error"]["type"] == "rate_limit_error"
        assert nested["error"]["code"] == "QUEUE_FULL"

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for exception_handlers.py."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hfl.api.exception_handlers import register_exception_handlers
from hfl.exceptions import (
    HFLError,
    ModelNotFoundError,
    ModelNotLoadedError,
    RateLimitError,
    ValidationError,
)


@pytest.fixture
def app_with_handlers():
    """Create a test app with exception handlers registered."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/test-hfl-error")
    async def raise_hfl_error():
        raise HFLError("Test error", details="some details")

    @app.get("/test-not-found")
    async def raise_not_found():
        raise ModelNotFoundError("my-model")

    @app.get("/test-not-loaded")
    async def raise_not_loaded():
        raise ModelNotLoadedError()

    @app.get("/test-validation")
    async def raise_validation():
        raise ValidationError("Invalid input")

    @app.get("/test-rate-limit")
    async def raise_rate_limit():
        raise RateLimitError(retry_after=30)

    @app.get("/test-runtime-error")
    async def raise_runtime():
        raise RuntimeError("engine crashed mid-generation")

    @app.get("/test-value-error")
    async def raise_value():
        raise ValueError("bad sampling config")

    return app


@pytest.fixture
def client(app_with_handlers):
    # raise_server_exceptions=False so TestClient round-trips uncaught
    # exceptions through the registered handlers instead of re-raising
    # them into the test body (mirrors production behaviour).
    return TestClient(app_with_handlers, raise_server_exceptions=False)


class TestExceptionHandlers:
    def test_hfl_error_returns_500(self, client):
        resp = client.get("/test-hfl-error")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "Test error"
        assert data["code"] == "HFLError"
        assert data["details"] == "some details"

    def test_model_not_found_returns_correct_status(self, client):
        resp = client.get("/test-not-found")
        # ModelNotFoundError now carries status_code=404 so the
        # global handler emits the proper HTTP status.
        assert resp.status_code == 404
        assert "Model not found" in resp.json()["error"]
        assert resp.json()["code"] == "ModelNotFoundError"

    def test_not_loaded_returns_500(self, client):
        resp = client.get("/test-not-loaded")
        assert resp.status_code == 500
        assert "not loaded" in resp.json()["error"].lower()

    def test_validation_error_returns_400(self, client):
        resp = client.get("/test-validation")
        assert resp.status_code == 400
        assert resp.json()["code"] == "ValidationError"

    def test_rate_limit_returns_429(self, client):
        resp = client.get("/test-rate-limit")
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["error"]


class TestUnhandledExceptionHandler:
    """Tests for the Exception fallback handler.

    Closes the ``"Internal Server Error"`` plain-text diagnostic gap
    that used to surface when a non-HFLError escaped a route (e.g. an
    engine ``RuntimeError`` after a thread-pool worker was left
    orphaned by a cancelled request).
    """

    def test_runtime_error_returns_structured_500(self, client):
        resp = client.get("/test-runtime-error")
        assert resp.status_code == 500
        body = resp.json()  # must be JSON, not plain text
        assert body["code"] == "UnhandledError"
        assert body["error_type"] == "RuntimeError"
        assert "engine crashed" in body["message"]

    def test_value_error_returns_structured_500(self, client):
        resp = client.get("/test-value-error")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_type"] == "ValueError"

    def test_message_is_capped_to_500_chars(self, client, app_with_handlers):
        """Don't let a pathological exception message blow up the
        response (or leak long internals)."""

        @app_with_handlers.get("/test-huge-msg")
        async def _huge():
            raise RuntimeError("x" * 10_000)

        client2 = TestClient(app_with_handlers, raise_server_exceptions=False)
        resp = client2.get("/test-huge-msg")
        assert resp.status_code == 500
        assert len(resp.json()["message"]) <= 500

    def test_unhandled_handler_logs_traceback(self, client):
        """Every 500 must produce a server-side traceback for diagnosis.

        HFL's ``configure_logging`` may have set ``propagate=False`` on
        the ``hfl`` root logger (by the time another test ran), which
        would bypass caplog. Attach our own handler directly to dodge
        ordering coupling.
        """
        import logging

        target = logging.getLogger("hfl.api.exception_handlers")
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.ERROR)
        target.addHandler(handler)
        # Ensure level isn't suppressing the message regardless of
        # what the ambient configuration did.
        original_level = target.level
        target.setLevel(logging.ERROR)
        try:
            client.get("/test-runtime-error")
        finally:
            target.removeHandler(handler)
            target.setLevel(original_level)

        assert any(
            "unhandled exception" in rec.getMessage() and "RuntimeError" in rec.getMessage()
            for rec in records
        ), f"expected traceback log, got: {[r.getMessage() for r in records]}"

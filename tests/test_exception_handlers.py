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

    return app


@pytest.fixture
def client(app_with_handlers):
    return TestClient(app_with_handlers)


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
        # ModelNotFoundError doesn't have status_code attr, falls back to 500
        assert resp.status_code == 500
        assert "Model not found" in resp.json()["error"]

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

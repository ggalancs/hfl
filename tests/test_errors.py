# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for API error responses."""

import json

import pytest

from hfl.api.errors import (
    ErrorDetail,
    HFLHTTPException,
    forbidden,
    internal_error,
    model_loading,
    model_not_found,
    model_type_mismatch,
    rate_limit_exceeded,
    service_unavailable,
    timeout_error,
    unauthorized,
    validation_error,
)


class TestErrorDetail:
    """Tests for ErrorDetail model."""

    def test_basic_creation(self):
        """Should create basic error detail."""
        detail = ErrorDetail(error="Something went wrong", code="ERROR")

        assert detail.error == "Something went wrong"
        assert detail.code == "ERROR"
        assert detail.details is None
        assert detail.request_id is None

    def test_with_details(self):
        """Should include additional details."""
        detail = ErrorDetail(
            error="Validation failed",
            code="VALIDATION_ERROR",
            details={"field": "temperature", "value": 5.0},
        )

        assert detail.details["field"] == "temperature"
        assert detail.details["value"] == 5.0

    def test_serialization(self):
        """Should serialize to dict."""
        detail = ErrorDetail(
            error="Test error",
            code="TEST",
            request_id="req-123",
        )

        data = detail.model_dump()

        assert data["error"] == "Test error"
        assert data["code"] == "TEST"
        assert data["request_id"] == "req-123"


class TestHFLHTTPException:
    """Tests for HFLHTTPException class."""

    def test_creates_structured_detail(self):
        """Should create structured error detail."""
        exc = HFLHTTPException(
            status_code=400,
            error="Invalid input",
            code="INVALID_INPUT",
        )

        assert exc.status_code == 400
        assert exc.detail["error"] == "Invalid input"
        assert exc.detail["code"] == "INVALID_INPUT"

    def test_includes_details(self):
        """Should include additional details."""
        exc = HFLHTTPException(
            status_code=400,
            error="Validation failed",
            code="VALIDATION_ERROR",
            details={"field": "model"},
        )

        assert exc.detail["details"]["field"] == "model"


class TestErrorFactories:
    """Tests for error factory functions."""

    def test_model_not_found(self):
        """model_not_found should create 404 error."""
        exc = model_not_found("llama-7b")

        assert exc.status_code == 404
        assert "llama-7b" in exc.detail["error"]
        assert exc.detail["code"] == "MODEL_NOT_FOUND"
        assert exc.detail["details"]["model"] == "llama-7b"

    def test_model_type_mismatch(self):
        """model_type_mismatch should create 400 error."""
        exc = model_type_mismatch("llm", "tts", "bark")

        assert exc.status_code == 400
        assert exc.detail["code"] == "MODEL_TYPE_MISMATCH"
        assert exc.detail["details"]["expected"] == "llm"
        assert exc.detail["details"]["got"] == "tts"
        assert exc.detail["details"]["model"] == "bark"

    def test_validation_error(self):
        """validation_error should create 400 error."""
        exc = validation_error("Invalid temperature", "temperature")

        assert exc.status_code == 400
        assert exc.detail["code"] == "VALIDATION_ERROR"
        assert exc.detail["details"]["field"] == "temperature"

    def test_rate_limit_exceeded(self):
        """rate_limit_exceeded should create 429 response."""
        response = rate_limit_exceeded(60, window_seconds=60)

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"

        body = json.loads(response.body)
        assert body["code"] == "RATE_LIMIT_EXCEEDED"
        assert body["details"]["retry_after_seconds"] == 60

    def test_unauthorized(self):
        """unauthorized should create 401 response."""
        response = unauthorized("Invalid token")

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

        body = json.loads(response.body)
        assert body["code"] == "UNAUTHORIZED"

    def test_forbidden(self):
        """forbidden should create 403 response."""
        response = forbidden("Access denied to model")

        assert response.status_code == 403

        body = json.loads(response.body)
        assert body["code"] == "FORBIDDEN"

    def test_internal_error(self):
        """internal_error should create 500 error."""
        exc = internal_error("Database connection failed", "DatabaseError")

        assert exc.status_code == 500
        assert exc.detail["code"] == "INTERNAL_ERROR"
        assert exc.detail["details"]["type"] == "DatabaseError"

    def test_service_unavailable(self):
        """service_unavailable should create 503 response."""
        response = service_unavailable("Server overloaded", retry_after=30)

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "30"

        body = json.loads(response.body)
        assert body["code"] == "SERVICE_UNAVAILABLE"

    def test_model_loading(self):
        """model_loading should create 503 response."""
        response = model_loading("llama-70b")

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"

        body = json.loads(response.body)
        assert body["code"] == "MODEL_LOADING"
        assert "llama-70b" in body["error"]

    def test_timeout_error(self):
        """timeout_error should create 504 error."""
        exc = timeout_error("model_load", 300.0)

        assert exc.status_code == 504
        assert exc.detail["code"] == "TIMEOUT"
        assert exc.detail["details"]["operation"] == "model_load"
        assert exc.detail["details"]["timeout_seconds"] == 300.0

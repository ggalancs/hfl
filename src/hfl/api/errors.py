# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Consistent error responses for HFL API.

Provides structured error response format and factory functions
for common error types.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hfl.logging_config import get_request_id


class ErrorDetail(BaseModel):
    """Structured error response model.

    Used for consistent error response format across all endpoints.
    """

    error: str
    code: str
    details: dict[str, Any] | None = None
    request_id: str | None = None

    model_config = {"extra": "forbid"}


class HFLHTTPException(HTTPException):
    """HTTP exception with structured error details.

    Extends FastAPI's HTTPException with structured error format.
    """

    def __init__(
        self,
        status_code: int,
        error: str,
        code: str,
        details: dict[str, Any] | None = None,
    ):
        """Create structured HTTP exception.

        Args:
            status_code: HTTP status code
            error: Human-readable error message
            code: Machine-readable error code
            details: Additional error details
        """
        detail = ErrorDetail(
            error=error,
            code=code,
            details=details,
            request_id=get_request_id(),
        )
        super().__init__(status_code=status_code, detail=detail.model_dump())
        self.error_code = code


# Common error factory functions


def model_not_found(model_name: str) -> HFLHTTPException:
    """Create 404 error for model not found.

    Args:
        model_name: Name of the model

    Returns:
        HFLHTTPException with 404 status
    """
    return HFLHTTPException(
        status_code=404,
        error=f"Model not found: {model_name}",
        code="MODEL_NOT_FOUND",
        details={"model": model_name},
    )


def model_type_mismatch(expected: str, got: str, model: str | None = None) -> HFLHTTPException:
    """Create 400 error for model type mismatch.

    Args:
        expected: Expected model type
        got: Actual model type
        model: Model name (optional)

    Returns:
        HFLHTTPException with 400 status
    """
    details = {"expected": expected, "got": got}
    if model:
        details["model"] = model

    return HFLHTTPException(
        status_code=400,
        error=f"Expected {expected} model, got {got}",
        code="MODEL_TYPE_MISMATCH",
        details=details,
    )


def validation_error(message: str, field: str | None = None) -> HFLHTTPException:
    """Create 400 error for validation failure.

    Args:
        message: Validation error message
        field: Field that failed validation (optional)

    Returns:
        HFLHTTPException with 400 status
    """
    return HFLHTTPException(
        status_code=400,
        error=message,
        code="VALIDATION_ERROR",
        details={"field": field} if field else None,
    )


def rate_limit_exceeded(retry_after: int, window_seconds: int | None = None) -> JSONResponse:
    """Create 429 response for rate limit exceeded.

    Args:
        retry_after: Seconds until rate limit resets
        window_seconds: Rate limit window (optional)

    Returns:
        JSONResponse with 429 status and Retry-After header
    """
    details = {"retry_after_seconds": retry_after}
    if window_seconds:
        details["window_seconds"] = window_seconds

    return JSONResponse(
        status_code=429,
        content=ErrorDetail(
            error="Rate limit exceeded",
            code="RATE_LIMIT_EXCEEDED",
            details=details,
            request_id=get_request_id(),
        ).model_dump(),
        headers={"Retry-After": str(retry_after)},
    )


def unauthorized(message: str = "Authentication required") -> JSONResponse:
    """Create 401 response for unauthorized access.

    Args:
        message: Error message

    Returns:
        JSONResponse with 401 status
    """
    return JSONResponse(
        status_code=401,
        content=ErrorDetail(
            error=message,
            code="UNAUTHORIZED",
            request_id=get_request_id(),
        ).model_dump(),
        headers={"WWW-Authenticate": "Bearer"},
    )


def forbidden(message: str = "Access denied") -> JSONResponse:
    """Create 403 response for forbidden access.

    Args:
        message: Error message

    Returns:
        JSONResponse with 403 status
    """
    return JSONResponse(
        status_code=403,
        content=ErrorDetail(
            error=message,
            code="FORBIDDEN",
            request_id=get_request_id(),
        ).model_dump(),
    )


def internal_error(
    message: str = "Internal server error",
    error_type: str | None = None,
) -> HFLHTTPException:
    """Create 500 error for internal server errors.

    Args:
        message: Error message
        error_type: Type of error (optional)

    Returns:
        HFLHTTPException with 500 status
    """
    return HFLHTTPException(
        status_code=500,
        error=message,
        code="INTERNAL_ERROR",
        details={"type": error_type} if error_type else None,
    )


def service_unavailable(
    message: str = "Service temporarily unavailable",
    retry_after: int | None = None,
) -> JSONResponse:
    """Create 503 response for service unavailable.

    Args:
        message: Error message
        retry_after: Seconds until service may be available

    Returns:
        JSONResponse with 503 status
    """
    headers = {}
    if retry_after:
        headers["Retry-After"] = str(retry_after)

    return JSONResponse(
        status_code=503,
        content=ErrorDetail(
            error=message,
            code="SERVICE_UNAVAILABLE",
            details={"retry_after": retry_after} if retry_after else None,
            request_id=get_request_id(),
        ).model_dump(),
        headers=headers if headers else None,
    )


def model_loading(model_name: str) -> JSONResponse:
    """Create 503 response for model currently loading.

    Args:
        model_name: Name of the loading model

    Returns:
        JSONResponse with 503 status
    """
    return JSONResponse(
        status_code=503,
        content=ErrorDetail(
            error=f"Model '{model_name}' is currently loading",
            code="MODEL_LOADING",
            details={"model": model_name},
            request_id=get_request_id(),
        ).model_dump(),
        headers={"Retry-After": "5"},
    )


def timeout_error(operation: str, timeout_seconds: float) -> HFLHTTPException:
    """Create 504 error for operation timeout.

    Args:
        operation: Name of the timed out operation
        timeout_seconds: Timeout duration

    Returns:
        HFLHTTPException with 504 status
    """
    return HFLHTTPException(
        status_code=504,
        error=f"Operation '{operation}' timed out",
        code="TIMEOUT",
        details={"operation": operation, "timeout_seconds": timeout_seconds},
    )

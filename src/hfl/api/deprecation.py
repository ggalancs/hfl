# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Deprecation utilities for API endpoints.

Provides helpers for marking endpoints as deprecated with standard
HTTP headers (Deprecation, Sunset) per RFC 8594.
"""

from __future__ import annotations

from datetime import datetime
from functools import wraps
from typing import Any, Callable

from fastapi import Response


def add_deprecation_headers(
    response: Response,
    deprecated_at: str | None = None,
    sunset: str | None = None,
    alternative: str | None = None,
) -> None:
    """Add deprecation headers to a response.

    Implements RFC 8594 deprecation header format.

    Args:
        response: FastAPI/Starlette Response object
        deprecated_at: ISO 8601 date when deprecation started
        sunset: ISO 8601 date when endpoint will be removed
        alternative: URL or path to the replacement endpoint

    Example:
        >>> response = Response()
        >>> add_deprecation_headers(
        ...     response,
        ...     deprecated_at="2026-01-01",
        ...     sunset="2027-01-01",
        ...     alternative="/v2/chat/completions"
        ... )
    """
    response.headers["Deprecation"] = "true"

    if deprecated_at:
        # RFC 8594 format: "@" followed by Unix timestamp or HTTP date
        try:
            dt = datetime.fromisoformat(deprecated_at)
            response.headers["Deprecation"] = f"@{int(dt.timestamp())}"
        except ValueError:
            response.headers["Deprecation"] = "true"

    if sunset:
        # Sunset header uses HTTP date format
        try:
            dt = datetime.fromisoformat(sunset)
            response.headers["Sunset"] = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
        except ValueError:
            pass

    if alternative:
        # Link header for the replacement
        response.headers["Link"] = f'<{alternative}>; rel="successor-version"'


def deprecated_endpoint(
    deprecated_at: str | None = None,
    sunset: str | None = None,
    alternative: str | None = None,
    message: str | None = None,
) -> Callable:
    """Decorator to mark an endpoint as deprecated.

    Adds deprecation headers to the response and includes
    deprecation warning in the response body if applicable.

    Args:
        deprecated_at: ISO 8601 date when deprecation started
        sunset: ISO 8601 date when endpoint will be removed
        alternative: URL or path to the replacement endpoint
        message: Custom deprecation message for response body

    Example:
        >>> @deprecated_endpoint(
        ...     sunset="2027-01-01",
        ...     alternative="/v2/generate",
        ...     message="Use /v2/generate instead"
        ... )
        ... async def old_generate():
        ...     return {"text": "..."}
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)

            # Find response object if available
            response = kwargs.get("response")
            if response is None and hasattr(result, "headers"):
                response = result

            if response is not None:
                add_deprecation_headers(
                    response,
                    deprecated_at=deprecated_at,
                    sunset=sunset,
                    alternative=alternative,
                )

            # If result is a dict, we can add deprecation warning
            if isinstance(result, dict) and message:
                result["_deprecation_warning"] = message

            return result

        return wrapper

    return decorator


def format_sunset_date(date: datetime) -> str:
    """Format a datetime as an RFC 7231 HTTP date.

    Args:
        date: datetime object

    Returns:
        HTTP date string (e.g., "Mon, 01 Jan 2027 00:00:00 GMT")
    """
    return date.strftime("%a, %d %b %Y %H:%M:%S GMT")

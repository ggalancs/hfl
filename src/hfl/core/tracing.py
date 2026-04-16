# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Request tracing using context variables.

Provides request ID tracking that propagates through the entire request
lifecycle, including async operations and thread pool executors.

Usage:
    # In middleware
    from hfl.core.tracing import set_request_id, get_request_id

    rid = set_request_id()  # Generate new ID
    # or
    rid = set_request_id("custom-id")  # Use provided ID

    # Anywhere in the request context
    logger.info("[%s] Processing request...", get_request_id())

Features:
    - Thread-safe via contextvars
    - Propagates through asyncio tasks
    - Short 8-char IDs for readability
    - Optional custom ID for correlation with external systems
"""

from __future__ import annotations

import inspect
import uuid
from contextvars import ContextVar, Token
from typing import Callable

# Context variable for request ID
# Default to None when not in a request context
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Context variable for additional trace context (parent span, etc.)
_trace_context: ContextVar[dict | None] = ContextVar("trace_context", default=None)


def generate_request_id() -> str:
    """Generate a short unique request ID.

    Returns:
        8-character hex string (e.g., "a1b2c3d4")
    """
    return uuid.uuid4().hex[:8]


def set_request_id(rid: str | None = None) -> str:
    """Set the request ID for the current context.

    If no ID is provided, generates a new one.

    Args:
        rid: Optional custom request ID to use

    Returns:
        The request ID that was set
    """
    if rid is None:
        rid = generate_request_id()
    _request_id.set(rid)
    return rid


def get_request_id() -> str | None:
    """Get the request ID for the current context.

    Returns:
        The current request ID, or None if not in a request context
    """
    return _request_id.get()


def clear_request_id() -> None:
    """Clear the request ID for the current context.

    This resets the context variable to its default (None).
    Primarily useful for testing and cleanup.
    """
    _request_id.set(None)


def set_trace_context(context: dict) -> Token[dict | None]:
    """Set additional trace context.

    Args:
        context: Dictionary with trace information (e.g., parent_span_id)

    Returns:
        Token for restoring previous context
    """
    return _trace_context.set(context)


def get_trace_context() -> dict | None:
    """Get additional trace context.

    Returns:
        The trace context dictionary, or None
    """
    return _trace_context.get()


def with_request_id(rid: str | None = None) -> Callable:
    """Decorator to set request ID for a function.

    Args:
        rid: Optional request ID (generates one if not provided)

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            old_id = get_request_id()
            set_request_id(rid)
            try:
                return func(*args, **kwargs)
            finally:
                if old_id is not None:
                    set_request_id(old_id)
                else:
                    clear_request_id()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            old_id = get_request_id()
            set_request_id(rid)
            try:
                return await func(*args, **kwargs)
            finally:
                if old_id is not None:
                    set_request_id(old_id)
                else:
                    clear_request_id()

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class RequestContext:
    """Context manager for request tracing.

    Usage:
        with RequestContext() as ctx:
            # request_id is set
            do_something()
        # request_id is cleared

        # With custom ID
        with RequestContext("my-request-id"):
            do_something()

        # Async usage
        async with RequestContext():
            await do_something_async()
    """

    def __init__(self, request_id: str | None = None):
        """Initialize request context.

        Args:
            request_id: Optional custom request ID
        """
        self._request_id = request_id
        self._previous_id: str | None = None

    def __enter__(self) -> "RequestContext":
        """Enter the context, setting the request ID."""
        self._previous_id = get_request_id()
        self._request_id = set_request_id(self._request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context, restoring previous request ID."""
        if self._previous_id is not None:
            set_request_id(self._previous_id)
        else:
            clear_request_id()

    async def __aenter__(self) -> "RequestContext":
        """Async enter."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async exit."""
        self.__exit__(exc_type, exc_val, exc_tb)

    @property
    def request_id(self) -> str | None:
        """Get the request ID for this context."""
        return self._request_id


def format_log_prefix() -> str:
    """Format a log prefix with request ID if available.

    Returns:
        String like "[abc123de] " or empty string if no request ID
    """
    rid = get_request_id()
    if rid:
        return f"[{rid}] "
    return ""

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Universal timeout decorator and helper for HFL API endpoints.

Provides ``with_timeout`` (decorator) and ``run_with_timeout`` (standalone
coroutine wrapper) that enforce a maximum execution time on async operations.

When the deadline is exceeded the helpers raise
:class:`~hfl.api.errors.HFLHTTPException` with HTTP 504 via
:func:`~hfl.api.errors.timeout_error`.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar("T")


def with_timeout(timeout_seconds: float | None = None) -> Callable:
    """Decorator that enforces a timeout on an async endpoint.

    Args:
        timeout_seconds: Maximum seconds to wait.  When *None* the value
            is read from ``hfl.config.config.generation_timeout`` at
            call time (lazy import keeps module import cheap).

    Returns:
        A decorator that wraps the target coroutine function with
        :func:`asyncio.wait_for`.

    Raises:
        HFLHTTPException: 504 Gateway Timeout when the deadline is exceeded.

    Example::

        @router.post("/v1/completions")
        @with_timeout(30.0)
        async def completions(request: Request):
            ...
    """

    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from hfl.config import config

            effective_timeout = timeout_seconds if timeout_seconds is not None else config.generation_timeout
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                from hfl.api.errors import timeout_error

                raise timeout_error(
                    operation=func.__qualname__,
                    timeout_seconds=effective_timeout,
                )

        return wrapper

    return decorator


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float | None = None,
    *,
    operation_name: str = "operation",
) -> T:
    """Run *coro* with an enforced timeout.

    This is a non-decorator alternative for one-off awaitable calls.

    Args:
        coro: The coroutine to execute.
        timeout_seconds: Maximum seconds to wait.  Falls back to
            ``config.generation_timeout`` when *None*.
        operation_name: Human-readable label used in the error message
            if the timeout fires.

    Returns:
        The return value of *coro*.

    Raises:
        HFLHTTPException: 504 Gateway Timeout when the deadline is exceeded.
    """
    from hfl.config import config

    effective_timeout = timeout_seconds if timeout_seconds is not None else config.generation_timeout
    try:
        return await asyncio.wait_for(coro, timeout=effective_timeout)
    except asyncio.TimeoutError:
        from hfl.api.errors import timeout_error

        raise timeout_error(
            operation=operation_name,
            timeout_seconds=effective_timeout,
        )

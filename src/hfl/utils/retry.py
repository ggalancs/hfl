# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Retry utilities with exponential backoff.

Provides decorators and context managers for robust error handling
with configurable retry logic.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import random
import time
from typing import Awaitable, Callable, ParamSpec, Type, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


class RetryExhausted(Exception):
    """All retry attempts failed."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.1,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for retry with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Jitter factor (0.1 = ±10% randomization)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt) called on each retry

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(max_retries=3, exceptions=(ConnectionError,))
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        # Add jitter to prevent thundering herd
                        actual_delay = delay * (1 + random.uniform(-jitter, jitter))
                        time.sleep(actual_delay)
                        delay = min(delay * 2, max_delay)

            raise RetryExhausted(
                f"Failed after {max_retries + 1} attempts: {last_exception}",
                last_exception=last_exception,
            ) from last_exception

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            delay = base_delay
            # This wrapper is only installed when ``func`` is an
            # async function (see the dispatch below), but the outer
            # decorator is generic over ``Callable[P, T]`` and can't
            # express that. Cast once so mypy can type-check the
            # await correctly instead of seeing a ``T``-typed value.
            async_func = cast(Callable[P, Awaitable[T]], func)

            for attempt in range(max_retries + 1):
                try:
                    return await async_func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        actual_delay = delay * (1 + random.uniform(-jitter, jitter))
                        await asyncio.sleep(actual_delay)
                        delay = min(delay * 2, max_delay)

            raise RetryExhausted(
                f"Failed after {max_retries + 1} attempts: {last_exception}",
                last_exception=last_exception,
            ) from last_exception

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator

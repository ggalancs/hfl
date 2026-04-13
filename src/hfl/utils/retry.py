# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Retry utilities with exponential backoff.

Provides decorators and context managers for robust error handling
with configurable retry logic.
"""

from __future__ import annotations

import asyncio
import functools
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
                f"Failed after {max_retries + 1} attempts",
                last_exception=last_exception,
            )

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
                f"Failed after {max_retries + 1} attempts",
                last_exception=last_exception,
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator


class RetryContext:
    """Context manager for retry logic.

    Example:
        async with RetryContext(max_retries=3) as ctx:
            for attempt in ctx:
                try:
                    result = await risky_operation()
                    break
                except ConnectionError:
                    await ctx.handle_error()
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.1,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._attempt = 0
        self._delay = base_delay
        self._last_exception: Exception | None = None

    def __iter__(self):
        """Iterate over retry attempts."""
        for self._attempt in range(self.max_retries + 1):
            yield self._attempt

    async def handle_error(self, exception: Exception | None = None) -> None:
        """Handle error and wait before next retry (async)."""
        self._last_exception = exception
        if self._attempt < self.max_retries:
            actual_delay = self._delay * (1 + random.uniform(-self.jitter, self.jitter))
            await asyncio.sleep(actual_delay)
            self._delay = min(self._delay * 2, self.max_delay)
        else:
            raise RetryExhausted(
                f"Failed after {self.max_retries + 1} attempts",
                last_exception=self._last_exception,
            )

    def handle_error_sync(self, exception: Exception | None = None) -> None:
        """Handle error and wait before next retry (sync)."""
        self._last_exception = exception
        if self._attempt < self.max_retries:
            actual_delay = self._delay * (1 + random.uniform(-self.jitter, self.jitter))
            time.sleep(actual_delay)
            self._delay = min(self._delay * 2, self.max_delay)
        else:
            raise RetryExhausted(
                f"Failed after {self.max_retries + 1} attempts",
                last_exception=self._last_exception,
            )

    @property
    def attempt(self) -> int:
        """Current attempt number (0-indexed)."""
        return self._attempt

    @property
    def attempts_remaining(self) -> int:
        """Number of attempts remaining."""
        return self.max_retries - self._attempt

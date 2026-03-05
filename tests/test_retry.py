# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for retry utilities."""

import time

import pytest

from hfl.utils.retry import RetryContext, RetryExhausted, with_retry


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_successful_call_no_retry(self):
        """Successful call should not retry."""
        call_count = 0

        @with_retry(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_exception(self):
        """Should retry on specified exceptions."""
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01, exceptions=(ValueError,))
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 3

    def test_exhaust_retries(self):
        """Should raise RetryExhausted after all retries fail."""

        @with_retry(max_retries=2, base_delay=0.01, exceptions=(ValueError,))
        def always_failing():
            raise ValueError("permanent error")

        with pytest.raises(RetryExhausted) as exc_info:
            always_failing()

        assert "Failed after 3 attempts" in str(exc_info.value)
        assert exc_info.value.last_exception is not None
        assert isinstance(exc_info.value.last_exception, ValueError)

    def test_no_retry_on_different_exception(self):
        """Should not retry on exceptions not in the list."""

        @with_retry(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def raise_type_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            raise_type_error()

    def test_on_retry_callback(self):
        """Should call on_retry callback on each retry."""
        retry_attempts = []

        def on_retry(exc, attempt):
            retry_attempts.append((str(exc), attempt))

        @with_retry(
            max_retries=2,
            base_delay=0.01,
            exceptions=(ValueError,),
            on_retry=on_retry,
        )
        def failing_twice():
            if len(retry_attempts) < 2:
                raise ValueError("fail")
            return "success"

        result = failing_twice()
        assert result == "success"
        assert len(retry_attempts) == 2
        assert retry_attempts[0][1] == 1
        assert retry_attempts[1][1] == 2

    def test_exponential_backoff(self):
        """Should use exponential backoff."""
        times = []

        @with_retry(max_retries=3, base_delay=0.1, max_delay=1.0, jitter=0)
        def track_time():
            times.append(time.time())
            if len(times) < 4:
                raise ValueError("retry")
            return "done"

        track_time()

        # Check delays increase exponentially (with tolerance for system jitter)
        delays = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        # Expected: 0.1, 0.2, 0.4 (exponential with factor 2)
        # Use tolerance check instead of strict ordering due to system timing variance
        assert delays[0] >= 0.08  # ~0.1s with tolerance
        assert delays[1] >= 0.15  # ~0.2s with tolerance
        assert delays[2] >= 0.30  # ~0.4s with tolerance
        # Total should be roughly 0.7s (0.1 + 0.2 + 0.4)
        assert sum(delays) >= 0.5


class TestWithRetryAsync:
    """Tests for with_retry decorator with async functions."""

    @pytest.mark.asyncio
    async def test_async_successful_call(self):
        """Async successful call should not retry."""
        call_count = 0

        @with_retry(max_retries=3)
        async def async_func():
            nonlocal call_count
            call_count += 1
            return "async success"

        result = await async_func()
        assert result == "async success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_exception(self):
        """Async should retry on exceptions."""
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01, exceptions=(ConnectionError,))
        async def flaky_async():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("network error")
            return "connected"

        result = await flaky_async()
        assert result == "connected"
        assert call_count == 2


class TestRetryContext:
    """Tests for RetryContext class."""

    def test_iteration_count(self):
        """Should iterate correct number of times."""
        ctx = RetryContext(max_retries=3)
        attempts = list(ctx)
        assert len(attempts) == 4  # 0, 1, 2, 3

    def test_attempts_remaining(self):
        """Should track remaining attempts."""
        ctx = RetryContext(max_retries=2)

        for attempt in ctx:
            if attempt == 0:
                assert ctx.attempts_remaining == 2
            elif attempt == 1:
                assert ctx.attempts_remaining == 1
            elif attempt == 2:
                assert ctx.attempts_remaining == 0

    def test_handle_error_sync(self):
        """Sync error handling should wait and allow retry."""
        ctx = RetryContext(max_retries=1, base_delay=0.01)

        for attempt in ctx:
            if attempt == 0:
                ctx.handle_error_sync(ValueError("test"))
            else:
                break

        assert ctx.attempt == 1

    def test_exhaust_sync_retries(self):
        """Should raise after exhausting retries."""
        ctx = RetryContext(max_retries=1, base_delay=0.01)

        with pytest.raises(RetryExhausted):
            for attempt in ctx:
                ctx.handle_error_sync(ValueError("always fail"))

    @pytest.mark.asyncio
    async def test_handle_error_async(self):
        """Async error handling should wait and allow retry."""
        ctx = RetryContext(max_retries=1, base_delay=0.01)

        for attempt in ctx:
            if attempt == 0:
                await ctx.handle_error(ConnectionError("network"))
            else:
                break

        assert ctx.attempt == 1

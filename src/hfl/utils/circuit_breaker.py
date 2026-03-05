# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Circuit breaker pattern for fault tolerance.

Prevents cascading failures by temporarily disabling calls to failing services.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpen(Exception):
    """Circuit breaker is open, request rejected."""

    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class CircuitBreaker(Generic[T]):
    """Circuit breaker for fault tolerance.

    Tracks failures and opens the circuit when threshold is reached.
    After recovery_timeout, allows test requests (half-open state).
    If test requests succeed, closes the circuit.

    Example:
        breaker = CircuitBreaker(failure_threshold=5)

        try:
            result = breaker.call(risky_function, arg1, arg2)
        except CircuitBreakerOpen:
            # Use fallback or return cached result
            pass
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 1

    _failures: int = field(default=0, init=False, repr=False)
    _successes_in_half_open: int = field(default=0, init=False, repr=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)
    _half_open_calls: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitBreakerOpen: If circuit is open
            Any exception from func
        """
        with self._lock:
            self._maybe_transition()

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpen(
                    f"Circuit open. Retry after {self._time_until_retry():.1f}s",
                    retry_after=self._time_until_retry(),
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen(
                        "Circuit half-open, max test calls reached",
                        retry_after=self._time_until_retry(),
                    )
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _maybe_transition(self) -> None:
        """Check if state should transition (must hold lock)."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._successes_in_half_open = 0

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._successes_in_half_open += 1
                if self._successes_in_half_open >= self.half_open_max_calls:
                    # All test calls succeeded, close circuit
                    self._state = CircuitState.CLOSED
                    self._failures = 0
            else:
                # Reset failures on success in closed state
                self._failures = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Test call failed, reopen circuit
                self._state = CircuitState.OPEN
            elif self._failures >= self.failure_threshold:
                # Too many failures, open circuit
                self._state = CircuitState.OPEN

    def _time_until_retry(self) -> float:
        """Time until circuit might transition to half-open."""
        elapsed = time.time() - self._last_failure_time
        return max(0, self.recovery_timeout - elapsed)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            self._maybe_transition()
            return self._state

    @property
    def failures(self) -> int:
        """Current failure count."""
        with self._lock:
            return self._failures

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._last_failure_time = 0.0
            self._half_open_calls = 0
            self._successes_in_half_open = 0


# Pre-configured circuit breakers for common services
class CircuitBreakers:
    """Registry of circuit breakers for different services."""

    _breakers: dict[str, CircuitBreaker] = {}
    _lock = threading.Lock()

    @classmethod
    def get(
        cls,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        with cls._lock:
            if name not in cls._breakers:
                cls._breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
            return cls._breakers[name]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers."""
        with cls._lock:
            for breaker in cls._breakers.values():
                breaker.reset()
            cls._breakers.clear()

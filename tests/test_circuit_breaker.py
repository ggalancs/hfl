# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for circuit breaker pattern."""

import time

import pytest

from hfl.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakers,
    CircuitState,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Circuit should start in closed state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failures == 0

    def test_successful_calls(self):
        """Successful calls should not affect circuit."""
        breaker = CircuitBreaker(failure_threshold=3)

        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failures == 0

    def test_failures_counted(self):
        """Failures should be counted."""
        breaker = CircuitBreaker(failure_threshold=3)

        def failing():
            raise ValueError("error")

        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing)

        assert breaker.failures == 2
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_opens_on_threshold(self):
        """Circuit should open when failure threshold is reached."""
        breaker = CircuitBreaker(failure_threshold=3)

        def failing():
            raise ValueError("error")

        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failures == 3

    def test_open_circuit_rejects_calls(self):
        """Open circuit should reject calls immediately."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60)

        # Trip the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Subsequent calls should be rejected
        with pytest.raises(CircuitBreakerOpen) as exc_info:
            breaker.call(lambda: "should not run")

        assert "Circuit open" in str(exc_info.value)
        assert exc_info.value.retry_after > 0

    def test_circuit_transitions_to_half_open(self):
        """Circuit should transition to half-open after recovery timeout."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,  # Very short for testing
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # State should transition on next check
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Successful call in half-open state should close circuit."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            half_open_max_calls=1,
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        time.sleep(0.1)

        # Successful test call should close circuit
        result = breaker.call(lambda: "recovered")
        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failures == 0

    def test_half_open_failure_reopens_circuit(self):
        """Failed call in half-open state should reopen circuit."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
        )

        # Trip the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        time.sleep(0.1)

        # Failed test call should reopen circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("still failing")))

        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failures(self):
        """Successful call should reset failure count."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Some failures
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.failures == 2

        # Success resets count
        breaker.call(lambda: "ok")
        assert breaker.failures == 0

    def test_reset(self):
        """Manual reset should restore closed state."""
        breaker = CircuitBreaker(failure_threshold=1)

        # Trip the circuit
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failures == 0


class TestCircuitBreakers:
    """Tests for CircuitBreakers registry."""

    def test_get_creates_breaker(self):
        """get() should create new breaker if not exists."""
        CircuitBreakers.reset_all()

        breaker = CircuitBreakers.get("test-service")
        assert breaker is not None
        assert breaker.state == CircuitState.CLOSED

    def test_get_returns_same_breaker(self):
        """get() should return same breaker for same name."""
        CircuitBreakers.reset_all()

        breaker1 = CircuitBreakers.get("my-service")
        breaker2 = CircuitBreakers.get("my-service")
        assert breaker1 is breaker2

    def test_get_different_services(self):
        """Different service names should get different breakers."""
        CircuitBreakers.reset_all()

        breaker1 = CircuitBreakers.get("service-a")
        breaker2 = CircuitBreakers.get("service-b")
        assert breaker1 is not breaker2

    def test_reset_all(self):
        """reset_all() should clear all breakers."""
        CircuitBreakers.reset_all()

        breaker = CircuitBreakers.get("test")
        # Trip it
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        CircuitBreakers.reset_all()

        # New breaker should be fresh
        new_breaker = CircuitBreakers.get("test")
        assert new_breaker.state == CircuitState.CLOSED
        assert new_breaker.failures == 0

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for PerModelRateLimiter."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from hfl.api.rate_limit import PerModelRateLimiter, create_rate_limiter


class TestPerModelRateLimiterGlobal:
    """Global rate limiting works correctly."""

    def test_global_limit_allows_requests_within_limit(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=5, per_model_rpm=10, window_seconds=60)
        for i in range(5):
            allowed, remaining = limiter.is_allowed("client1", "modelA")
            assert allowed, f"Request {i} should be allowed"

    def test_global_limit_blocks_after_exhaustion(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=3, per_model_rpm=10, window_seconds=60)
        for _ in range(3):
            limiter.is_allowed("client1", "modelA")
        allowed, remaining = limiter.is_allowed("client1", "modelA")
        assert not allowed
        assert remaining == 0

    def test_global_limit_shared_across_models(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=4, per_model_rpm=10, window_seconds=60)
        # Use 2 on modelA, 2 on modelB -> global exhausted
        limiter.is_allowed("client1", "modelA")
        limiter.is_allowed("client1", "modelA")
        limiter.is_allowed("client1", "modelB")
        limiter.is_allowed("client1", "modelB")
        allowed, _ = limiter.is_allowed("client1", "modelA")
        assert not allowed


class TestPerModelRateLimiterPerModel:
    """Per-model rate limiting works independently."""

    def test_per_model_limit_blocks_single_model(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=100, per_model_rpm=2, window_seconds=60)
        limiter.is_allowed("client1", "modelA")
        limiter.is_allowed("client1", "modelA")
        allowed, _ = limiter.is_allowed("client1", "modelA")
        assert not allowed

    def test_different_models_have_independent_limits(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=100, per_model_rpm=2, window_seconds=60)
        # Exhaust modelA
        limiter.is_allowed("client1", "modelA")
        limiter.is_allowed("client1", "modelA")
        allowed_a, _ = limiter.is_allowed("client1", "modelA")
        assert not allowed_a
        # modelB should still work
        allowed_b, _ = limiter.is_allowed("client1", "modelB")
        assert allowed_b

    def test_remaining_returns_min_of_global_and_model(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=10, per_model_rpm=3, window_seconds=60)
        allowed, remaining = limiter.is_allowed("client1", "modelA")
        assert allowed
        # remaining should be min(global_remaining=9, model_remaining=2) = 2
        assert remaining == 2


class TestPerModelConcurrency:
    """Concurrent slot acquisition and release."""

    def test_acquire_concurrent_succeeds(self) -> None:
        limiter = PerModelRateLimiter(concurrent_per_model=2)
        assert limiter.acquire_concurrent("modelA") is True
        assert limiter.acquire_concurrent("modelA") is True

    def test_concurrent_slots_can_be_exhausted(self) -> None:
        limiter = PerModelRateLimiter(concurrent_per_model=2)
        assert limiter.acquire_concurrent("modelA") is True
        assert limiter.acquire_concurrent("modelA") is True
        assert limiter.acquire_concurrent("modelA") is False

    def test_release_frees_slot(self) -> None:
        limiter = PerModelRateLimiter(concurrent_per_model=1)
        assert limiter.acquire_concurrent("modelA") is True
        assert limiter.acquire_concurrent("modelA") is False
        limiter.release_concurrent("modelA")
        assert limiter.acquire_concurrent("modelA") is True

    def test_different_models_have_independent_concurrency(self) -> None:
        limiter = PerModelRateLimiter(concurrent_per_model=1)
        assert limiter.acquire_concurrent("modelA") is True
        assert limiter.acquire_concurrent("modelA") is False
        # modelB is independent
        assert limiter.acquire_concurrent("modelB") is True


class TestPerModelReset:
    """Reset clears all state."""

    def test_reset_all_clears_state(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=2, per_model_rpm=2, window_seconds=60)
        limiter.is_allowed("client1", "modelA")
        limiter.is_allowed("client1", "modelA")
        allowed, _ = limiter.is_allowed("client1", "modelA")
        assert not allowed
        limiter.reset()
        allowed, _ = limiter.is_allowed("client1", "modelA")
        assert allowed

    def test_reset_specific_client(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=2, per_model_rpm=2, window_seconds=60)
        limiter.is_allowed("client1", "modelA")
        limiter.is_allowed("client1", "modelA")
        limiter.reset("client1")
        allowed, _ = limiter.is_allowed("client1", "modelA")
        assert allowed

    def test_reset_does_not_affect_other_clients(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=100, per_model_rpm=2, window_seconds=60)
        limiter.is_allowed("client1", "modelA")
        limiter.is_allowed("client2", "modelA")
        limiter.is_allowed("client2", "modelA")
        limiter.reset("client1")
        # client2 should still be limited
        allowed, _ = limiter.is_allowed("client2", "modelA")
        assert not allowed


class TestPerModelGlobalLimiterProperty:
    """The global_limiter property works."""

    def test_global_limiter_property(self) -> None:
        limiter = PerModelRateLimiter(global_rpm=42)
        assert limiter.global_limiter.requests_per_window == 42


class TestPerModelThreadSafety:
    """Thread-safety under concurrent access."""

    def test_concurrent_is_allowed_calls(self) -> None:
        limiter = PerModelRateLimiter(
            global_rpm=100, per_model_rpm=50, window_seconds=60, concurrent_per_model=10
        )
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            allowed, _ = limiter.is_allowed("client1", "modelA")
            with lock:
                results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(results)  # all should be allowed (limit is 50+100)

    def test_concurrent_acquire_release(self) -> None:
        limiter = PerModelRateLimiter(concurrent_per_model=5)
        acquired = []
        lock = threading.Lock()

        def worker() -> None:
            got = limiter.acquire_concurrent("modelA")
            with lock:
                acquired.append(got)
            if got:
                time.sleep(0.01)
                limiter.release_concurrent("modelA")

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(worker) for _ in range(10)]
            for f in futures:
                f.result()

        # At most 5 could acquire at a time, but due to quick release
        # most should succeed. At minimum, 5 must have succeeded.
        assert sum(acquired) >= 5

    def test_concurrent_model_creation(self) -> None:
        """Multiple threads creating limiters for the same model simultaneously."""
        limiter = PerModelRateLimiter(global_rpm=1000, per_model_rpm=100)
        errors: list[Exception] = []

        def worker(model: str) -> None:
            try:
                for _ in range(10):
                    limiter.is_allowed("c1", model)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=("shared-model",)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"


class TestCreateRateLimiterPerModel:
    """Factory function supports per_model parameter."""

    def test_create_per_model_rate_limiter(self) -> None:
        limiter = create_rate_limiter(per_model=True, per_model_rpm=10, concurrent_per_model=2)
        assert isinstance(limiter, PerModelRateLimiter)

    def test_create_per_model_uses_global_rpm(self) -> None:
        limiter = create_rate_limiter(per_model=True, requests_per_window=42, per_model_rpm=10)
        assert isinstance(limiter, PerModelRateLimiter)
        assert limiter.global_limiter.requests_per_window == 42

    def test_create_without_per_model_returns_regular(self) -> None:
        from hfl.api.rate_limit import InMemoryRateLimiter

        limiter = create_rate_limiter(per_model=False)
        assert isinstance(limiter, InMemoryRateLimiter)

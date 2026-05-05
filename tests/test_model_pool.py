# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for model pooling and caching."""

from dataclasses import dataclass

import pytest

from hfl.engine.model_pool import CachedModel, ModelPool, get_model_pool, reset_model_pool


@dataclass
class MockManifest:
    """Mock model manifest for testing."""

    name: str


class MockEngine:
    """Mock inference engine for testing."""

    def __init__(self, name: str = "test"):
        self.name = name
        self._loaded = True
        self.unload_called = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False
        self.unload_called = True


class TestCachedModel:
    """Tests for CachedModel class."""

    def test_touch_updates_timestamp(self):
        """touch() should update last_used."""
        engine = MockEngine()
        manifest = MockManifest("test")
        cached = CachedModel(
            engine=engine,
            manifest=manifest,
            last_used=100.0,
            load_time_ms=50.0,
        )

        old_time = cached.last_used
        cached.touch()

        assert cached.last_used > old_time


class TestModelPool:
    """Tests for ModelPool class."""

    @pytest.fixture
    def pool(self):
        """Create a fresh model pool."""
        return ModelPool(max_models=3, idle_timeout_seconds=3600)

    @pytest.mark.asyncio
    async def test_get_empty_pool(self, pool):
        """get() on empty pool should return None."""
        result = await pool.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_load_new_model(self, pool):
        """get_or_load() should load new model."""
        engine = MockEngine("model-a")
        manifest = MockManifest("model-a")

        async def loader():
            return engine, manifest, 1000.0

        cached = await pool.get_or_load("model-a", loader)

        assert cached.engine is engine
        assert cached.manifest is manifest
        assert cached.memory_estimate_mb == 1000.0
        assert pool.size == 1

    @pytest.mark.asyncio
    async def test_get_or_load_cached(self, pool):
        """get_or_load() should return cached model."""
        engine = MockEngine()
        manifest = MockManifest("model-a")
        load_count = 0

        async def loader():
            nonlocal load_count
            load_count += 1
            return engine, manifest, 500.0

        # First call - loads
        cached1 = await pool.get_or_load("model-a", loader)
        assert load_count == 1

        # Second call - cached
        cached2 = await pool.get_or_load("model-a", loader)
        assert load_count == 1
        assert cached1.engine is cached2.engine

    @pytest.mark.asyncio
    async def test_lru_eviction(self, pool):
        """Should evict LRU model when at capacity."""
        engines = {}

        async def create_loader(name):
            engine = MockEngine(name)
            engines[name] = engine
            manifest = MockManifest(name)
            return engine, manifest, 100.0

        # Fill pool
        for i in range(3):
            await pool.get_or_load(f"model-{i}", lambda i=i: create_loader(f"model-{i}"))

        assert pool.size == 3

        # Access model-0 to make it recently used
        await pool.get("model-0")

        # Add new model - should evict model-1 (LRU)
        await pool.get_or_load("model-3", lambda: create_loader("model-3"))

        assert pool.size == 3
        assert "model-0" in pool.cached_models
        assert "model-1" not in pool.cached_models
        assert "model-3" in pool.cached_models

    @pytest.mark.asyncio
    async def test_evict_specific_model(self, pool):
        """Should evict specific model."""
        engine = MockEngine()
        manifest = MockManifest("test")

        async def loader():
            return engine, manifest, 500.0

        await pool.get_or_load("test", loader)
        assert pool.size == 1

        result = await pool.evict("test")
        assert result is True
        assert pool.size == 0
        assert engine.unload_called

    @pytest.mark.asyncio
    async def test_evict_nonexistent(self, pool):
        """Evicting nonexistent model should return False."""
        result = await pool.evict("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear(self, pool):
        """clear() should remove all models."""
        engines = []

        for i in range(2):
            engine = MockEngine(f"model-{i}")
            engines.append(engine)
            manifest = MockManifest(f"model-{i}")

            async def loader(e=engine, m=manifest):
                return e, m, 100.0

            await pool.get_or_load(f"model-{i}", loader)

        assert pool.size == 2

        await pool.clear()

        assert pool.size == 0
        for engine in engines:
            assert engine.unload_called

    @pytest.mark.asyncio
    async def test_memory_tracking(self, pool):
        """Should track total memory usage."""

        async def loader(mem):
            return MockEngine(), MockManifest("test"), mem

        await pool.get_or_load("model-a", lambda: loader(1000.0))
        await pool.get_or_load("model-b", lambda: loader(2000.0))

        assert pool.total_memory_mb == 3000.0

    @pytest.mark.asyncio
    async def test_get_stats(self, pool):
        """Should return cache statistics."""

        async def loader():
            return MockEngine(), MockManifest("test"), 500.0

        await pool.get_or_load("test-model", loader)

        stats = pool.get_stats()
        assert stats["cached_models"] == 1
        assert stats["max_models"] == 3
        assert stats["total_memory_mb"] == 500.0
        assert len(stats["models"]) == 1
        assert stats["models"][0]["name"] == "test-model"


class TestBackgroundEviction:
    """Tests for background eviction functionality."""

    @pytest.mark.asyncio
    async def test_start_background_eviction(self):
        """start_background_eviction starts the background task."""
        pool = ModelPool(background_eviction_interval=0.1)
        pool.start_background_eviction()

        assert pool._background_task is not None
        assert not pool._background_task.done()

        await pool.stop_background_eviction()

    @pytest.mark.asyncio
    async def test_stop_background_eviction(self):
        """stop_background_eviction stops the background task."""
        pool = ModelPool(background_eviction_interval=0.1)
        pool.start_background_eviction()
        await pool.stop_background_eviction()

        assert pool._background_task is None or pool._background_task.done()

    @pytest.mark.asyncio
    async def test_background_evicts_idle_models(self):
        """Background eviction removes idle models."""
        import asyncio

        pool = ModelPool(
            idle_timeout_seconds=0.1,  # Very short timeout
            background_eviction_interval=0.05,  # Check every 50ms
        )

        engine = MockEngine()
        manifest = MockManifest("test")

        async def loader():
            return engine, manifest, 100.0

        await pool.get_or_load("test-model", loader)
        assert pool.size == 1

        # Start background eviction
        pool.start_background_eviction()

        # Wait for idle timeout + eviction interval
        await asyncio.sleep(0.3)

        # Model should be evicted
        assert pool.size == 0
        assert engine.unload_called

        await pool.stop_background_eviction()

    @pytest.mark.asyncio
    async def test_shutdown_stops_eviction_and_clears(self):
        """shutdown() stops background eviction and clears models."""
        pool = ModelPool(background_eviction_interval=0.1)

        engine = MockEngine()
        manifest = MockManifest("test")

        async def loader():
            return engine, manifest, 100.0

        await pool.get_or_load("test-model", loader)
        pool.start_background_eviction()

        await pool.shutdown()

        assert pool.size == 0
        assert pool._background_task is None or pool._background_task.done()

    @pytest.mark.asyncio
    async def test_get_stats_shows_background_eviction_status(self):
        """get_stats() shows background eviction status."""
        pool = ModelPool(background_eviction_interval=0.1)

        stats1 = pool.get_stats()
        assert stats1["background_eviction_active"] is False

        pool.start_background_eviction()
        stats2 = pool.get_stats()
        assert stats2["background_eviction_active"] is True

        await pool.stop_background_eviction()

    @pytest.mark.asyncio
    async def test_eviction_loop_recovers_from_single_error(self):
        """Background eviction should recover from single errors."""
        import asyncio

        pool = ModelPool(
            idle_timeout_seconds=0.1,
            background_eviction_interval=0.05,
        )

        error_count = 0

        original_evict = pool._evict_locked

        def failing_evict(name):
            nonlocal error_count
            error_count += 1
            if error_count == 1:
                raise RuntimeError("Simulated error")
            return original_evict(name)

        pool._evict_locked = failing_evict

        engine = MockEngine()
        manifest = MockManifest("test")

        async def loader():
            return engine, manifest, 100.0

        await pool.get_or_load("test-model", loader)
        pool.start_background_eviction()

        # Wait for recovery + successful eviction
        await asyncio.sleep(0.5)

        # Should have recovered and evicted
        await pool.stop_background_eviction()
        # Error occurred but loop recovered
        assert error_count >= 1

    @pytest.mark.asyncio
    async def test_eviction_loop_logs_errors_correctly(self):
        """Background eviction should log consecutive errors."""
        import asyncio

        pool = ModelPool(
            idle_timeout_seconds=0.01,
            background_eviction_interval=0.01,
        )

        error_count = 0

        def counting_fail(name):
            nonlocal error_count
            error_count += 1
            raise RuntimeError("Test failure")

        pool._evict_locked = counting_fail

        engine = MockEngine()
        manifest = MockManifest("test")

        async def loader():
            return engine, manifest, 100.0

        await pool.get_or_load("test-model", loader)
        pool.start_background_eviction()

        # Wait for first error cycle
        await asyncio.sleep(0.2)

        await pool.stop_background_eviction()

        # Should have hit at least one error
        assert error_count >= 1


class TestGetModelPool:
    """Tests for get_model_pool singleton."""

    def test_returns_same_instance(self):
        """Should return same instance."""
        reset_model_pool()

        pool1 = get_model_pool()
        pool2 = get_model_pool()

        assert pool1 is pool2

    def test_reset_clears_instance(self):
        """reset_model_pool should clear singleton."""
        pool1 = get_model_pool()
        reset_model_pool()
        pool2 = get_model_pool()

        assert pool1 is not pool2

    def test_max_loaded_models_default_is_one(self):
        """Default cap preserves V1 single-resident behaviour."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            for var in ("HFL_MAX_LOADED_MODELS", "OLLAMA_MAX_LOADED_MODELS"):
                os.environ.pop(var, None)
            reset_model_pool()
            pool = get_model_pool()
            assert pool._max_models == 1

    def test_max_loaded_models_picks_up_hfl_env(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"HFL_MAX_LOADED_MODELS": "4"}, clear=False):
            os.environ.pop("OLLAMA_MAX_LOADED_MODELS", None)
            reset_model_pool()
            pool = get_model_pool()
            assert pool._max_models == 4

    def test_max_loaded_models_picks_up_ollama_env(self):
        """Drop-in replace: an environment with ``OLLAMA_MAX_LOADED_MODELS``
        already set must work without changes."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"OLLAMA_MAX_LOADED_MODELS": "5"}, clear=False):
            os.environ.pop("HFL_MAX_LOADED_MODELS", None)
            reset_model_pool()
            pool = get_model_pool()
            assert pool._max_models == 5

    def test_max_loaded_models_explicit_arg_wins(self):
        """An explicit ``max_models=`` overrides any env var — tests
        rely on this to bypass the resolution chain."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"HFL_MAX_LOADED_MODELS": "10"}, clear=False):
            reset_model_pool()
            pool = get_model_pool(max_models=2)
            assert pool._max_models == 2


class TestConcurrentLoading:
    """Tests for concurrent model loading (deadlock prevention)."""

    @pytest.mark.asyncio
    async def test_concurrent_load_same_model_loads_once(self):
        """Concurrent loads of same model should only load once."""
        import asyncio

        pool = ModelPool(max_models=3, idle_timeout_seconds=3600)
        load_count = 0
        load_started = asyncio.Event()
        load_proceed = asyncio.Event()

        async def slow_loader():
            nonlocal load_count
            load_count += 1
            load_started.set()
            await load_proceed.wait()
            return MockEngine("test"), MockManifest("test"), 100.0

        # Start two concurrent loads
        task1 = asyncio.create_task(pool.get_or_load("test-model", slow_loader))
        await load_started.wait()

        # Second task should wait for first to complete
        task2 = asyncio.create_task(pool.get_or_load("test-model", slow_loader))
        await asyncio.sleep(0.05)  # Give task2 time to check cache

        # Let first load complete
        load_proceed.set()

        cached1, cached2 = await asyncio.gather(task1, task2)

        # Should have loaded only once
        assert load_count == 1
        assert cached1.engine is cached2.engine

    @pytest.mark.asyncio
    async def test_concurrent_load_different_models_parallel(self):
        """Concurrent loads of different models should run in parallel."""
        import asyncio

        pool = ModelPool(max_models=3, idle_timeout_seconds=3600)
        load_order = []

        async def loader_a():
            load_order.append("a_start")
            await asyncio.sleep(0.05)
            load_order.append("a_end")
            return MockEngine("a"), MockManifest("a"), 100.0

        async def loader_b():
            load_order.append("b_start")
            await asyncio.sleep(0.05)
            load_order.append("b_end")
            return MockEngine("b"), MockManifest("b"), 100.0

        # Load both concurrently
        await asyncio.gather(
            pool.get_or_load("model-a", loader_a),
            pool.get_or_load("model-b", loader_b),
        )

        # Both should start before either ends (parallel execution)
        a_start = load_order.index("a_start")
        b_start = load_order.index("b_start")
        a_end = load_order.index("a_end")
        b_end = load_order.index("b_end")

        # Both starts should happen before both ends
        assert a_start < a_end
        assert b_start < b_end
        assert pool.size == 2

    @pytest.mark.asyncio
    async def test_loader_not_blocking_other_operations(self):
        """Loading should not block get() operations on other models."""
        import asyncio

        pool = ModelPool(max_models=3, idle_timeout_seconds=3600)

        # Pre-load a model
        engine_existing = MockEngine("existing")
        manifest_existing = MockManifest("existing")

        async def existing_loader():
            return engine_existing, manifest_existing, 100.0

        await pool.get_or_load("existing-model", existing_loader)

        # Now start a slow load
        slow_load_started = asyncio.Event()

        async def slow_loader():
            slow_load_started.set()
            await asyncio.sleep(1.0)  # Long load time
            return MockEngine("slow"), MockManifest("slow"), 100.0

        slow_task = asyncio.create_task(pool.get_or_load("slow-model", slow_loader))
        await slow_load_started.wait()

        # Should be able to get existing model while slow load is in progress
        get_start = asyncio.get_event_loop().time()
        cached = await pool.get("existing-model")
        get_time = asyncio.get_event_loop().time() - get_start

        # Get should complete quickly (not blocked by slow load)
        assert cached is not None
        assert get_time < 0.1

        # Cancel the slow task
        slow_task.cancel()
        try:
            await slow_task
        except asyncio.CancelledError:
            pass


class TestEngineCleanupOnFailure:
    """Tests for engine cleanup when loading fails."""

    @pytest.mark.asyncio
    async def test_cleanup_engine_on_loader_failure(self):
        """Should cleanup engine if loader fails after creating engine."""
        pool = ModelPool(max_models=3, idle_timeout_seconds=3600)
        _engine = MockEngine("failing")  # noqa: F841

        async def failing_loader():
            # Simulate engine creation then failure
            raise RuntimeError("Load failed")

        with pytest.raises(RuntimeError, match="Load failed"):
            await pool.get_or_load("failing-model", failing_loader)

        # Pool should be empty
        assert pool.size == 0

    @pytest.mark.asyncio
    async def test_loading_set_cleared_on_failure(self):
        """Loading set should be cleared even on failure."""
        pool = ModelPool(max_models=3, idle_timeout_seconds=3600)

        async def failing_loader():
            raise RuntimeError("Load failed")

        with pytest.raises(RuntimeError):
            await pool.get_or_load("failing-model", failing_loader)

        # Model should not be in loading set
        assert "failing-model" not in pool._loading

    @pytest.mark.asyncio
    async def test_can_retry_after_failure(self):
        """Should be able to retry loading after failure."""
        pool = ModelPool(max_models=3, idle_timeout_seconds=3600)
        attempt = 0

        async def sometimes_failing_loader():
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise RuntimeError("First attempt fails")
            return MockEngine("success"), MockManifest("success"), 100.0

        # First attempt fails
        with pytest.raises(RuntimeError):
            await pool.get_or_load("retry-model", sometimes_failing_loader)

        # Second attempt succeeds
        cached = await pool.get_or_load("retry-model", sometimes_failing_loader)
        assert cached is not None
        assert pool.size == 1

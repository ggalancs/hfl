# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for model pooling and caching."""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

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

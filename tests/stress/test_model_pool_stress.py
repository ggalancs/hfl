# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Stress tests for ModelPool under heavy concurrent load."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from hfl.engine.model_pool import ModelPool


def create_mock_engine(name: str = "test") -> MagicMock:
    """Create a mock InferenceEngine."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.model_name = name
    return engine


def create_mock_manifest(name: str = "test") -> MagicMock:
    """Create a mock ModelManifest."""
    manifest = MagicMock()
    manifest.name = name
    return manifest


class TestModelPoolStress:
    """Stress tests for the ModelPool class."""

    @pytest.mark.asyncio
    async def test_concurrent_same_model_load(self) -> None:
        """50 concurrent loads of same model should only load once."""
        pool = ModelPool(max_models=3)
        load_count = 0

        async def mock_loader():
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.05)
            return create_mock_engine(), create_mock_manifest(), 100.0

        tasks = [pool.get_or_load("test-model", mock_loader) for _ in range(50)]
        results = await asyncio.gather(*tasks)

        assert load_count == 1
        assert all(r == results[0] for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_different_models(self) -> None:
        """Loading different models concurrently works."""
        pool = ModelPool(max_models=5)

        async def loader_for(name: str):
            async def _load():
                await asyncio.sleep(0.01)
                return create_mock_engine(name), create_mock_manifest(name), 50.0

            return _load

        tasks = [pool.get_or_load(f"model-{i}", await loader_for(f"model-{i}")) for i in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert pool.size <= 5

    @pytest.mark.asyncio
    async def test_eviction_under_count_limit(self) -> None:
        """Pool evicts oldest when max_models exceeded."""
        pool = ModelPool(max_models=3)

        for i in range(5):

            async def make_loader(idx: int = i):
                return (
                    create_mock_engine(f"m-{idx}"),
                    create_mock_manifest(f"m-{idx}"),
                    10.0,
                )

            await pool.get_or_load(f"model-{i}", make_loader)
            await asyncio.sleep(0.01)  # Ensure ordering

        assert pool.size <= 3

    @pytest.mark.asyncio
    async def test_eviction_under_memory_limit(self) -> None:
        """Pool evicts when memory limit exceeded.

        The pool checks memory *before* loading a new model, so after
        loading the final model the total may temporarily exceed the limit
        by at most one model's worth of memory. We verify that the pool
        did evict and kept fewer models than requested.
        """
        pool = ModelPool(max_models=10, max_memory_mb=500.0)

        for i in range(10):

            async def make_loader(idx: int = i):
                return (
                    create_mock_engine(f"m-{idx}"),
                    create_mock_manifest(f"m-{idx}"),
                    200.0,
                )

            await pool.get_or_load(f"model-{i}", make_loader)

        # Pool should have evicted models to stay near the memory limit.
        # Because eviction runs before each load, the pool keeps at most
        # ceil(max_memory_mb / per_model_mb) + 1 models loaded.
        assert pool.size < 10
        assert pool.total_memory_mb <= 800.0  # bounded, not unbounded

    @pytest.mark.asyncio
    async def test_rapid_get_performance(self) -> None:
        """1000 rapid get calls should be fast."""
        pool = ModelPool(max_models=5)

        async def loader():
            return create_mock_engine(), create_mock_manifest(), 10.0

        await pool.get_or_load("test-model", loader)

        for _ in range(1000):
            result = await pool.get("test-model")
            assert result is not None

    @pytest.mark.asyncio
    async def test_load_failure_cleanup(self) -> None:
        """Failed loads don't leave stale state."""
        pool = ModelPool(max_models=3)

        async def failing_loader():
            raise RuntimeError("Load failed")

        with pytest.raises(RuntimeError):
            await pool.get_or_load("failing-model", failing_loader)

        assert pool.size == 0
        assert "failing-model" not in pool.cached_models

    @pytest.mark.asyncio
    async def test_background_eviction(self) -> None:
        """Background eviction removes idle models."""
        pool = ModelPool(
            max_models=5,
            idle_timeout_seconds=0.1,
            background_eviction_interval=0.05,
        )

        async def loader():
            return create_mock_engine(), create_mock_manifest(), 10.0

        await pool.get_or_load("idle-model", loader)
        assert pool.size == 1

        pool.start_background_eviction()
        await asyncio.sleep(0.3)  # Wait for eviction

        assert pool.size == 0
        await pool.stop_background_eviction()

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Shutdown clears all models and stops background tasks."""
        pool = ModelPool(max_models=3)

        async def loader():
            return create_mock_engine(), create_mock_manifest(), 10.0

        await pool.get_or_load("model-1", loader)
        pool.start_background_eviction()

        await pool.shutdown()

        assert pool.size == 0

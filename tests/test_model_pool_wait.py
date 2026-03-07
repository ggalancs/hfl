# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for model pool non-recursive waiting."""

import asyncio
from unittest.mock import MagicMock

import pytest

from hfl.engine.model_pool import ModelPool


@pytest.fixture
def pool():
    return ModelPool(max_models=2, idle_timeout_seconds=60.0)


class TestModelPoolConcurrentLoad:
    @pytest.mark.asyncio
    async def test_concurrent_get_or_load_same_model(self, pool):
        """Two coroutines loading the same model should not deadlock."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_manifest = MagicMock()

        load_count = 0

        async def loader():
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.2)  # Simulate loading
            return mock_engine, mock_manifest, 100.0

        # Launch two concurrent loads
        results = await asyncio.gather(
            pool.get_or_load("model-a", loader),
            pool.get_or_load("model-a", loader),
        )

        # Both should succeed
        assert results[0] is not None
        assert results[1] is not None
        # Only one should have actually loaded (the other waited)
        assert load_count == 1

    @pytest.mark.asyncio
    async def test_get_or_load_no_stack_overflow(self, pool):
        """Non-recursive waiting should not cause stack overflow."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_manifest = MagicMock()

        async def loader():
            await asyncio.sleep(0.05)
            return mock_engine, mock_manifest, 50.0

        result = await pool.get_or_load("model-x", loader)
        assert result is not None
        assert result.engine == mock_engine

    @pytest.mark.asyncio
    async def test_cached_model_returns_immediately(self, pool):
        """Cached model should be returned without loading."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_manifest = MagicMock()

        async def loader():
            return mock_engine, mock_manifest, 50.0

        # First load
        await pool.get_or_load("model-y", loader)

        # Second call should return cached
        call_count = 0
        async def counting_loader():
            nonlocal call_count
            call_count += 1
            return mock_engine, mock_manifest, 50.0

        result = await pool.get_or_load("model-y", counting_loader)
        assert result is not None
        assert call_count == 0  # Loader not called for cached model

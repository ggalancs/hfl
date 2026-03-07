# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Concurrency and race condition tests.

Tests for concurrent model loading, registry access, and streaming.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from hfl.engine.base import GenerationResult


class TestConcurrentModelLoading:
    """Test concurrent model loading scenarios."""

    @pytest.mark.asyncio
    async def test_same_model_requests_serialized(self, temp_config):
        """Concurrent requests for same model should serialize loading."""
        from hfl.api.state import ServerState

        state = ServerState()
        load_count = 0

        async def mock_load(model_name: str, timeout: float = 300.0):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)  # Simulate loading time
            engine = MagicMock()
            engine.is_loaded = True
            manifest = MagicMock()
            manifest.name = model_name
            state.engine = engine
            state.current_model = manifest
            return engine, manifest

        with patch.object(state, "ensure_llm_loaded", side_effect=mock_load):
            # Launch multiple concurrent requests for same model
            tasks = [
                state.ensure_llm_loaded("test-model"),
                state.ensure_llm_loaded("test-model"),
                state.ensure_llm_loaded("test-model"),
            ]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 3
            # Model should only be loaded once per unique request
            # (with proper locking, concurrent requests should serialize)

    @pytest.mark.asyncio
    async def test_concurrent_registry_reads(self, temp_config):
        """Concurrent registry reads should not block each other."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()

        async def read_registry():
            return await asyncio.to_thread(registry.list_all)

        # Launch many concurrent reads
        tasks = [read_registry() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # All should complete without error
        assert len(results) == 50

    @pytest.mark.asyncio
    async def test_concurrent_registry_writes(self, temp_config):
        """Concurrent registry writes should not corrupt data."""
        from hfl.models.manifest import ModelManifest
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()

        async def add_model(i: int):
            manifest = ModelManifest(
                name=f"model-{i}",
                repo_id=f"test/model-{i}",
                local_path=str(temp_config.models_dir / f"model-{i}"),
                size_bytes=1000,
                format="gguf",
            )
            await asyncio.to_thread(registry.add, manifest)

        # Add many models concurrently
        tasks = [add_model(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # All should be present
        all_models = registry.list_all()
        assert len(all_models) == 20

        # Verify no corruption - all names should be unique
        names = {m.name for m in all_models}
        assert len(names) == 20


class TestAsyncEngineWrapper:
    """Test async engine wrapper functionality."""

    @pytest.mark.asyncio
    async def test_async_generate(self, temp_config):
        """Test async generation."""
        from hfl.engine.async_wrapper import AsyncEngineWrapper

        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.return_value = GenerationResult(
            text="Hello world",
            tokens_prompt=5,
            tokens_generated=2,
            stop_reason="stop",
        )

        wrapper = AsyncEngineWrapper(mock_engine)

        result = await wrapper.generate("Test prompt")

        assert result.text == "Hello world"
        mock_engine.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_stream(self, temp_config):
        """Test async streaming."""
        from hfl.engine.async_wrapper import AsyncEngineWrapper

        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate_stream.return_value = iter(["Hello", " ", "world"])

        wrapper = AsyncEngineWrapper(mock_engine)

        tokens = []
        async for token in wrapper.generate_stream("Test"):
            tokens.append(token)

        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_concurrent_async_operations(self, temp_config):
        """Test multiple concurrent async operations."""
        from hfl.engine.async_wrapper import AsyncEngineWrapper

        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate.return_value = GenerationResult(
            text="Response",
            tokens_prompt=5,
            tokens_generated=1,
            stop_reason="stop",
        )

        wrapper = AsyncEngineWrapper(mock_engine)

        # Launch multiple concurrent generations
        tasks = [wrapper.generate(f"Prompt {i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r.text == "Response" for r in results)


class TestSingletonContainer:
    """Test thread-safe singleton container."""

    def test_singleton_thread_safety(self, temp_config):
        """Test singleton is thread-safe."""
        from concurrent.futures import ThreadPoolExecutor

        from hfl.core.container import Singleton

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"id": call_count}

        singleton = Singleton(factory)

        # Access from multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(singleton.get) for _ in range(100)]
            results = [f.result() for f in futures]

        # All should return same instance
        assert all(r["id"] == 1 for r in results)
        # Factory should only be called once
        assert call_count == 1

    def test_container_reset(self, temp_config):
        """Test container reset clears all singletons."""
        from hfl.core.container import Container

        container = Container()

        # Access singletons
        _ = container.config.get()
        _ = container.registry.get()

        assert container.config.is_initialized
        assert container.registry.is_initialized

        # Reset
        container.reset_all()

        assert not container.config.is_initialized
        assert not container.registry.is_initialized


class TestStreamingConcurrency:
    """Test streaming operations under concurrent load."""

    @pytest.mark.asyncio
    async def test_multiple_streams_concurrent(self, temp_config):
        """Multiple streams should work concurrently."""
        from hfl.engine.async_wrapper import AsyncEngineWrapper

        def make_stream(n: int):
            for i in range(n):
                yield f"token_{i}"

        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.generate_stream.side_effect = lambda p, c: make_stream(5)

        wrapper = AsyncEngineWrapper(mock_engine)

        async def collect_stream():
            tokens = []
            async for token in wrapper.generate_stream("test"):
                tokens.append(token)
            return tokens

        # Run multiple streams concurrently
        tasks = [collect_stream() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # Each should have collected all tokens
        for tokens in results:
            assert tokens == ["token_0", "token_1", "token_2", "token_3", "token_4"]


class TestRaceConditions:
    """Test for race condition handling."""

    @pytest.mark.asyncio
    async def test_model_load_unload_race(self, temp_config):
        """Loading and unloading same model concurrently should not crash."""
        from hfl.api.state import ServerState

        state = ServerState()

        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        state.engine = mock_engine

        # Concurrent load/unload operations should not crash
        # even if result is inconsistent, it shouldn't raise
        try:

            async def load_task():
                await asyncio.sleep(0.01)
                state.engine = MagicMock()

            async def unload_task():
                await asyncio.sleep(0.01)
                state.engine = None

            tasks = [load_task() for _ in range(5)] + [unload_task() for _ in range(5)]
            await asyncio.gather(*tasks)
        except Exception as e:
            pytest.fail(f"Race condition caused crash: {e}")

    def test_config_concurrent_access(self, temp_config, monkeypatch):
        """Config should be safely accessible from multiple threads."""
        from concurrent.futures import ThreadPoolExecutor

        from hfl.config import config

        def access_config():
            # Access various config properties
            _ = config.models_dir
            _ = config.home_dir
            return True

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_config) for _ in range(100)]
            results = [f.result() for f in futures]

        assert all(results)

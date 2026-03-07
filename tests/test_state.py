# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for server state management."""

import asyncio
from dataclasses import dataclass

import pytest

from hfl.api.state import ServerState, get_state, reset_state


@dataclass
class MockManifest:
    """Mock model manifest for testing."""

    name: str
    local_path: str = "/mock/path"
    context_length: int = 4096


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

    def load(self, path: str, **kwargs) -> None:
        self._loaded = True


class MockTTSEngine:
    """Mock TTS engine for testing."""

    def __init__(self, name: str = "test-tts"):
        self.name = name
        self._loaded = True
        self.unload_called = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False
        self.unload_called = True

    def load(self, path: str, **kwargs) -> None:
        self._loaded = True


class TestServerStateProperties:
    """Tests for ServerState property accessors."""

    def test_engine_property_getter_setter(self):
        """engine property works correctly."""
        state = ServerState()
        assert state.engine is None

        engine = MockEngine()
        state.engine = engine
        assert state.engine is engine

    def test_current_model_property_getter_setter(self):
        """current_model property works correctly."""
        state = ServerState()
        assert state.current_model is None

        manifest = MockManifest("test-model")
        state.current_model = manifest
        assert state.current_model is manifest

    def test_tts_engine_property_getter_setter(self):
        """tts_engine property works correctly."""
        state = ServerState()
        assert state.tts_engine is None

        engine = MockTTSEngine()
        state.tts_engine = engine
        assert state.tts_engine is engine

    def test_current_tts_model_property_getter_setter(self):
        """current_tts_model property works correctly."""
        state = ServerState()
        assert state.current_tts_model is None

        manifest = MockManifest("test-tts-model")
        state.current_tts_model = manifest
        assert state.current_tts_model is manifest

    def test_api_key_property_getter_setter(self):
        """api_key property works correctly."""
        state = ServerState()
        assert state.api_key is None

        state.api_key = "test-key-123"
        assert state.api_key == "test-key-123"

    def test_is_loading_property(self):
        """is_loading returns correct state."""
        state = ServerState()
        assert state.is_loading is False

        state._loading_models.add("test-model")
        assert state.is_loading is True

        state._loading_models.clear()
        assert state.is_loading is False

    def test_loading_models_property_returns_copy(self):
        """loading_models returns a copy, not the original set."""
        state = ServerState()
        state._loading_models.add("model1")

        models = state.loading_models
        models.add("model2")

        # Original should not be modified
        assert "model2" not in state._loading_models


class TestServerStateLLMOperations:
    """Tests for LLM-related server state operations."""

    @pytest.mark.asyncio
    async def test_set_llm_engine_new_engine(self):
        """set_llm_engine sets new engine correctly."""
        state = ServerState()
        engine = MockEngine()
        manifest = MockManifest("test-model")

        await state.set_llm_engine(engine, manifest)

        assert state.engine is engine
        assert state.current_model is manifest

    @pytest.mark.asyncio
    async def test_set_llm_engine_unloads_previous(self):
        """set_llm_engine unloads previous engine."""
        state = ServerState()
        old_engine = MockEngine("old")
        new_engine = MockEngine("new")
        old_manifest = MockManifest("old-model")
        new_manifest = MockManifest("new-model")

        await state.set_llm_engine(old_engine, old_manifest)
        await state.set_llm_engine(new_engine, new_manifest)

        assert old_engine.unload_called
        assert state.engine is new_engine

    @pytest.mark.asyncio
    async def test_set_llm_engine_none_clears_state(self):
        """set_llm_engine with None clears state."""
        state = ServerState()
        engine = MockEngine()
        manifest = MockManifest("test-model")

        await state.set_llm_engine(engine, manifest)
        await state.set_llm_engine(None, None)

        assert state.engine is None
        assert state.current_model is None
        assert engine.unload_called

    def test_is_llm_loaded_true(self):
        """is_llm_loaded returns True when engine is loaded."""
        state = ServerState()
        engine = MockEngine()
        state._engine = engine

        assert state.is_llm_loaded() is True

    def test_is_llm_loaded_false_no_engine(self):
        """is_llm_loaded returns False when no engine."""
        state = ServerState()
        assert state.is_llm_loaded() is False

    def test_is_llm_loaded_false_engine_not_loaded(self):
        """is_llm_loaded returns False when engine not loaded."""
        state = ServerState()
        engine = MockEngine()
        engine._loaded = False
        state._engine = engine

        assert state.is_llm_loaded() is False

    @pytest.mark.asyncio
    async def test_with_llm_engine_yields_engine(self):
        """with_llm_engine yields the engine."""
        state = ServerState()
        engine = MockEngine()
        manifest = MockManifest("test-model")
        state._engine = engine
        state._current_model = manifest

        async with state.with_llm_engine() as yielded_engine:
            assert yielded_engine is engine

    @pytest.mark.asyncio
    async def test_with_llm_engine_raises_when_no_engine(self):
        """with_llm_engine raises ModelNotLoadedError when no engine."""
        from hfl.exceptions import ModelNotLoadedError

        state = ServerState()

        with pytest.raises(ModelNotLoadedError):
            async with state.with_llm_engine():
                pass

    @pytest.mark.asyncio
    async def test_with_llm_engine_holds_lock(self):
        """with_llm_engine holds the lock while in context."""
        state = ServerState()
        engine = MockEngine()
        state._engine = engine

        lock_was_held = False

        async def check_lock():
            nonlocal lock_was_held
            lock_was_held = state._llm_lock.locked()

        async with state.with_llm_engine():
            await check_lock()

        assert lock_was_held

    @pytest.mark.asyncio
    async def test_with_llm_engine_releases_lock_after(self):
        """with_llm_engine releases lock after context exit."""
        state = ServerState()
        engine = MockEngine()
        state._engine = engine

        async with state.with_llm_engine():
            pass

        assert not state._llm_lock.locked()

    @pytest.mark.asyncio
    async def test_with_llm_engine_releases_lock_on_exception(self):
        """with_llm_engine releases lock on exception."""
        state = ServerState()
        engine = MockEngine()
        state._engine = engine

        with pytest.raises(ValueError):
            async with state.with_llm_engine():
                raise ValueError("Test error")

        assert not state._llm_lock.locked()


class TestServerStateTTSOperations:
    """Tests for TTS-related server state operations."""

    @pytest.mark.asyncio
    async def test_set_tts_engine_new_engine(self):
        """set_tts_engine sets new engine correctly."""
        state = ServerState()
        engine = MockTTSEngine()
        manifest = MockManifest("test-tts")

        await state.set_tts_engine(engine, manifest)

        assert state.tts_engine is engine
        assert state.current_tts_model is manifest

    @pytest.mark.asyncio
    async def test_set_tts_engine_unloads_previous(self):
        """set_tts_engine unloads previous engine."""
        state = ServerState()
        old_engine = MockTTSEngine("old")
        new_engine = MockTTSEngine("new")
        old_manifest = MockManifest("old-model")
        new_manifest = MockManifest("new-model")

        await state.set_tts_engine(old_engine, old_manifest)
        await state.set_tts_engine(new_engine, new_manifest)

        assert old_engine.unload_called
        assert state.tts_engine is new_engine

    @pytest.mark.asyncio
    async def test_set_tts_engine_none_clears_state(self):
        """set_tts_engine with None clears state."""
        state = ServerState()
        engine = MockTTSEngine()
        manifest = MockManifest("test-tts")

        await state.set_tts_engine(engine, manifest)
        await state.set_tts_engine(None, None)

        assert state.tts_engine is None
        assert state.current_tts_model is None
        assert engine.unload_called

    def test_is_tts_loaded_true(self):
        """is_tts_loaded returns True when engine is loaded."""
        state = ServerState()
        engine = MockTTSEngine()
        state._tts_engine = engine

        assert state.is_tts_loaded() is True

    def test_is_tts_loaded_false_no_engine(self):
        """is_tts_loaded returns False when no engine."""
        state = ServerState()
        assert state.is_tts_loaded() is False

    def test_is_tts_loaded_false_engine_not_loaded(self):
        """is_tts_loaded returns False when engine not loaded."""
        state = ServerState()
        engine = MockTTSEngine()
        engine._loaded = False
        state._tts_engine = engine

        assert state.is_tts_loaded() is False

    @pytest.mark.asyncio
    async def test_with_tts_engine_yields_engine(self):
        """with_tts_engine yields the TTS engine."""
        state = ServerState()
        engine = MockTTSEngine()
        manifest = MockManifest("test-tts")
        state._tts_engine = engine
        state._current_tts_model = manifest

        async with state.with_tts_engine() as yielded_engine:
            assert yielded_engine is engine

    @pytest.mark.asyncio
    async def test_with_tts_engine_raises_when_no_engine(self):
        """with_tts_engine raises ModelNotLoadedError when no engine."""
        from hfl.exceptions import ModelNotLoadedError

        state = ServerState()

        with pytest.raises(ModelNotLoadedError):
            async with state.with_tts_engine():
                pass

    @pytest.mark.asyncio
    async def test_with_tts_engine_holds_lock(self):
        """with_tts_engine holds the lock while in context."""
        state = ServerState()
        engine = MockTTSEngine()
        state._tts_engine = engine

        lock_was_held = False

        async def check_lock():
            nonlocal lock_was_held
            lock_was_held = state._tts_lock.locked()

        async with state.with_tts_engine():
            await check_lock()

        assert lock_was_held

    @pytest.mark.asyncio
    async def test_with_tts_engine_releases_lock_on_exception(self):
        """with_tts_engine releases lock on exception."""
        state = ServerState()
        engine = MockTTSEngine()
        state._tts_engine = engine

        with pytest.raises(ValueError):
            async with state.with_tts_engine():
                raise ValueError("Test error")

        assert not state._tts_lock.locked()


class TestEnsureLLMLoaded:
    """Tests for ensure_llm_loaded method."""

    @pytest.mark.asyncio
    async def test_fast_path_already_loaded(self):
        """ensure_llm_loaded returns immediately if model loaded."""
        state = ServerState()
        engine = MockEngine()
        manifest = MockManifest("test-model")
        state._engine = engine
        state._current_model = manifest

        loader_called = False

        async def loader():
            nonlocal loader_called
            loader_called = True
            return MockEngine(), MockManifest("other")

        result_engine, result_manifest = await state.ensure_llm_loaded("test-model", loader)

        assert not loader_called
        assert result_engine is engine
        assert result_manifest is manifest

    @pytest.mark.asyncio
    async def test_loads_new_model(self):
        """ensure_llm_loaded loads new model."""
        state = ServerState()
        new_engine = MockEngine("new")
        new_manifest = MockManifest("new-model")

        async def loader():
            return new_engine, new_manifest

        result_engine, result_manifest = await state.ensure_llm_loaded("new-model", loader)

        assert result_engine is new_engine
        assert result_manifest is new_manifest
        assert state.engine is new_engine

    @pytest.mark.asyncio
    async def test_tracks_loading_state(self):
        """ensure_llm_loaded tracks loading state."""
        state = ServerState()
        loading_during_call = False

        async def loader():
            nonlocal loading_during_call
            loading_during_call = "test-model" in state._loading_models
            return MockEngine(), MockManifest("test-model")

        await state.ensure_llm_loaded("test-model", loader)

        assert loading_during_call
        assert "test-model" not in state._loading_models

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self):
        """ensure_llm_loaded raises TimeoutError on timeout."""
        state = ServerState()

        async def slow_loader():
            await asyncio.sleep(10)
            return MockEngine(), MockManifest("test")

        with pytest.raises(asyncio.TimeoutError):
            await state.ensure_llm_loaded("test-model", slow_loader, timeout=0.1)

    @pytest.mark.asyncio
    async def test_cleans_up_loading_state_on_error(self):
        """ensure_llm_loaded cleans up loading state on error."""
        state = ServerState()

        async def failing_loader():
            raise ValueError("Load failed")

        with pytest.raises(ValueError):
            await state.ensure_llm_loaded("test-model", failing_loader)

        assert "test-model" not in state._loading_models

    @pytest.mark.asyncio
    async def test_serializes_concurrent_loads(self):
        """ensure_llm_loaded serializes concurrent loads of same model."""
        state = ServerState()
        load_count = 0

        async def counting_loader():
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)
            return MockEngine(), MockManifest("test-model")

        # Start two concurrent loads for same model
        results = await asyncio.gather(
            state.ensure_llm_loaded("test-model", counting_loader),
            state.ensure_llm_loaded("test-model", counting_loader),
        )

        # Should only load once due to double-check locking
        assert load_count == 1


class TestEnsureTTSLoaded:
    """Tests for ensure_tts_loaded method."""

    @pytest.mark.asyncio
    async def test_fast_path_already_loaded(self):
        """ensure_tts_loaded returns immediately if model loaded."""
        state = ServerState()
        engine = MockTTSEngine()
        manifest = MockManifest("test-tts")
        state._tts_engine = engine
        state._current_tts_model = manifest

        loader_called = False

        async def loader():
            nonlocal loader_called
            loader_called = True
            return MockTTSEngine(), MockManifest("other")

        result_engine, result_manifest = await state.ensure_tts_loaded("test-tts", loader)

        assert not loader_called
        assert result_engine is engine

    @pytest.mark.asyncio
    async def test_loads_new_model(self):
        """ensure_tts_loaded loads new model."""
        state = ServerState()
        new_engine = MockTTSEngine("new")
        new_manifest = MockManifest("new-tts")

        async def loader():
            return new_engine, new_manifest

        result_engine, result_manifest = await state.ensure_tts_loaded("new-tts", loader)

        assert result_engine is new_engine
        assert state.tts_engine is new_engine

    @pytest.mark.asyncio
    async def test_tracks_loading_state(self):
        """ensure_tts_loaded tracks loading state."""
        state = ServerState()
        loading_during_call = False

        async def loader():
            nonlocal loading_during_call
            loading_during_call = "tts:test-tts" in state._loading_models
            return MockTTSEngine(), MockManifest("test-tts")

        await state.ensure_tts_loaded("test-tts", loader)

        assert loading_during_call
        assert "tts:test-tts" not in state._loading_models

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self):
        """ensure_tts_loaded raises TimeoutError on timeout."""
        state = ServerState()

        async def slow_loader():
            await asyncio.sleep(10)
            return MockTTSEngine(), MockManifest("test")

        with pytest.raises(asyncio.TimeoutError):
            await state.ensure_tts_loaded("test-tts", slow_loader, timeout=0.1)


class TestServerStateCleanup:
    """Tests for cleanup method."""

    @pytest.mark.asyncio
    async def test_cleanup_unloads_llm_engine(self):
        """cleanup unloads LLM engine."""
        state = ServerState()
        engine = MockEngine()
        manifest = MockManifest("test")
        state._engine = engine
        state._current_model = manifest

        await state.cleanup()

        assert engine.unload_called
        assert state.engine is None
        assert state.current_model is None

    @pytest.mark.asyncio
    async def test_cleanup_unloads_tts_engine(self):
        """cleanup unloads TTS engine."""
        state = ServerState()
        engine = MockTTSEngine()
        manifest = MockManifest("test-tts")
        state._tts_engine = engine
        state._current_tts_model = manifest

        await state.cleanup()

        assert engine.unload_called
        assert state.tts_engine is None
        assert state.current_tts_model is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_no_engines(self):
        """cleanup handles case with no engines loaded."""
        state = ServerState()

        # Should not raise
        await state.cleanup()

        assert state.engine is None
        assert state.tts_engine is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_unloaded_engines(self):
        """cleanup handles already unloaded engines."""
        state = ServerState()
        engine = MockEngine()
        engine._loaded = False
        state._engine = engine

        # Should not call unload again
        await state.cleanup()

        assert not engine.unload_called


class TestSingletonFunctions:
    """Tests for get_state and reset_state functions."""

    def test_get_state_returns_singleton(self):
        """get_state returns same instance."""
        reset_state()

        state1 = get_state()
        state2 = get_state()

        assert state1 is state2

    def test_reset_state_clears_instance(self):
        """reset_state clears singleton."""
        state1 = get_state()
        reset_state()
        state2 = get_state()

        assert state1 is not state2

    def test_get_state_creates_new_after_reset(self):
        """get_state creates fresh instance after reset."""
        state = get_state()
        state.api_key = "test-key"

        reset_state()
        new_state = get_state()

        assert new_state.api_key is None

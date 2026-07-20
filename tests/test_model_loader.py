# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for model loader module."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hfl.api.model_loader import load_llm, load_llm_sync, load_tts
from hfl.api.state import reset_state
from hfl.converter.formats import ModelType
from hfl.exceptions import (
    ModelNotFoundError,
    ModelNotReadyError,
    ModelTypeMismatchError,
)
from hfl.exceptions import (
    ValidationError as APIValidationError,
)


@dataclass
class MockManifest:
    """Mock model manifest for testing."""

    name: str
    local_path: str = "/mock/path/to/model"
    context_length: int = 4096


class MockEngine:
    """Mock inference engine for testing."""

    def __init__(self, name: str = "test"):
        self.name = name
        self._loaded = True
        self.load_called = False
        self.load_args = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str, **kwargs) -> None:
        self.load_called = True
        self.load_args = (path, kwargs)
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False


class MockTTSEngine:
    """Mock TTS engine for testing."""

    def __init__(self, name: str = "test-tts"):
        self.name = name
        self._loaded = True
        self.load_called = False
        self.load_args = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str, **kwargs) -> None:
        self.load_called = True
        self.load_args = (path, kwargs)
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False


class TestLoadLLM:
    """Tests for load_llm async function."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset state before each test."""
        reset_state()
        yield
        reset_state()

    @pytest.mark.asyncio
    async def test_invalid_model_name_raises_400(self):
        """Invalid model name raises ValidationError (maps to 400)."""
        with pytest.raises(APIValidationError) as exc_info:
            await load_llm("")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    async def test_fast_path_returns_loaded_model(self, mock_get_state):
        """Fast path returns already loaded model."""
        mock_engine = MockEngine()
        mock_manifest = MockManifest("test-model")

        mock_state = MagicMock()
        mock_state.current_model = mock_manifest
        mock_state.engine = mock_engine
        mock_get_state.return_value = mock_state

        engine, manifest = await load_llm("test-model")

        assert engine is mock_engine
        assert manifest is mock_manifest

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    async def test_fast_path_engine_none_raises_503(self, mock_get_state):
        """Fast path with engine None raises ModelNotReadyError (503)."""
        mock_manifest = MockManifest("test-model")

        mock_state = MagicMock()
        mock_state.current_model = mock_manifest
        mock_state.engine = None
        mock_get_state.return_value = mock_state

        with pytest.raises(ModelNotReadyError) as exc_info:
            await load_llm("test-model")

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    @patch("hfl.api.model_loader.get_registry")
    async def test_model_not_found_raises_404(self, mock_get_registry, mock_get_state):
        """Model not in registry raises ModelNotFoundError (404)."""
        mock_state = MagicMock()
        mock_state.current_model = None
        mock_get_state.return_value = mock_state

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        with pytest.raises(ModelNotFoundError) as exc_info:
            await load_llm("nonexistent-model")

        assert exc_info.value.status_code == 404
        assert "Model not found" in exc_info.value.message
        assert exc_info.value.model_name == "nonexistent-model"

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    @patch("hfl.api.model_loader.get_registry")
    @patch("hfl.api.model_loader.detect_model_type")
    async def test_wrong_model_type_raises_400(
        self,
        mock_detect,
        mock_get_registry,
        mock_get_state,
    ):
        """Wrong model type raises ModelTypeMismatchError (400)."""
        mock_state = MagicMock()
        mock_state.current_model = None
        mock_get_state.return_value = mock_state

        mock_manifest = MockManifest("test-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.TTS  # Wrong type for LLM

        with pytest.raises(ModelTypeMismatchError) as exc_info:
            await load_llm("test-model")

        assert exc_info.value.status_code == 400
        assert exc_info.value.expected == "llm"
        assert exc_info.value.got == "tts"

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.asyncio.to_thread")
    @patch("hfl.api.model_loader.select_engine")
    @patch("hfl.api.model_loader.detect_model_type")
    @patch("hfl.api.model_loader.get_registry")
    @patch("hfl.api.model_loader.get_state")
    async def test_loads_model_successfully(
        self,
        mock_get_state,
        mock_get_registry,
        mock_detect,
        mock_select,
        mock_to_thread,
    ):
        """A cold load is delegated to state.ensure_llm_loaded (the per-model
        coalescing primitive); the loader it passes loads off the event loop and
        returns (engine, manifest)."""
        mock_engine = MockEngine()
        mock_manifest = MockManifest("new-model")

        mock_state = MagicMock()
        mock_state.current_model = None
        mock_state.context_size_override = 0
        mock_state.ensure_llm_loaded = AsyncMock(return_value=(mock_engine, mock_manifest))
        mock_get_state.return_value = mock_state

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry
        mock_detect.return_value = ModelType.LLM
        mock_select.return_value = mock_engine
        mock_to_thread.return_value = None

        engine, manifest = await load_llm("new-model")
        assert engine is mock_engine
        assert manifest is mock_manifest

        # Delegated to the coalescing primitive with the requested model name.
        mock_state.ensure_llm_loaded.assert_called_once()
        call = mock_state.ensure_llm_loaded.call_args
        assert call.args[0] == "new-model"

        # The passed loader loads off-loop and returns (engine, manifest).
        loader = call.args[1]
        result = await loader()
        assert result == (mock_engine, mock_manifest)
        mock_to_thread.assert_called()  # engine.load ran via to_thread


class TestLoadLLMCleanupOnFailure:
    """Tests for the cleanup path after a post-load failure."""

    @pytest.fixture(autouse=True)
    def reset(self):
        reset_state()
        yield
        reset_state()

    @staticmethod
    def _driving_ensure_loaded():
        """A stand-in for state.ensure_llm_loaded that actually DRIVES the loader
        load_llm passes it (the real primitive does the same under a per-model
        lock), so the loader's load-failure cleanup is exercised."""

        async def _ensure(model_name, loader, **kwargs):
            return await loader()

        return _ensure

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.asyncio.to_thread")
    @patch("hfl.api.model_loader.select_engine")
    @patch("hfl.api.model_loader.detect_model_type")
    @patch("hfl.api.model_loader.get_registry")
    @patch("hfl.api.model_loader.get_state")
    async def test_load_failure_unloads_engine(
        self,
        mock_get_state,
        mock_get_registry,
        mock_detect,
        mock_select,
        mock_to_thread,
    ):
        """If ``engine.load()`` fails, the loader must unload the half-loaded
        engine so we don't leak an orphaned model (ram/GPU held with no owner),
        then re-raise the original error.
        """
        mock_state = MagicMock()
        mock_state.current_model = None
        mock_state.context_size_override = 0
        mock_state.ensure_llm_loaded = self._driving_ensure_loaded()
        mock_get_state.return_value = mock_state

        mock_manifest = MockManifest("new-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.LLM

        mock_engine = MockEngine()  # is_loaded stays True so the cleanup unload runs
        mock_select.return_value = mock_engine
        # to_thread: load() raises, cleanup unload() succeeds.
        mock_to_thread.side_effect = [RuntimeError("load boom"), None]

        with pytest.raises(RuntimeError, match="load boom"):
            await load_llm("new-model")

        # Both the load and the cleanup unload were routed through to_thread.
        assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.logger")
    @patch("hfl.api.model_loader.asyncio.to_thread")
    @patch("hfl.api.model_loader.select_engine")
    @patch("hfl.api.model_loader.detect_model_type")
    @patch("hfl.api.model_loader.get_registry")
    @patch("hfl.api.model_loader.get_state")
    async def test_unload_failure_is_logged_not_propagated(
        self,
        mock_get_state,
        mock_get_registry,
        mock_detect,
        mock_select,
        mock_to_thread,
        mock_logger,
    ):
        """If the cleanup ``unload()`` itself raises, the ORIGINAL load error
        still propagates and the unload error is only logged — otherwise the
        root cause gets masked by a cleanup failure.
        """
        mock_state = MagicMock()
        mock_state.current_model = None
        mock_state.context_size_override = 0
        mock_state.ensure_llm_loaded = self._driving_ensure_loaded()
        mock_get_state.return_value = mock_state

        mock_manifest = MockManifest("new-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.LLM

        mock_engine = MockEngine()
        mock_select.return_value = mock_engine

        # to_thread: load() raises the root cause, then the cleanup unload() also
        # raises — the cleanup error must be swallowed (logged), root re-raised.
        mock_to_thread.side_effect = [RuntimeError("root cause"), RuntimeError("cleanup boom")]

        with pytest.raises(RuntimeError, match="root cause"):
            await load_llm("new-model")

        # The cleanup error must be logged, not re-raised.
        assert mock_logger.error.called
        args, _ = mock_logger.error.call_args
        assert "cleanup" in args[0].lower()


class TestLoadTTS:
    """Tests for load_tts async function."""

    @pytest.fixture(autouse=True)
    def reset(self):
        """Reset state before each test."""
        reset_state()
        yield
        reset_state()

    @pytest.mark.asyncio
    async def test_invalid_model_name_raises_400(self):
        """Invalid model name raises ValidationError (maps to 400)."""
        with pytest.raises(APIValidationError) as exc_info:
            await load_tts("")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    async def test_fast_path_returns_loaded_tts(self, mock_get_state):
        """Fast path returns already loaded TTS model."""
        mock_engine = MockTTSEngine()
        mock_manifest = MockManifest("test-tts")

        mock_state = MagicMock()
        mock_state.current_tts_model = mock_manifest
        mock_state.tts_engine = mock_engine
        mock_get_state.return_value = mock_state

        engine, manifest = await load_tts("test-tts")

        assert engine is mock_engine
        assert manifest is mock_manifest

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    async def test_fast_path_engine_none_raises_503(self, mock_get_state):
        """Fast path with tts_engine None raises ModelNotReadyError (503)."""
        mock_manifest = MockManifest("test-tts")

        mock_state = MagicMock()
        mock_state.current_tts_model = mock_manifest
        mock_state.tts_engine = None
        mock_get_state.return_value = mock_state

        with pytest.raises(ModelNotReadyError) as exc_info:
            await load_tts("test-tts")

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    @patch("hfl.api.model_loader.get_registry")
    async def test_model_not_found_raises_404(self, mock_get_registry, mock_get_state):
        """Model not in registry raises ModelNotFoundError (404)."""
        mock_state = MagicMock()
        mock_state.current_tts_model = None
        mock_get_state.return_value = mock_state

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        with pytest.raises(ModelNotFoundError) as exc_info:
            await load_tts("nonexistent-tts")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    @patch("hfl.api.model_loader.get_registry")
    @patch("hfl.api.model_loader.detect_model_type")
    async def test_wrong_model_type_raises_400(
        self,
        mock_detect,
        mock_get_registry,
        mock_get_state,
    ):
        """Wrong model type raises ModelTypeMismatchError (400)."""
        mock_state = MagicMock()
        mock_state.current_tts_model = None
        mock_get_state.return_value = mock_state

        mock_manifest = MockManifest("test-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.LLM  # Wrong type for TTS

        with pytest.raises(ModelTypeMismatchError) as exc_info:
            await load_tts("test-model")

        assert exc_info.value.status_code == 400
        assert exc_info.value.expected == "tts"
        assert exc_info.value.got == "llm"

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.asyncio.to_thread")
    @patch("hfl.api.model_loader.select_tts_engine")
    @patch("hfl.api.model_loader.detect_model_type")
    @patch("hfl.api.model_loader.get_registry")
    @patch("hfl.api.model_loader.get_state")
    async def test_loads_tts_successfully(
        self,
        mock_get_state,
        mock_get_registry,
        mock_detect,
        mock_select,
        mock_to_thread,
    ):
        """Successfully loads TTS model through full path."""
        mock_state = MagicMock()
        mock_state.current_tts_model = None
        mock_state.set_tts_engine = AsyncMock()
        mock_get_state.return_value = mock_state

        mock_manifest = MockManifest("new-tts")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.TTS

        mock_engine = MockTTSEngine()
        mock_select.return_value = mock_engine

        mock_to_thread.return_value = None

        engine, manifest = await load_tts("new-tts")

        assert engine is mock_engine
        assert manifest is mock_manifest
        mock_state.set_tts_engine.assert_called_once_with(mock_engine, mock_manifest)


class TestLoadLLMSync:
    """Tests for load_llm_sync function."""

    def test_invalid_model_name_raises_value_error(self):
        """Invalid model name raises ValueError."""
        from hfl.validators import ValidationError

        with pytest.raises(ValidationError):
            load_llm_sync("")

    @patch("hfl.api.model_loader.get_registry")
    def test_model_not_found_raises_value_error(self, mock_get_registry):
        """Model not in registry raises ValueError."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        with pytest.raises(ValueError) as exc_info:
            load_llm_sync("nonexistent-model")

        assert "Model not found" in str(exc_info.value)

    @patch("hfl.api.model_loader.detect_model_type")
    @patch("hfl.api.model_loader.get_registry")
    def test_wrong_model_type_raises_value_error(self, mock_get_registry, mock_detect):
        """Wrong model type raises ValueError."""
        mock_manifest = MockManifest("test-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.TTS

        with pytest.raises(ValueError) as exc_info:
            load_llm_sync("test-model")

        assert "Expected LLM model" in str(exc_info.value)

    @patch("hfl.api.model_loader.select_engine")
    @patch("hfl.api.model_loader.detect_model_type")
    @patch("hfl.api.model_loader.get_registry")
    def test_loads_model_successfully(self, mock_get_registry, mock_detect, mock_select):
        """Successfully loads model synchronously."""
        mock_manifest = MockManifest("test-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.LLM

        mock_engine = MockEngine()
        mock_select.return_value = mock_engine

        engine, manifest = load_llm_sync("test-model")

        assert engine is mock_engine
        assert manifest is mock_manifest
        assert mock_engine.load_called

    @patch("hfl.api.model_loader.select_engine")
    @patch("hfl.api.model_loader.detect_model_type")
    @patch("hfl.api.model_loader.get_registry")
    def test_passes_context_length_to_engine(self, mock_get_registry, mock_detect, mock_select):
        """Context length from manifest is passed to engine.load."""
        mock_manifest = MockManifest("test-model", context_length=8192)
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.LLM

        mock_engine = MockEngine()
        mock_select.return_value = mock_engine

        load_llm_sync("test-model")

        assert mock_engine.load_args[1]["n_ctx"] == 8192

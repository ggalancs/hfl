# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for model loader module."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from hfl.api.model_loader import load_llm, load_llm_sync, load_tts
from hfl.api.state import reset_state
from hfl.converter.formats import ModelType


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
        """Invalid model name raises HTTPException 400."""
        with pytest.raises(HTTPException) as exc_info:
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
        """Fast path with engine None raises 503."""
        mock_manifest = MockManifest("test-model")

        mock_state = MagicMock()
        mock_state.current_model = mock_manifest
        mock_state.engine = None
        mock_get_state.return_value = mock_state

        with pytest.raises(HTTPException) as exc_info:
            await load_llm("test-model")

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    @patch("hfl.api.model_loader.get_registry")
    async def test_model_not_found_raises_404(self, mock_get_registry, mock_get_state):
        """Model not in registry raises HTTPException 404."""
        mock_state = MagicMock()
        mock_state.current_model = None
        mock_get_state.return_value = mock_state

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        with pytest.raises(HTTPException) as exc_info:
            await load_llm("nonexistent-model")

        assert exc_info.value.status_code == 404
        assert "Model not found" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    @patch("hfl.api.model_loader.get_registry")
    @patch("hfl.api.model_loader.detect_model_type")
    async def test_wrong_model_type_raises_400(
        self, mock_detect, mock_get_registry, mock_get_state,
    ):
        """Wrong model type raises HTTPException 400."""
        mock_state = MagicMock()
        mock_state.current_model = None
        mock_get_state.return_value = mock_state

        mock_manifest = MockManifest("test-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.TTS  # Wrong type for LLM

        with pytest.raises(HTTPException) as exc_info:
            await load_llm("test-model")

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "MODEL_TYPE_MISMATCH"

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
        """Successfully loads model through full path."""
        mock_state = MagicMock()
        mock_state.current_model = None
        mock_state.set_llm_engine = AsyncMock()
        mock_get_state.return_value = mock_state

        mock_manifest = MockManifest("new-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.LLM

        mock_engine = MockEngine()
        mock_select.return_value = mock_engine

        mock_to_thread.return_value = None

        engine, manifest = await load_llm("new-model")

        assert engine is mock_engine
        assert manifest is mock_manifest
        mock_state.set_llm_engine.assert_called_once_with(mock_engine, mock_manifest)


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
        """Invalid model name raises HTTPException 400."""
        with pytest.raises(HTTPException) as exc_info:
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
        """Fast path with tts_engine None raises 503."""
        mock_manifest = MockManifest("test-tts")

        mock_state = MagicMock()
        mock_state.current_tts_model = mock_manifest
        mock_state.tts_engine = None
        mock_get_state.return_value = mock_state

        with pytest.raises(HTTPException) as exc_info:
            await load_tts("test-tts")

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    @patch("hfl.api.model_loader.get_registry")
    async def test_model_not_found_raises_404(self, mock_get_registry, mock_get_state):
        """Model not in registry raises HTTPException 404."""
        mock_state = MagicMock()
        mock_state.current_tts_model = None
        mock_get_state.return_value = mock_state

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        with pytest.raises(HTTPException) as exc_info:
            await load_tts("nonexistent-tts")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch("hfl.api.model_loader.get_state")
    @patch("hfl.api.model_loader.get_registry")
    @patch("hfl.api.model_loader.detect_model_type")
    async def test_wrong_model_type_raises_400(
        self, mock_detect, mock_get_registry, mock_get_state,
    ):
        """Wrong model type raises HTTPException 400."""
        mock_state = MagicMock()
        mock_state.current_tts_model = None
        mock_get_state.return_value = mock_state

        mock_manifest = MockManifest("test-model")
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_get_registry.return_value = mock_registry

        mock_detect.return_value = ModelType.LLM  # Wrong type for TTS

        with pytest.raises(HTTPException) as exc_info:
            await load_tts("test-model")

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "MODEL_TYPE_MISMATCH"
        assert exc_info.value.detail["expected"] == "tts"

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

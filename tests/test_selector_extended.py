# SPDX-License-Identifier: HRUL-1.0
"""Extended tests for engine selector module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest

from hfl.engine.selector import (
    select_engine,
    _create_engine,
    _has_cuda,
    MissingDependencyError,
)


class TestGetVLLMEngine:
    """Tests for _get_vllm_engine function."""

    def test_create_vllm_via_create_engine(self):
        """Test creating vllm engine via _create_engine."""
        with patch("hfl.engine.selector._get_vllm_engine") as mock_get:
            mock_engine = MagicMock()
            mock_get.return_value = mock_engine

            result = _create_engine("vllm")

            mock_get.assert_called_once()
            assert result is mock_engine


class TestCreateEngine:
    """Tests for _create_engine function."""

    def test_create_llama_cpp_engine(self):
        """Test creating llama-cpp engine."""
        with patch("hfl.engine.selector._get_llama_cpp_engine") as mock_get:
            mock_engine = MagicMock()
            mock_get.return_value = mock_engine

            result = _create_engine("llama-cpp")

            mock_get.assert_called_once()
            assert result is mock_engine

    def test_create_transformers_engine(self):
        """Test creating transformers engine."""
        with patch("hfl.engine.selector._get_transformers_engine") as mock_get:
            mock_engine = MagicMock()
            mock_get.return_value = mock_engine

            result = _create_engine("transformers")

            mock_get.assert_called_once()
            assert result is mock_engine

    def test_create_vllm_engine(self):
        """Test creating vllm engine."""
        with patch("hfl.engine.selector._get_vllm_engine") as mock_get:
            mock_engine = MagicMock()
            mock_get.return_value = mock_engine

            result = _create_engine("vllm")

            mock_get.assert_called_once()
            assert result is mock_engine

    def test_create_unknown_engine_raises(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _create_engine("unknown-backend")

        assert "unknown" in str(exc_info.value).lower()


class TestHasCuda:
    """Tests for _has_cuda function."""

    def test_has_cuda_function_exists(self):
        """Test _has_cuda function exists and can be imported."""
        from hfl.engine.selector import _has_cuda

        # Function should be callable
        assert callable(_has_cuda)


class TestSelectEngineExtended:
    """Extended tests for select_engine function."""

    def test_select_engine_for_gguf(self, tmp_path):
        """Test selecting engine for GGUF file."""
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()

        with patch("hfl.engine.selector._get_llama_cpp_engine") as mock_get:
            mock_engine = MagicMock()
            mock_get.return_value = mock_engine

            result = select_engine(gguf_file)

            mock_get.assert_called_once()
            assert result is mock_engine

    def test_select_engine_for_safetensors_with_cuda(self, tmp_path):
        """Test selecting engine for safetensors with CUDA available."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()

        with patch("hfl.engine.selector._has_cuda") as mock_cuda:
            mock_cuda.return_value = True

            with patch("hfl.engine.selector._get_transformers_engine") as mock_get:
                mock_engine = MagicMock()
                mock_get.return_value = mock_engine

                result = select_engine(model_dir)

                mock_get.assert_called_once()

    def test_select_engine_for_safetensors_without_cuda(self, tmp_path):
        """Test selecting engine for safetensors without CUDA falls back to llama.cpp."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()

        with patch("hfl.engine.selector._has_cuda") as mock_cuda:
            mock_cuda.return_value = False

            with patch("hfl.engine.selector._get_llama_cpp_engine") as mock_get:
                mock_engine = MagicMock()
                mock_get.return_value = mock_engine

                result = select_engine(model_dir)

                # Falls back to llama.cpp when no CUDA
                mock_get.assert_called_once()

    def test_select_engine_for_safetensors_transformers_missing(self, tmp_path):
        """Test fallback to llama.cpp when transformers not available."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()

        with patch("hfl.engine.selector._has_cuda") as mock_cuda:
            mock_cuda.return_value = True

            with patch("hfl.engine.selector._get_transformers_engine") as mock_trans:
                mock_trans.side_effect = MissingDependencyError("transformers not installed")

                with patch("hfl.engine.selector._get_llama_cpp_engine") as mock_llama:
                    mock_engine = MagicMock()
                    mock_llama.return_value = mock_engine

                    result = select_engine(model_dir)

                    # Falls back to llama.cpp
                    mock_llama.assert_called_once()

    def test_select_engine_explicit_backend(self, tmp_path):
        """Test selecting engine with explicit backend."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("hfl.engine.selector._create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine

            result = select_engine(model_dir, backend="vllm")

            mock_create.assert_called_once_with("vllm")


class TestMissingDependencyError:
    """Tests for MissingDependencyError exception."""

    def test_missing_dependency_error_message(self):
        """Test MissingDependencyError message."""
        error = MissingDependencyError("Test dependency missing")

        assert "Test dependency missing" in str(error)

    def test_missing_dependency_error_inheritance(self):
        """Test MissingDependencyError is an Exception."""
        error = MissingDependencyError("Test")

        assert isinstance(error, Exception)

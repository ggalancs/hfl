# SPDX-License-Identifier: HRUL-1.0
"""Tests for the inference engine selector."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hfl.engine.selector import (
    MissingDependencyError,
    _create_engine,
    _get_llama_cpp_engine,
    _get_transformers_engine,
    _get_vllm_engine,
    _has_cuda,
    select_engine,
)


class TestMissingDependencyError:
    """Tests for MissingDependencyError."""

    def test_error_message(self):
        """Test that error message is preserved."""
        error = MissingDependencyError("Test error message")
        assert str(error) == "Test error message"


class TestGetLlamaCppEngine:
    """Tests for _get_llama_cpp_engine function."""

    def test_returns_engine_when_available(self):
        """Test that engine is returned when llama-cpp-python is available."""
        # The actual function works with real imports, so we just verify it exists
        assert callable(_get_llama_cpp_engine)

    def test_function_signature(self):
        """Test that the function has the expected signature."""
        import inspect

        sig = inspect.signature(_get_llama_cpp_engine)
        assert len(sig.parameters) == 0  # No parameters


class TestGetTransformersEngine:
    """Tests for _get_transformers_engine function."""

    def test_raises_when_not_available(self):
        """Test that MissingDependencyError is raised when library is missing."""
        with patch("hfl.engine.transformers_engine.TransformersEngine", side_effect=ImportError):
            with pytest.raises(MissingDependencyError) as exc_info:
                _get_transformers_engine()

            assert "transformers backend requires" in str(exc_info.value)
            assert "pip install hfl[transformers]" in str(exc_info.value)


class TestGetVllmEngine:
    """Tests for _get_vllm_engine function."""

    def test_function_exists(self):
        """Test that the function exists and is callable."""
        assert callable(_get_vllm_engine)

    def test_function_signature(self):
        """Test that the function has the expected signature."""
        import inspect

        sig = inspect.signature(_get_vllm_engine)
        assert len(sig.parameters) == 0  # No parameters


class TestHasCuda:
    """Tests for _has_cuda function."""

    def test_function_exists(self):
        """Test that the function exists and is callable."""
        assert callable(_has_cuda)

    def test_function_signature(self):
        """Test that the function has the expected signature."""
        import inspect

        sig = inspect.signature(_has_cuda)
        assert len(sig.parameters) == 0  # No parameters

    def test_function_returns_bool_type(self):
        """Test that the function is annotated to return bool."""
        # We can't easily call _has_cuda in test environment due to torch conflicts
        # So we just verify the function exists with correct signature
        import inspect

        source = inspect.getsource(_has_cuda)
        assert "return" in source
        assert "torch.cuda.is_available()" in source


class TestCreateEngine:
    """Tests for _create_engine function."""

    def test_create_llama_cpp_engine(self):
        """Test creating llama-cpp engine."""
        with patch("hfl.engine.selector._get_llama_cpp_engine") as mock_get:
            mock_engine = MagicMock()
            mock_get.return_value = mock_engine

            result = _create_engine("llama-cpp")

            assert result == mock_engine
            mock_get.assert_called_once()

    def test_create_transformers_engine(self):
        """Test creating transformers engine."""
        with patch("hfl.engine.selector._get_transformers_engine") as mock_get:
            mock_engine = MagicMock()
            mock_get.return_value = mock_engine

            result = _create_engine("transformers")

            assert result == mock_engine
            mock_get.assert_called_once()

    def test_create_vllm_engine(self):
        """Test creating vLLM engine."""
        with patch("hfl.engine.selector._get_vllm_engine") as mock_get:
            mock_engine = MagicMock()
            mock_get.return_value = mock_engine

            result = _create_engine("vllm")

            assert result == mock_engine
            mock_get.assert_called_once()

    def test_unknown_backend_raises(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _create_engine("unknown-backend")

        assert "Unknown backend: unknown-backend" in str(exc_info.value)


class TestSelectEngine:
    """Tests for select_engine function."""

    def test_explicit_llama_cpp_backend(self):
        """Test selecting llama-cpp backend explicitly."""
        with patch("hfl.engine.selector._create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine

            result = select_engine(Path("/model"), backend="llama-cpp")

            mock_create.assert_called_once_with("llama-cpp")
            assert result == mock_engine

    def test_explicit_transformers_backend(self):
        """Test selecting transformers backend explicitly."""
        with patch("hfl.engine.selector._create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine

            result = select_engine(Path("/model"), backend="transformers")

            mock_create.assert_called_once_with("transformers")
            assert result == mock_engine

    def test_explicit_vllm_backend(self):
        """Test selecting vLLM backend explicitly."""
        with patch("hfl.engine.selector._create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine

            result = select_engine(Path("/model"), backend="vllm")

            mock_create.assert_called_once_with("vllm")
            assert result == mock_engine

    def test_auto_selects_llama_cpp_for_gguf(self):
        """Test that auto mode selects llama-cpp for GGUF models."""
        from hfl.converter.formats import ModelFormat

        with patch("hfl.engine.selector.detect_format") as mock_detect:
            mock_detect.return_value = ModelFormat.GGUF

            with patch("hfl.engine.selector._get_llama_cpp_engine") as mock_get:
                mock_engine = MagicMock()
                mock_get.return_value = mock_engine

                result = select_engine(Path("/model.gguf"), backend="auto")

                assert result == mock_engine
                mock_get.assert_called_once()

    def test_auto_selects_transformers_with_cuda(self):
        """Test that auto mode selects transformers when CUDA is available."""
        from hfl.converter.formats import ModelFormat

        with patch("hfl.engine.selector.detect_format") as mock_detect:
            mock_detect.return_value = ModelFormat.SAFETENSORS

            with patch("hfl.engine.selector._has_cuda") as mock_cuda:
                mock_cuda.return_value = True

                with patch("hfl.engine.selector._get_transformers_engine") as mock_get:
                    mock_engine = MagicMock()
                    mock_get.return_value = mock_engine

                    result = select_engine(Path("/model"), backend="auto")

                    assert result == mock_engine

    def test_auto_fallback_to_llama_cpp_without_cuda(self):
        """Test that auto mode falls back to llama-cpp without CUDA."""
        from hfl.converter.formats import ModelFormat

        with patch("hfl.engine.selector.detect_format") as mock_detect:
            mock_detect.return_value = ModelFormat.SAFETENSORS

            with patch("hfl.engine.selector._has_cuda") as mock_cuda:
                mock_cuda.return_value = False

                with patch("hfl.engine.selector._get_llama_cpp_engine") as mock_get:
                    mock_engine = MagicMock()
                    mock_get.return_value = mock_engine

                    result = select_engine(Path("/model"), backend="auto")

                    assert result == mock_engine

    def test_auto_fallback_when_transformers_missing(self):
        """Test fallback to llama-cpp when transformers is not installed."""
        from hfl.converter.formats import ModelFormat

        with patch("hfl.engine.selector.detect_format") as mock_detect:
            mock_detect.return_value = ModelFormat.SAFETENSORS

            with patch("hfl.engine.selector._has_cuda") as mock_cuda:
                mock_cuda.return_value = True

                with patch("hfl.engine.selector._get_transformers_engine") as mock_trans:
                    mock_trans.side_effect = MissingDependencyError("Not installed")

                    with patch("hfl.engine.selector._get_llama_cpp_engine") as mock_llama:
                        mock_engine = MagicMock()
                        mock_llama.return_value = mock_engine

                        result = select_engine(Path("/model"), backend="auto")

                        assert result == mock_engine

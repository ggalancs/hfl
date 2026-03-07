# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Error handling tests for the TransformersEngine backend."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestTransformersEngineLoadErrors:
    """Tests for error handling during model loading."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = True
        mock.cuda.empty_cache = MagicMock()
        mock.bfloat16 = "bfloat16"
        return mock

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers module."""
        mock = MagicMock()
        mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
        mock.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
        mock.BitsAndBytesConfig = MagicMock()
        return mock

    def test_load_tokenizer_failure(self, mock_torch, mock_transformers):
        """Test that tokenizer load failure is handled properly."""
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = RuntimeError(
            "Cannot load tokenizer"
        )

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            with pytest.raises(RuntimeError, match="Cannot load tokenizer"):
                engine.load("/path/to/model")

            # Engine should remain unloaded
            assert engine.is_loaded is False
            assert engine._model is None
            assert engine._tokenizer is None

    def test_load_model_failure(self, mock_torch, mock_transformers):
        """Test that model load failure cleans up tokenizer."""
        mock_tokenizer = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError(
            "Out of memory"
        )

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
                "gc": MagicMock(),
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            with pytest.raises(RuntimeError, match="Out of memory"):
                engine.load("/path/to/model")

            # Engine should remain unloaded
            assert engine.is_loaded is False
            assert engine._model is None
            assert engine._tokenizer is None

            # CUDA cache should be cleared on failure
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_load_failure_clears_cuda_cache(self, mock_torch, mock_transformers):
        """Test that CUDA cache is cleared on load failure."""
        mock_transformers.AutoModelForCausalLM.from_pretrained.side_effect = Exception("GPU error")

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            with pytest.raises(Exception, match="GPU error"):
                engine.load("/path/to/model")

            mock_torch.cuda.empty_cache.assert_called()

    def test_load_failure_no_cuda_available(self, mock_torch, mock_transformers):
        """Test load failure when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        mock_transformers.AutoModelForCausalLM.from_pretrained.side_effect = Exception(
            "Load failed"
        )

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            with pytest.raises(Exception, match="Load failed"):
                engine.load("/path/to/model")

            # Should not try to clear CUDA cache if not available
            mock_torch.cuda.empty_cache.assert_not_called()


class TestTransformersEngineUnloadErrors:
    """Tests for error handling during model unloading."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = True
        mock.cuda.empty_cache = MagicMock()
        mock.cuda.synchronize = MagicMock()
        return mock

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers module."""
        mock = MagicMock()
        mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
        mock.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
        return mock

    def test_unload_cuda_error_handled(self, mock_torch, mock_transformers):
        """Test that CUDA errors during unload are handled gracefully."""
        mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA error")

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()
            engine.load("/path/to/model")

            # Should not raise despite CUDA error
            engine.unload()

            # Model should be unloaded
            assert engine._model is None
            assert engine._tokenizer is None

    def test_unload_synchronize_error_handled(self, mock_torch, mock_transformers):
        """Test that CUDA synchronize errors are handled gracefully."""
        mock_torch.cuda.synchronize.side_effect = RuntimeError("Sync error")

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()
            engine.load("/path/to/model")

            # Should not raise despite synchronize error
            engine.unload()

            assert engine._model is None

    def test_unload_when_cuda_not_available(self, mock_torch, mock_transformers):
        """Test unload works when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()
            engine.load("/path/to/model")
            engine.unload()

            # Should not try to clear CUDA cache or synchronize
            mock_torch.cuda.empty_cache.assert_not_called()
            mock_torch.cuda.synchronize.assert_not_called()
            assert engine._model is None


class TestTransformersEngineGenerateErrors:
    """Tests for error handling during text generation."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module with tensor support."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = True
        mock.no_grad.return_value.__enter__ = MagicMock()
        mock.no_grad.return_value.__exit__ = MagicMock()
        return mock

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers module."""
        mock = MagicMock()
        return mock

    def test_generate_without_loaded_model_raises(self, mock_torch, mock_transformers):
        """Test that calling generate without a loaded model raises TypeError."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            # Should raise TypeError because tokenizer is None
            with pytest.raises(TypeError):
                engine.generate("Hello")

    def test_chat_without_loaded_model_raises(self, mock_torch, mock_transformers):
        """Test that calling chat without a loaded model raises TypeError."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.base import ChatMessage
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()
            messages = [ChatMessage(role="user", content="Hello")]

            # Should raise TypeError because tokenizer is None
            with pytest.raises(TypeError):
                engine.chat(messages)

    def test_generate_stream_without_loaded_model_raises(self, mock_torch, mock_transformers):
        """Test that calling generate_stream without a loaded model raises."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            # Should raise TypeError because tokenizer is None
            with pytest.raises(TypeError):
                list(engine.generate_stream("Hello"))


class TestTransformersEngineQuantizationErrors:
    """Tests for error handling during quantization setup."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = True
        mock.bfloat16 = "bfloat16"
        return mock

    def test_4bit_quantization_with_missing_bitsandbytes(self, mock_torch):
        """Test 4-bit quantization fails gracefully without bitsandbytes."""
        mock_transformers = MagicMock()
        # Simulate BitsAndBytesConfig not being available
        del mock_transformers.BitsAndBytesConfig

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            # Should raise ImportError because BitsAndBytesConfig is not available
            with pytest.raises(ImportError):
                engine.load("/path/to/model", quantization="4bit")

    def test_8bit_quantization_with_missing_bitsandbytes(self, mock_torch):
        """Test 8-bit quantization fails gracefully without bitsandbytes."""
        mock_transformers = MagicMock()
        # Simulate BitsAndBytesConfig not being available
        del mock_transformers.BitsAndBytesConfig

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            # Should raise ImportError because BitsAndBytesConfig is not available
            with pytest.raises(ImportError):
                engine.load("/path/to/model", quantization="8bit")


class TestTransformersEngineBuildPromptErrors:
    """Tests for error handling in prompt building."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        return MagicMock()

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers module."""
        return MagicMock()

    def test_build_prompt_with_empty_messages(self, mock_torch, mock_transformers):
        """Test _build_prompt with empty messages list."""
        mock_tokenizer = MagicMock(spec=[])  # No apply_chat_template
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()
            engine.load("/path/to/model")

            # Empty messages should return empty string
            result = engine._build_prompt([])
            assert result == ""

    def test_build_prompt_with_chat_template_error(self, mock_torch, mock_transformers):
        """Test _build_prompt handles chat template errors gracefully."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = ValueError("Invalid template")
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.base import ChatMessage
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()
            engine.load("/path/to/model")

            messages = [ChatMessage(role="user", content="Hello")]

            # Should propagate the error
            with pytest.raises(ValueError, match="Invalid template"):
                engine._build_prompt(messages)

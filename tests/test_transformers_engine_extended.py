# SPDX-License-Identifier: HRUL-1.0
"""Extended tests for the TransformersEngine backend."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestTransformersEngineProperties:
    """Tests for TransformersEngine properties."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = True
        mock.cuda.empty_cache = MagicMock()
        mock.no_grad.return_value.__enter__ = MagicMock()
        mock.no_grad.return_value.__exit__ = MagicMock()
        mock.bfloat16 = "bfloat16"
        return mock

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers module."""
        mock = MagicMock()
        mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
        mock.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
        mock.BitsAndBytesConfig = MagicMock()
        mock.TextIteratorStreamer = MagicMock()
        return mock

    def test_is_loaded_true_after_load(self, mock_torch, mock_transformers):
        """Test is_loaded returns True after loading a model."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            assert engine.is_loaded is False

            engine.load("/path/to/model")

            assert engine.is_loaded is True

    def test_model_name_after_load(self, mock_torch, mock_transformers):
        """Test model_name returns correct path after loading."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            assert engine.model_name == ""

            engine.load("/path/to/my-model")

            assert engine.model_name == "/path/to/my-model"

    def test_is_loaded_false_after_unload(self, mock_torch, mock_transformers):
        """Test is_loaded returns False after unloading."""
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

            assert engine.is_loaded is True

            engine.unload()

            assert engine.is_loaded is False


class TestTransformersEngineBuildPrompt:
    """Tests for _build_prompt method."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = False
        return mock

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers module."""
        mock = MagicMock()
        return mock

    def test_build_prompt_user_only(self, mock_torch, mock_transformers):
        """Test _build_prompt with user message only."""
        mock_tokenizer = MagicMock(spec=[])  # No apply_chat_template
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

            messages = [
                ChatMessage(role="user", content="Hello"),
            ]
            result = engine._build_prompt(messages)

            assert "[INST] Hello [/INST]" in result

    def test_build_prompt_assistant_only(self, mock_torch, mock_transformers):
        """Test _build_prompt with assistant message."""
        mock_tokenizer = MagicMock(spec=[])  # No apply_chat_template
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

            messages = [
                ChatMessage(role="assistant", content="I can help!"),
            ]
            result = engine._build_prompt(messages)

            assert "I can help!" in result


class TestTransformersEngineLoadOptions:
    """Tests for load method options."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        mock = MagicMock()
        mock.cuda.is_available.return_value = True
        mock.bfloat16 = "bfloat16"
        return mock

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers module."""
        mock = MagicMock()
        return mock

    def test_load_with_no_quantization(self, mock_torch, mock_transformers):
        """Test load without quantization."""
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

            # BitsAndBytesConfig should NOT be called
            mock_transformers.BitsAndBytesConfig.assert_not_called()

    def test_load_with_torch_dtype(self, mock_torch, mock_transformers):
        """Test load with custom torch_dtype."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()
            engine.load("/path/to/model", torch_dtype="float16")

            call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args[1]
            assert call_kwargs["torch_dtype"] == "float16"

    def test_load_with_max_memory(self, mock_torch, mock_transformers):
        """Test load with max_memory setting."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()
            engine.load("/path/to/model", device_map="auto")

            call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args[1]
            assert call_kwargs["device_map"] == "auto"


class TestTransformersEngineMethodSignatures:
    """Tests for method signatures and existence."""

    @pytest.fixture
    def mock_torch(self):
        """Mock torch module."""
        return MagicMock()

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers module."""
        return MagicMock()

    def test_generate_method_signature(self, mock_torch, mock_transformers):
        """Test generate method has correct signature."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            import inspect

            sig = inspect.signature(engine.generate)
            params = list(sig.parameters.keys())

            assert "prompt" in params
            assert "config" in params

    def test_generate_stream_method_signature(self, mock_torch, mock_transformers):
        """Test generate_stream method has correct signature."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            import inspect

            sig = inspect.signature(engine.generate_stream)
            params = list(sig.parameters.keys())

            assert "prompt" in params
            assert "config" in params

    def test_chat_method_signature(self, mock_torch, mock_transformers):
        """Test chat method has correct signature."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            import inspect

            sig = inspect.signature(engine.chat)
            params = list(sig.parameters.keys())

            assert "messages" in params
            assert "config" in params

    def test_chat_stream_method_signature(self, mock_torch, mock_transformers):
        """Test chat_stream method has correct signature."""
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            from hfl.engine.transformers_engine import TransformersEngine

            engine = TransformersEngine()

            import inspect

            sig = inspect.signature(engine.chat_stream)
            params = list(sig.parameters.keys())

            assert "messages" in params
            assert "config" in params

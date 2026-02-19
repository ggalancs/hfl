# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the engine/vllm_engine module."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock vLLM before importing the module
@pytest.fixture(autouse=True)
def mock_vllm():
    """Mock of the vLLM module for tests without GPU."""
    mock_llm = MagicMock()
    mock_sampling = MagicMock()

    with patch.dict(sys.modules, {"vllm": MagicMock(LLM=mock_llm, SamplingParams=mock_sampling)}):
        yield mock_llm, mock_sampling


class TestVLLMEngine:
    """Tests for VLLMEngine."""

    def test_initialization(self, mock_vllm):
        """Verifies correct initialization."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        assert engine._model is None
        assert engine._model_path == ""

    def test_is_loaded_false_initially(self, mock_vllm):
        """Not loaded initially."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        assert engine.is_loaded is False

    def test_is_loaded_true_after_load(self, mock_vllm):
        """Loaded after load()."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._model = MagicMock()

        assert engine.is_loaded is True

    def test_model_name(self, mock_vllm):
        """Returns the model name."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._model_path = "test-model-path"

        assert engine.model_name == "test-model-path"

    def test_load_model(self, mock_vllm):
        """Loads a model."""
        mock_llm_class, _ = mock_vllm

        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine.load("/path/to/model", tensor_parallel_size=2)

        assert engine._model_path == "/path/to/model"
        mock_llm_class.assert_called_once_with(model="/path/to/model", tensor_parallel_size=2)

    def test_unload_model(self, mock_vllm):
        """Unloads the model."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._model = MagicMock()

        engine.unload()

        assert engine._model is None

    def test_generate_without_model_raises(self, mock_vllm):
        """Generate without model raises error."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.generate("test prompt")

    def test_generate_with_model(self, mock_vllm):
        """Generate with loaded model."""
        from hfl.engine.base import GenerationConfig
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        # Configure model mock
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated text", token_ids=[1, 2, 3])]
        engine._model = MagicMock()
        engine._model.generate.return_value = [mock_output]

        result = engine.generate("Test prompt", GenerationConfig(max_tokens=100))

        assert result.text == "Generated text"
        assert result.tokens_generated == 3

    def test_generate_stream(self, mock_vllm):
        """Generate stream returns text in one chunk."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Streamed text", token_ids=[1, 2])]
        engine._model = MagicMock()
        engine._model.generate.return_value = [mock_output]

        chunks = list(engine.generate_stream("Test prompt"))

        assert len(chunks) == 1
        assert chunks[0] == "Streamed text"

    def test_chat(self, mock_vllm):
        """Chat completion."""
        from hfl.engine.base import ChatMessage
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Chat response", token_ids=[1])]
        engine._model = MagicMock()
        engine._model.generate.return_value = [mock_output]

        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
        ]
        result = engine.chat(messages)

        assert result.text == "Chat response"

    def test_chat_stream(self, mock_vllm):
        """Chat stream."""
        from hfl.engine.base import ChatMessage
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Stream chat", token_ids=[1])]
        engine._model = MagicMock()
        engine._model.generate.return_value = [mock_output]

        messages = [ChatMessage(role="user", content="Hi")]
        chunks = list(engine.chat_stream(messages))

        assert len(chunks) == 1
        assert chunks[0] == "Stream chat"

    def test_build_prompt_system(self, mock_vllm):
        """Builds prompt with system message."""
        from hfl.engine.base import ChatMessage
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        messages = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hello"),
        ]
        prompt = engine._build_prompt(messages)

        assert "System: Be helpful" in prompt
        assert "User: Hello" in prompt
        assert prompt.endswith("Assistant:")

    def test_build_prompt_with_assistant(self, mock_vllm):
        """Builds prompt with assistant history."""
        from hfl.engine.base import ChatMessage
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        prompt = engine._build_prompt(messages)

        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "User: How are you?" in prompt

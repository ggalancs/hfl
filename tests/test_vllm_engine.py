# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests para el módulo engine/vllm_engine."""

import pytest
import sys
from unittest.mock import MagicMock, patch


# Mock vLLM antes de importar el módulo
@pytest.fixture(autouse=True)
def mock_vllm():
    """Mock del módulo vLLM para tests sin GPU."""
    mock_llm = MagicMock()
    mock_sampling = MagicMock()

    with patch.dict(sys.modules, {"vllm": MagicMock(LLM=mock_llm, SamplingParams=mock_sampling)}):
        yield mock_llm, mock_sampling


class TestVLLMEngine:
    """Tests para VLLMEngine."""

    def test_initialization(self, mock_vllm):
        """Verifica inicialización correcta."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        assert engine._model is None
        assert engine._model_path == ""

    def test_is_loaded_false_initially(self, mock_vllm):
        """No cargado inicialmente."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        assert engine.is_loaded is False

    def test_is_loaded_true_after_load(self, mock_vllm):
        """Cargado después de load()."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._model = MagicMock()

        assert engine.is_loaded is True

    def test_model_name(self, mock_vllm):
        """Devuelve el nombre del modelo."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._model_path = "test-model-path"

        assert engine.model_name == "test-model-path"

    def test_load_model(self, mock_vllm):
        """Carga un modelo."""
        mock_llm_class, _ = mock_vllm

        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine.load("/path/to/model", tensor_parallel_size=2)

        assert engine._model_path == "/path/to/model"
        mock_llm_class.assert_called_once_with(model="/path/to/model", tensor_parallel_size=2)

    def test_unload_model(self, mock_vllm):
        """Descarga el modelo."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._model = MagicMock()

        engine.unload()

        assert engine._model is None

    def test_generate_without_model_raises(self, mock_vllm):
        """Generate sin modelo lanza error."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()

        with pytest.raises(RuntimeError, match="Modelo no cargado"):
            engine.generate("test prompt")

    def test_generate_with_model(self, mock_vllm):
        """Generate con modelo cargado."""
        from hfl.engine.vllm_engine import VLLMEngine
        from hfl.engine.base import GenerationConfig

        engine = VLLMEngine()

        # Configurar mock del modelo
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated text", token_ids=[1, 2, 3])]
        engine._model = MagicMock()
        engine._model.generate.return_value = [mock_output]

        result = engine.generate("Test prompt", GenerationConfig(max_tokens=100))

        assert result.text == "Generated text"
        assert result.tokens_generated == 3

    def test_generate_stream(self, mock_vllm):
        """Generate stream devuelve texto en un chunk."""
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
        from hfl.engine.vllm_engine import VLLMEngine
        from hfl.engine.base import ChatMessage

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
        from hfl.engine.vllm_engine import VLLMEngine
        from hfl.engine.base import ChatMessage

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
        """Construye prompt con mensaje system."""
        from hfl.engine.vllm_engine import VLLMEngine
        from hfl.engine.base import ChatMessage

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
        """Construye prompt con historial de assistant."""
        from hfl.engine.vllm_engine import VLLMEngine
        from hfl.engine.base import ChatMessage

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

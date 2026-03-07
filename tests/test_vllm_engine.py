# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the engine/vllm_engine module with true streaming support."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


async def _async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


@pytest.fixture(autouse=True)
def mock_vllm():
    """Mock vLLM modules for testing without GPU."""
    # Clear cached import to force reimport with mocks
    sys.modules.pop("hfl.engine.vllm_engine", None)

    mock_sampling = MagicMock()
    mock_llm = MagicMock()

    # Async engine mocks
    mock_async_engine_instance = MagicMock()
    mock_async_engine_class = MagicMock()
    mock_async_engine_class.from_engine_args = AsyncMock(
        return_value=mock_async_engine_instance
    )
    mock_engine_args = MagicMock()

    # Main vllm module
    vllm_mock = MagicMock(SamplingParams=mock_sampling, LLM=mock_llm)

    # Submodule mocks
    async_engine_module = MagicMock(AsyncLLMEngine=mock_async_engine_class)
    arg_utils_module = MagicMock(AsyncEngineArgs=mock_engine_args)

    with patch.dict(
        sys.modules,
        {
            "vllm": vllm_mock,
            "vllm.engine": MagicMock(),
            "vllm.engine.async_llm_engine": async_engine_module,
            "vllm.engine.arg_utils": arg_utils_module,
        },
    ):
        yield {
            "vllm": vllm_mock,
            "sampling": mock_sampling,
            "llm": mock_llm,
            "async_engine_class": mock_async_engine_class,
            "async_engine": mock_async_engine_instance,
            "engine_args": mock_engine_args,
        }


class TestVLLMEngineInit:
    """Tests for initialization and properties."""

    def test_initialization(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        assert engine._engine is None
        assert engine._model_path == ""
        assert engine._is_async is False

    def test_is_loaded_false_initially(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        assert engine.is_loaded is False

    def test_is_loaded_true_when_engine_set(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._engine = MagicMock()
        assert engine.is_loaded is True

    def test_model_name(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._model_path = "test-model"
        assert engine.model_name == "test-model"


class TestVLLMEngineLoad:
    """Tests for model loading."""

    def test_load_async_engine(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine.load("/path/to/model")

        assert engine.is_loaded
        assert engine._is_async is True
        assert engine._model_path == "/path/to/model"
        mock_vllm["engine_args"].assert_called_once_with(model="/path/to/model")

        engine.unload()

    def test_load_with_kwargs(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine.load("/path/to/model", tensor_parallel_size=2, gpu_memory_utilization=0.9)

        mock_vllm["engine_args"].assert_called_once_with(
            model="/path/to/model",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
        )

        engine.unload()

    def test_load_fallback_to_sync(self, mock_vllm):
        """Falls back to sync LLM when async engine import fails."""
        # Clear module cache again
        sys.modules.pop("hfl.engine.vllm_engine", None)

        # Override submodule mocks to simulate ImportError
        with patch.dict(
            sys.modules,
            {
                "vllm.engine.async_llm_engine": None,  # Causes ImportError
                "vllm.engine.arg_utils": None,
            },
        ):
            from hfl.engine.vllm_engine import VLLMEngine

            engine = VLLMEngine()
            engine.load("/path/to/model")

            assert engine.is_loaded
            assert engine._is_async is False
            mock_vllm["llm"].assert_called_once_with(model="/path/to/model")

    def test_unload(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._engine = MagicMock()
        engine._is_async = True

        engine.unload()

        assert engine._engine is None
        assert engine._is_async is False


class TestVLLMEnginePromptFormat:
    """Tests for prompt format detection."""

    def test_detect_llama3(self, mock_vllm):
        from hfl.engine.prompt_builder import PromptFormat
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        assert engine._detect_prompt_format("meta-llama/Llama-3-8B") == PromptFormat.LLAMA3
        assert engine._detect_prompt_format("llama3-70b") == PromptFormat.LLAMA3

    def test_detect_llama2(self, mock_vllm):
        from hfl.engine.prompt_builder import PromptFormat
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        assert engine._detect_prompt_format("meta-llama/Llama-2-7B") == PromptFormat.LLAMA2
        assert engine._detect_prompt_format("llama2-13b") == PromptFormat.LLAMA2

    def test_detect_vicuna(self, mock_vllm):
        from hfl.engine.prompt_builder import PromptFormat
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        assert engine._detect_prompt_format("lmsys/vicuna-7b") == PromptFormat.VICUNA

    def test_detect_alpaca(self, mock_vllm):
        from hfl.engine.prompt_builder import PromptFormat
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        assert engine._detect_prompt_format("tatsu-lab/alpaca-7b") == PromptFormat.ALPACA

    def test_detect_default_chatml(self, mock_vllm):
        from hfl.engine.prompt_builder import PromptFormat
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        assert engine._detect_prompt_format("some-model") == PromptFormat.CHATML
        assert engine._detect_prompt_format("mistralai/Mistral-7B") == PromptFormat.CHATML


class TestVLLMEngineGenerate:
    """Tests for text generation."""

    def test_generate_not_loaded_raises(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.generate("test")

    def test_generate_sync_mode(self, mock_vllm):
        from hfl.engine.base import GenerationConfig
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = False

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated text", token_ids=[1, 2, 3])]
        engine._engine = MagicMock()
        engine._engine.generate.return_value = [mock_output]

        result = engine.generate("Test prompt", GenerationConfig(max_tokens=100))

        assert result.text == "Generated text"
        assert result.tokens_generated == 3

    def test_generate_async_mode(self, mock_vllm):
        from hfl.engine.base import GenerationConfig
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = True

        mock_completion = MagicMock()
        mock_completion.text = "Async generated"
        mock_completion.token_ids = [1, 2, 3, 4]
        mock_completion.finish_reason = "stop"

        mock_output = MagicMock()
        mock_output.outputs = [mock_completion]

        engine._engine = MagicMock()
        engine._engine.generate = MagicMock(
            return_value=_async_iter([mock_output])
        )
        engine._ensure_loop()

        result = engine.generate("Test", GenerationConfig(max_tokens=50))

        assert result.text == "Async generated"
        assert result.tokens_generated == 4
        assert result.stop_reason == "stop"

        engine.unload()

    def test_generate_async_no_finish_reason(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = True

        mock_completion = MagicMock()
        mock_completion.text = "Response"
        mock_completion.token_ids = [1]
        mock_completion.finish_reason = None

        mock_output = MagicMock()
        mock_output.outputs = [mock_completion]

        engine._engine = MagicMock()
        engine._engine.generate = MagicMock(
            return_value=_async_iter([mock_output])
        )
        engine._ensure_loop()

        result = engine.generate("Test")
        assert result.stop_reason == "stop"

        engine.unload()

    def test_generate_uses_sampling_params(self, mock_vllm):
        from hfl.engine.base import GenerationConfig
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = False

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Ok", token_ids=[1])]
        engine._engine = MagicMock()
        engine._engine.generate.return_value = [mock_output]

        config = GenerationConfig(
            temperature=0.5,
            top_p=0.8,
            top_k=30,
            max_tokens=256,
            repeat_penalty=1.2,
        )
        engine.generate("Test", config)

        # Verify SamplingParams was called with correct args
        mock_vllm["sampling"].assert_called_once_with(
            temperature=0.5,
            top_p=0.8,
            top_k=30,
            max_tokens=256,
            stop=None,
            repetition_penalty=1.2,
        )


class TestVLLMEngineStreaming:
    """Tests for streaming generation."""

    def test_stream_not_loaded_raises(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            list(engine.generate_stream("test"))

    def test_stream_sync_fallback(self, mock_vllm):
        """Sync mode yields complete response in one chunk."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = False

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Full response", token_ids=[1, 2])]
        engine._engine = MagicMock()
        engine._engine.generate.return_value = [mock_output]

        chunks = list(engine.generate_stream("Test"))

        assert len(chunks) == 1
        assert chunks[0] == "Full response"

    def test_stream_async_multiple_tokens(self, mock_vllm):
        """Async mode yields incremental token deltas."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = True

        # Simulate progressive output: "Hello" -> "Hello world" -> "Hello world!"
        outputs = []
        for text, ids in [
            ("Hello", [1]),
            ("Hello world", [1, 2]),
            ("Hello world!", [1, 2, 3]),
        ]:
            mock_completion = MagicMock()
            mock_completion.text = text
            mock_completion.token_ids = ids
            mock_completion.finish_reason = None if text != "Hello world!" else "stop"

            mock_output = MagicMock()
            mock_output.outputs = [mock_completion]
            outputs.append(mock_output)

        engine._engine = MagicMock()
        engine._engine.generate = MagicMock(
            return_value=_async_iter(outputs)
        )
        engine._ensure_loop()

        chunks = list(engine.generate_stream("Test"))

        assert chunks == ["Hello", " world", "!"]

        engine.unload()

    def test_stream_async_single_output(self, mock_vllm):
        """Async mode with single output still works."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = True

        mock_completion = MagicMock()
        mock_completion.text = "Single token"
        mock_completion.token_ids = [1]

        mock_output = MagicMock()
        mock_output.outputs = [mock_completion]

        engine._engine = MagicMock()
        engine._engine.generate = MagicMock(
            return_value=_async_iter([mock_output])
        )
        engine._ensure_loop()

        chunks = list(engine.generate_stream("Test"))
        assert chunks == ["Single token"]

        engine.unload()

    def test_stream_async_error_propagation(self, mock_vllm):
        """Errors in async streaming are propagated to caller."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = True

        async def _error_gen(*args, **kwargs):
            mock_completion = MagicMock()
            mock_completion.text = "partial"
            mock_completion.token_ids = [1]

            mock_output = MagicMock()
            mock_output.outputs = [mock_completion]
            yield mock_output
            raise RuntimeError("GPU error")

        engine._engine = MagicMock()
        engine._engine.generate = MagicMock(
            side_effect=lambda *a, **kw: _error_gen(*a, **kw)
        )
        engine._ensure_loop()

        with pytest.raises(RuntimeError, match="GPU error"):
            list(engine.generate_stream("Test"))

        engine.unload()

    def test_stream_async_empty_deltas_skipped(self, mock_vllm):
        """Empty deltas (duplicate outputs) are not yielded."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = True

        # Same text twice (no new content)
        outputs = []
        for text in ["Hello", "Hello", "Hello world"]:
            mock_completion = MagicMock()
            mock_completion.text = text
            mock_completion.token_ids = [1]
            mock_output = MagicMock()
            mock_output.outputs = [mock_completion]
            outputs.append(mock_output)

        engine._engine = MagicMock()
        engine._engine.generate = MagicMock(
            return_value=_async_iter(outputs)
        )
        engine._ensure_loop()

        chunks = list(engine.generate_stream("Test"))
        # "Hello" once, then " world" (second "Hello" skipped as empty delta)
        assert chunks == ["Hello", " world"]

        engine.unload()


class TestVLLMEngineChat:
    """Tests for chat completion."""

    def test_chat_uses_prompt_builder(self, mock_vllm):
        from hfl.engine.base import ChatMessage
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = False

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Response", token_ids=[1])]
        engine._engine = MagicMock()
        engine._engine.generate.return_value = [mock_output]

        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
        ]
        result = engine.chat(messages)

        assert result.text == "Response"
        # Verify prompt was built with ChatML format (default)
        call_args = engine._engine.generate.call_args
        prompt = call_args[0][0][0]
        assert "<|im_start|>" in prompt

    def test_chat_stream(self, mock_vllm):
        from hfl.engine.base import ChatMessage
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = False

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Stream response", token_ids=[1])]
        engine._engine = MagicMock()
        engine._engine.generate.return_value = [mock_output]

        messages = [ChatMessage(role="user", content="Hi")]
        chunks = list(engine.chat_stream(messages))

        assert len(chunks) == 1
        assert chunks[0] == "Stream response"

    def test_chat_with_llama3_format(self, mock_vllm):
        from hfl.engine.base import ChatMessage
        from hfl.engine.prompt_builder import PromptFormat
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = False
        engine._prompt_format = PromptFormat.LLAMA3

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Response", token_ids=[1])]
        engine._engine = MagicMock()
        engine._engine.generate.return_value = [mock_output]

        messages = [ChatMessage(role="user", content="Hello")]
        engine.chat(messages)

        call_args = engine._engine.generate.call_args
        prompt = call_args[0][0][0]
        assert "<|begin_of_text|>" in prompt

    def test_chat_with_llama2_format(self, mock_vllm):
        from hfl.engine.base import ChatMessage
        from hfl.engine.prompt_builder import PromptFormat
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = False
        engine._prompt_format = PromptFormat.LLAMA2

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Response", token_ids=[1])]
        engine._engine = MagicMock()
        engine._engine.generate.return_value = [mock_output]

        messages = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hello"),
        ]
        engine.chat(messages)

        call_args = engine._engine.generate.call_args
        prompt = call_args[0][0][0]
        assert "[INST]" in prompt
        assert "<<SYS>>" in prompt

    def test_chat_stream_async_mode(self, mock_vllm):
        """Chat stream works with async engine for true streaming."""
        from hfl.engine.base import ChatMessage
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._is_async = True

        outputs = []
        for text in ["Hi", "Hi there", "Hi there!"]:
            mock_completion = MagicMock()
            mock_completion.text = text
            mock_completion.token_ids = [1]
            mock_output = MagicMock()
            mock_output.outputs = [mock_completion]
            outputs.append(mock_output)

        engine._engine = MagicMock()
        engine._engine.generate = MagicMock(
            return_value=_async_iter(outputs)
        )
        engine._ensure_loop()

        messages = [ChatMessage(role="user", content="Hello")]
        chunks = list(engine.chat_stream(messages))

        assert chunks == ["Hi", " there", "!"]

        engine.unload()


class TestVLLMEngineLifecycle:
    """Tests for engine lifecycle management."""

    def test_unload_clears_state(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._engine = MagicMock()
        engine._is_async = True
        engine._model_path = "test"

        engine.unload()

        assert engine._engine is None
        assert engine._is_async is False
        assert engine._loop is None

    def test_unload_without_loop(self, mock_vllm):
        """Unload works even if loop was never started."""
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._engine = MagicMock()

        engine.unload()  # Should not raise

        assert engine._engine is None

    def test_context_manager(self, mock_vllm):
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine._engine = MagicMock()

        with engine:
            assert engine.is_loaded

        assert not engine.is_loaded

    def test_load_sets_prompt_format(self, mock_vllm):
        from hfl.engine.prompt_builder import PromptFormat
        from hfl.engine.vllm_engine import VLLMEngine

        engine = VLLMEngine()
        engine.load("meta-llama/Llama-3-8B")

        assert engine._prompt_format == PromptFormat.LLAMA3

        engine.unload()

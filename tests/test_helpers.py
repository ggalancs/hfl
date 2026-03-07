# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for API helpers."""

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from hfl.api.helpers import (
    StreamingContext,
    ensure_llm_loaded,
    ensure_tts_loaded,
    format_ndjson_chunk,
    options_to_config,
    request_to_config,
    run_async_with_timeout,
    run_with_timeout,
    stream_ollama_chat,
    stream_ollama_generate,
    stream_openai_chat,
    stream_openai_completion,
)
from hfl.api.state import reset_state
from hfl.converter.formats import ModelType
from hfl.engine.base import GenerationConfig


class TestOptionsToConfig:
    """Tests for options_to_config function."""

    def test_empty_options_returns_default(self):
        """Empty options returns default GenerationConfig."""
        config = options_to_config(None)
        assert isinstance(config, GenerationConfig)

    def test_empty_dict_returns_default(self):
        """Empty dict returns default GenerationConfig."""
        config = options_to_config({})
        assert isinstance(config, GenerationConfig)

    def test_temperature_mapping(self):
        """temperature option maps correctly."""
        config = options_to_config({"temperature": 0.5})
        assert config.temperature == 0.5

    def test_top_p_mapping(self):
        """top_p option maps correctly."""
        config = options_to_config({"top_p": 0.9})
        assert config.top_p == 0.9

    def test_top_k_mapping(self):
        """top_k option maps correctly."""
        config = options_to_config({"top_k": 40})
        assert config.top_k == 40

    def test_num_predict_maps_to_max_tokens(self):
        """num_predict maps to max_tokens."""
        config = options_to_config({"num_predict": 100})
        assert config.max_tokens == 100

    def test_repeat_penalty_mapping(self):
        """repeat_penalty option maps correctly."""
        config = options_to_config({"repeat_penalty": 1.2})
        assert config.repeat_penalty == 1.2

    def test_stop_mapping(self):
        """stop option maps correctly."""
        config = options_to_config({"stop": ["END", "STOP"]})
        assert config.stop == ["END", "STOP"]

    def test_seed_mapping(self):
        """seed option maps correctly."""
        config = options_to_config({"seed": 42})
        assert config.seed == 42

    def test_none_values_ignored(self):
        """None values in options are ignored."""
        config = options_to_config(
            {
                "temperature": None,
                "top_p": None,
            }
        )
        # Should not raise and return default
        assert isinstance(config, GenerationConfig)

    def test_multiple_options(self):
        """Multiple options map correctly."""
        config = options_to_config(
            {
                "temperature": 0.7,
                "top_p": 0.95,
                "num_predict": 200,
                "seed": 123,
            }
        )
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.max_tokens == 200
        assert config.seed == 123


class TestRequestToConfig:
    """Tests for request_to_config function."""

    def test_no_args_returns_default(self):
        """No arguments returns default config."""
        config = request_to_config()
        assert isinstance(config, GenerationConfig)

    def test_temperature_param(self):
        """temperature parameter works."""
        config = request_to_config(temperature=0.8)
        assert config.temperature == 0.8

    def test_top_p_param(self):
        """top_p parameter works."""
        config = request_to_config(top_p=0.85)
        assert config.top_p == 0.85

    def test_max_tokens_param(self):
        """max_tokens parameter works."""
        config = request_to_config(max_tokens=150)
        assert config.max_tokens == 150

    def test_seed_param(self):
        """seed parameter works."""
        config = request_to_config(seed=999)
        assert config.seed == 999

    def test_stop_as_string(self):
        """stop as string is normalized to list."""
        config = request_to_config(stop="END")
        assert config.stop == ["END"]

    def test_stop_as_list(self):
        """stop as list is preserved."""
        config = request_to_config(stop=["END", "STOP"])
        assert config.stop == ["END", "STOP"]

    def test_stop_as_tuple(self):
        """stop as tuple is converted to list."""
        config = request_to_config(stop=("END", "STOP"))
        assert config.stop == ["END", "STOP"]

    def test_extra_kwargs_ignored(self):
        """Extra kwargs are ignored."""
        config = request_to_config(
            temperature=0.5,
            unknown_param="value",
            another_param=123,
        )
        assert config.temperature == 0.5

    def test_none_values_ignored(self):
        """None values are not set."""
        config = request_to_config(
            temperature=None,
            top_p=None,
            max_tokens=None,
        )
        assert isinstance(config, GenerationConfig)


class TestStreamingContext:
    """Tests for StreamingContext class."""

    def test_init_generates_request_id(self):
        """StreamingContext generates a request ID."""
        ctx = StreamingContext("test-model")
        assert ctx.request_id.startswith("chatcmpl-")
        assert len(ctx.request_id) > 10

    def test_init_sets_created_timestamp(self):
        """StreamingContext sets created timestamp."""
        ctx = StreamingContext("test-model")
        assert isinstance(ctx.created, int)
        assert ctx.created > 0

    def test_init_sets_model(self):
        """StreamingContext sets model name."""
        ctx = StreamingContext("gpt-4")
        assert ctx.model == "gpt-4"

    def test_init_default_object_type(self):
        """Default object type is chat.completion.chunk."""
        ctx = StreamingContext("test")
        assert ctx.object_type == "chat.completion.chunk"

    def test_init_custom_object_type(self):
        """Custom object type can be specified."""
        ctx = StreamingContext("test", object_type="text_completion")
        assert ctx.object_type == "text_completion"

    def test_format_chunk_with_content(self):
        """format_chunk formats chunk with content."""
        ctx = StreamingContext("test-model")
        chunk = ctx.format_chunk(content="Hello")

        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")

        data = json.loads(chunk[6:-2])  # Remove "data: " and "\n\n"
        assert data["id"] == ctx.request_id
        assert data["model"] == "test-model"
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_format_chunk_with_finish_reason(self):
        """format_chunk formats chunk with finish reason."""
        ctx = StreamingContext("test-model")
        chunk = ctx.format_chunk(finish_reason="stop")

        data = json.loads(chunk[6:-2])
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["delta"] == {}

    def test_format_chunk_with_index(self):
        """format_chunk respects index parameter."""
        ctx = StreamingContext("test-model")
        chunk = ctx.format_chunk(content="test", index=2)

        data = json.loads(chunk[6:-2])
        assert data["choices"][0]["index"] == 2

    def test_format_done(self):
        """format_done returns [DONE] marker."""
        ctx = StreamingContext("test")
        done = ctx.format_done()

        assert done == "data: [DONE]\n\n"


class TestFormatNdjsonChunk:
    """Tests for format_ndjson_chunk function."""

    def test_basic_chunk(self):
        """format_ndjson_chunk creates basic chunk."""
        chunk = format_ndjson_chunk("Hello", "test-model")

        data = json.loads(chunk)
        assert data["model"] == "test-model"
        assert data["response"] == "Hello"
        assert data["done"] is False
        assert "created_at" in data

    def test_done_chunk(self):
        """format_ndjson_chunk creates done chunk."""
        chunk = format_ndjson_chunk("", "test-model", done=True)

        data = json.loads(chunk)
        assert data["done"] is True

    def test_extra_fields(self):
        """format_ndjson_chunk includes extra fields."""
        chunk = format_ndjson_chunk(
            "content",
            "test-model",
            total_tokens=100,
            custom_field="value",
        )

        data = json.loads(chunk)
        assert data["total_tokens"] == 100
        assert data["custom_field"] == "value"

    def test_ends_with_newline(self):
        """format_ndjson_chunk ends with newline."""
        chunk = format_ndjson_chunk("test", "model")
        assert chunk.endswith("\n")


class TestStreamOpenAIChat:
    """Tests for stream_openai_chat function."""

    @pytest.mark.asyncio
    async def test_streams_tokens(self):
        """stream_openai_chat yields formatted tokens."""
        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = ["Hello", " ", "world"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_openai_chat(mock_engine, [], config, "test-model"):
            chunks.append(chunk)

        # Should have 3 content chunks + 1 finish+done chunk (combined)
        assert len(chunks) == 4
        assert "Hello" in chunks[0]
        assert "[DONE]" in chunks[-1]

    @pytest.mark.asyncio
    async def test_ends_with_finish_reason(self):
        """stream_openai_chat ends with finish_reason=stop."""
        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = ["test"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_openai_chat(mock_engine, [], config, "model"):
            chunks.append(chunk)

        # Last chunk contains both finish_reason and [DONE]
        # The finish part is the first "data:" line in the combined chunk
        last_chunk = chunks[-1]
        # Extract the first data: line (finish chunk)
        lines = [line for line in last_chunk.split("\n") if line.startswith("data: {")]
        data = json.loads(lines[0][6:])
        assert data["choices"][0]["finish_reason"] == "stop"


class TestStreamOpenAICompletion:
    """Tests for stream_openai_completion function."""

    @pytest.mark.asyncio
    async def test_streams_tokens(self):
        """stream_openai_completion yields formatted tokens."""
        mock_engine = MagicMock()
        mock_engine.generate_stream.return_value = ["Hello"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_openai_completion(mock_engine, "prompt", config, "model"):
            chunks.append(chunk)

        assert len(chunks) == 2  # 1 content + 1 finish+done (combined)

    @pytest.mark.asyncio
    async def test_uses_text_completion_object_type(self):
        """stream_openai_completion uses text_completion object type."""
        mock_engine = MagicMock()
        mock_engine.generate_stream.return_value = ["test"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_openai_completion(mock_engine, "prompt", config, "model"):
            chunks.append(chunk)

        data = json.loads(chunks[0][6:-2])
        assert data["object"] == "text_completion"


class TestStreamOllamaGenerate:
    """Tests for stream_ollama_generate function."""

    @pytest.mark.asyncio
    async def test_streams_ndjson_tokens(self):
        """stream_ollama_generate yields NDJSON tokens."""
        mock_engine = MagicMock()
        mock_engine.generate_stream.return_value = ["Hello", " ", "world"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_ollama_generate(mock_engine, "prompt", config, "model"):
            chunks.append(chunk)

        # 3 tokens + 1 final
        assert len(chunks) == 4

        # Each should be valid JSON
        for chunk in chunks:
            data = json.loads(chunk)
            assert "model" in data
            assert "done" in data

    @pytest.mark.asyncio
    async def test_final_chunk_is_done(self):
        """stream_ollama_generate final chunk has done=True."""
        mock_engine = MagicMock()
        mock_engine.generate_stream.return_value = ["test"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_ollama_generate(mock_engine, "prompt", config, "model"):
            chunks.append(chunk)

        final = json.loads(chunks[-1])
        assert final["done"] is True


class TestStreamOllamaChat:
    """Tests for stream_ollama_chat function."""

    @pytest.mark.asyncio
    async def test_streams_chat_chunks(self):
        """stream_ollama_chat yields chat-formatted chunks."""
        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = ["Hi", " there"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_ollama_chat(mock_engine, [], config, "model"):
            chunks.append(chunk)

        # 2 tokens + 1 final
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_chunk_has_message_field(self):
        """stream_ollama_chat chunks have message field."""
        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = ["Hello"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_ollama_chat(mock_engine, [], config, "model"):
            chunks.append(chunk)

        data = json.loads(chunks[0])
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_final_has_done_marker(self):
        """stream_ollama_chat final chunk has done=True."""
        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = ["Hello", " ", "world"]
        config = GenerationConfig()

        chunks = []
        async for chunk in stream_ollama_chat(mock_engine, [], config, "model"):
            chunks.append(chunk)

        final = json.loads(chunks[-1])
        assert final["done"] is True
        assert final["message"]["role"] == "assistant"


@dataclass
class MockManifest:
    """Mock model manifest for testing."""

    name: str
    local_path: str = "/mock/path"


class MockEngine:
    """Mock inference engine for testing."""

    def __init__(self, name: str = "test"):
        self.name = name
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str, **kwargs) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False


class MockTTSEngine:
    """Mock TTS engine for testing."""

    def __init__(self, name: str = "test-tts"):
        self.name = name
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, path: str, **kwargs) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False


class TestEnsureLLMLoaded:
    """Tests for ensure_llm_loaded function."""

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
            await ensure_llm_loaded("")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.get_state")
    async def test_fast_path_returns_loaded_model(self, mock_get_state):
        """Fast path returns already loaded model."""
        mock_engine = MockEngine()
        mock_manifest = MockManifest("test-model")

        mock_state = MagicMock()
        mock_state.current_model = mock_manifest
        mock_state.engine = mock_engine
        mock_get_state.return_value = mock_state

        engine, manifest = await ensure_llm_loaded("test-model")

        assert engine is mock_engine
        assert manifest is mock_manifest

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.get_state")
    @patch("hfl.api.helpers.get_registry")
    async def test_model_not_found_raises_404(self, mock_get_registry, mock_get_state):
        """Model not in registry raises HTTPException 404."""
        mock_state = MagicMock()
        mock_state.current_model = None
        mock_get_state.return_value = mock_state

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        with pytest.raises(HTTPException) as exc_info:
            await ensure_llm_loaded("nonexistent-model")

        assert exc_info.value.status_code == 404
        assert "Model not found" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.get_state")
    @patch("hfl.api.helpers.get_registry")
    @patch("hfl.api.helpers.detect_model_type")
    async def test_wrong_model_type_raises_400(
        self,
        mock_detect,
        mock_get_registry,
        mock_get_state,
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
            await ensure_llm_loaded("test-model")

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "MODEL_TYPE_MISMATCH"

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.select_engine")
    @patch("hfl.api.helpers.detect_model_type")
    @patch("hfl.api.helpers.get_registry")
    @patch("hfl.api.helpers.get_state")
    async def test_loads_model_successfully(
        self,
        mock_get_state,
        mock_get_registry,
        mock_detect,
        mock_select,
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

        engine, manifest = await ensure_llm_loaded("new-model")

        assert engine is mock_engine
        assert manifest is mock_manifest
        mock_state.set_llm_engine.assert_called_once_with(mock_engine, mock_manifest)


class TestEnsureTTSLoaded:
    """Tests for ensure_tts_loaded function."""

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
            await ensure_tts_loaded("")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.get_state")
    async def test_fast_path_returns_loaded_tts(self, mock_get_state):
        """Fast path returns already loaded TTS model."""
        mock_engine = MockTTSEngine()
        mock_manifest = MockManifest("test-tts")

        mock_state = MagicMock()
        mock_state.current_tts_model = mock_manifest
        mock_state.tts_engine = mock_engine
        mock_get_state.return_value = mock_state

        engine, manifest = await ensure_tts_loaded("test-tts")

        assert engine is mock_engine
        assert manifest is mock_manifest

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.get_state")
    @patch("hfl.api.helpers.get_registry")
    async def test_model_not_found_raises_404(self, mock_get_registry, mock_get_state):
        """Model not in registry raises HTTPException 404."""
        mock_state = MagicMock()
        mock_state.current_tts_model = None
        mock_get_state.return_value = mock_state

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        with pytest.raises(HTTPException) as exc_info:
            await ensure_tts_loaded("nonexistent-tts")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.get_state")
    @patch("hfl.api.helpers.get_registry")
    @patch("hfl.api.helpers.detect_model_type")
    async def test_wrong_model_type_raises_400(
        self,
        mock_detect,
        mock_get_registry,
        mock_get_state,
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
            await ensure_tts_loaded("test-model")

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "MODEL_TYPE_MISMATCH"
        assert exc_info.value.detail["expected"] == "tts"

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.select_tts_engine")
    @patch("hfl.api.helpers.detect_model_type")
    @patch("hfl.api.helpers.get_registry")
    @patch("hfl.api.helpers.get_state")
    async def test_loads_tts_successfully(
        self,
        mock_get_state,
        mock_get_registry,
        mock_detect,
        mock_select,
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

        engine, manifest = await ensure_tts_loaded("new-tts")

        assert engine is mock_engine
        assert manifest is mock_manifest
        mock_state.set_tts_engine.assert_called_once_with(mock_engine, mock_manifest)


class TestRunWithTimeout:
    """Tests for run_with_timeout function."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Successful function execution returns result."""

        def simple_func(x, y):
            return x + y

        result = await run_with_timeout(simple_func, 2, 3, operation="test_add")
        assert result == 5

    @pytest.mark.asyncio
    async def test_timeout_raises_http_exception(self):
        """Timeout raises HTTPException 504."""
        import time

        def slow_func():
            time.sleep(5)
            return "done"

        with pytest.raises(HTTPException) as exc_info:
            await run_with_timeout(slow_func, timeout=0.1, operation="slow_op")

        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["code"] == "TIMEOUT"
        assert exc_info.value.detail["operation"] == "slow_op"
        assert exc_info.value.detail["timeout_seconds"] == 0.1

    @pytest.mark.asyncio
    async def test_kwargs_passed_correctly(self):
        """Keyword arguments are passed to function."""

        def func_with_kwargs(a, b=10, c=20):
            return a + b + c

        result = await run_with_timeout(func_with_kwargs, 1, b=5, c=10, operation="test")
        assert result == 16

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.config")
    async def test_uses_config_timeout_by_default(self, mock_config):
        """Uses config.generation_timeout when timeout not specified."""
        mock_config.generation_timeout = 300.0

        def fast_func():
            return "done"

        result = await run_with_timeout(fast_func, operation="test")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_explicit_timeout_overrides_config(self):
        """Explicit timeout parameter overrides config default."""
        import time

        def medium_func():
            time.sleep(0.2)
            return "done"

        # Should succeed with longer timeout
        result = await run_with_timeout(medium_func, timeout=1.0, operation="test")
        assert result == "done"

        # Should fail with shorter timeout
        with pytest.raises(HTTPException) as exc_info:
            await run_with_timeout(medium_func, timeout=0.05, operation="test")
        assert exc_info.value.status_code == 504


class TestRunAsyncWithTimeout:
    """Tests for run_async_with_timeout function."""

    @pytest.mark.asyncio
    async def test_successful_async_execution(self):
        """Successful async execution returns result."""
        import asyncio

        async def async_func():
            await asyncio.sleep(0.01)
            return "async_done"

        result = await run_async_with_timeout(async_func(), operation="test_async")
        assert result == "async_done"

    @pytest.mark.asyncio
    async def test_async_timeout_raises_http_exception(self):
        """Async timeout raises HTTPException 504."""
        import asyncio

        async def slow_async():
            await asyncio.sleep(5)
            return "done"

        with pytest.raises(HTTPException) as exc_info:
            await run_async_with_timeout(slow_async(), timeout=0.1, operation="slow_async")

        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["code"] == "TIMEOUT"
        assert exc_info.value.detail["operation"] == "slow_async"

    @pytest.mark.asyncio
    @patch("hfl.api.helpers.config")
    async def test_async_uses_config_timeout_by_default(self, mock_config):
        """Uses config.generation_timeout for async when timeout not specified."""
        mock_config.generation_timeout = 300.0

        async def fast_async():
            return "fast"

        result = await run_async_with_timeout(fast_async(), operation="test")
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_async_explicit_timeout_overrides_config(self):
        """Explicit timeout parameter overrides config for async."""
        import asyncio

        async def medium_async():
            await asyncio.sleep(0.2)
            return "done"

        # Should succeed with longer timeout
        result = await run_async_with_timeout(medium_async(), timeout=1.0, operation="test")
        assert result == "done"

        # Should fail with shorter timeout
        with pytest.raises(HTTPException) as exc_info:
            await run_async_with_timeout(medium_async(), timeout=0.05, operation="test")
        assert exc_info.value.status_code == 504

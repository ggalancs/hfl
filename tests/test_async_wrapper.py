# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for async engine wrapper."""


import pytest

from hfl.engine.async_wrapper import AsyncEngineWrapper
from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult


class MockEngine:
    """Mock inference engine for testing."""

    def __init__(self):
        self._loaded = False
        self._model_name = ""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self, model_path: str, **kwargs) -> None:
        self._loaded = True
        self._model_name = model_path

    def unload(self) -> None:
        self._loaded = False
        self._model_name = ""

    def generate(self, prompt: str, config=None) -> GenerationResult:
        return GenerationResult(
            text=f"Response to: {prompt}",
            tokens_generated=10,
            tokens_prompt=5,
        )

    def generate_stream(self, prompt: str, config=None):
        for word in ["Hello", " ", "world", "!"]:
            yield word

    def chat(self, messages: list, config=None) -> GenerationResult:
        return GenerationResult(
            text="Chat response",
            tokens_generated=8,
            tokens_prompt=12,
        )

    def chat_stream(self, messages: list, config=None):
        for word in ["Hi", " ", "there"]:
            yield word


class TestAsyncEngineWrapper:
    """Tests for AsyncEngineWrapper class."""

    @pytest.fixture
    def engine(self):
        """Create mock engine."""
        return MockEngine()

    @pytest.fixture
    def wrapper(self, engine):
        """Create wrapper with mock engine."""
        return AsyncEngineWrapper(engine)

    def test_engine_property(self, wrapper, engine):
        """engine property returns wrapped engine."""
        assert wrapper.engine is engine

    def test_is_loaded_property(self, wrapper, engine):
        """is_loaded delegates to wrapped engine."""
        assert wrapper.is_loaded is False
        engine._loaded = True
        assert wrapper.is_loaded is True

    def test_model_name_property(self, wrapper, engine):
        """model_name delegates to wrapped engine."""
        engine._model_name = "test-model"
        assert wrapper.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_load(self, wrapper, engine):
        """load() runs in thread pool."""
        await wrapper.load("/path/to/model")

        assert engine.is_loaded is True
        assert engine.model_name == "/path/to/model"

    @pytest.mark.asyncio
    async def test_unload(self, wrapper, engine):
        """unload() runs in thread pool."""
        engine._loaded = True

        await wrapper.unload()

        assert engine.is_loaded is False

    @pytest.mark.asyncio
    async def test_generate(self, wrapper):
        """generate() returns result from thread pool."""
        result = await wrapper.generate("Hello world")

        assert isinstance(result, GenerationResult)
        assert "Hello world" in result.text
        assert result.tokens_generated == 10

    @pytest.mark.asyncio
    async def test_generate_with_config(self, wrapper):
        """generate() passes config to engine."""
        config = GenerationConfig(temperature=0.5)
        result = await wrapper.generate("Test", config)

        assert isinstance(result, GenerationResult)

    @pytest.mark.asyncio
    async def test_chat(self, wrapper):
        """chat() returns result from thread pool."""
        messages = [ChatMessage(role="user", content="Hello")]
        result = await wrapper.chat(messages)

        assert isinstance(result, GenerationResult)
        assert result.text == "Chat response"

    @pytest.mark.asyncio
    async def test_generate_stream(self, wrapper):
        """generate_stream() yields tokens asynchronously."""
        tokens = []
        async for token in wrapper.generate_stream("Hello"):
            tokens.append(token)

        assert tokens == ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    async def test_chat_stream(self, wrapper):
        """chat_stream() yields tokens asynchronously."""
        messages = [ChatMessage(role="user", content="Hi")]
        tokens = []
        async for token in wrapper.chat_stream(messages):
            tokens.append(token)

        assert tokens == ["Hi", " ", "there"]

    @pytest.mark.asyncio
    async def test_generate_stream_propagates_errors(self):
        """generate_stream() propagates errors from producer."""
        engine = MockEngine()

        def error_generator(prompt, config=None):
            yield "first"
            raise ValueError("Test error")

        engine.generate_stream = error_generator
        wrapper = AsyncEngineWrapper(engine)

        tokens = []
        with pytest.raises(ValueError, match="Test error"):
            async for token in wrapper.generate_stream("Hello"):
                tokens.append(token)

        assert tokens == ["first"]

    @pytest.mark.asyncio
    async def test_chat_stream_propagates_errors(self):
        """chat_stream() propagates errors from producer."""
        engine = MockEngine()

        def error_generator(messages, config=None):
            yield "first"
            raise RuntimeError("Chat error")

        engine.chat_stream = error_generator
        wrapper = AsyncEngineWrapper(engine)

        tokens = []
        with pytest.raises(RuntimeError, match="Chat error"):
            async for token in wrapper.chat_stream([]):
                tokens.append(token)

        assert tokens == ["first"]

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Empty generator yields nothing."""
        engine = MockEngine()
        engine.generate_stream = lambda p, c=None: iter([])
        wrapper = AsyncEngineWrapper(engine)

        tokens = []
        async for token in wrapper.generate_stream("Hello"):
            tokens.append(token)

        assert tokens == []

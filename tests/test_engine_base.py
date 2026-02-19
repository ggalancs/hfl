# SPDX-License-Identifier: HRUL-1.0
"""Tests for the base inference engine interface."""

import pytest

from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = ChatMessage(role="system", content="You are a helpful assistant")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant"


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.repeat_penalty == 1.1
        assert config.stop is None
        assert config.seed == -1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.2,
            stop=["END"],
        )
        assert config.max_tokens == 1024
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 50
        assert config.repeat_penalty == 1.2
        assert config.stop == ["END"]

    def test_zero_temperature(self):
        """Test that zero temperature is allowed (deterministic)."""
        config = GenerationConfig(temperature=0.0)
        assert config.temperature == 0.0


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_create_result(self):
        """Test creating a generation result."""
        result = GenerationResult(
            text="Generated text",
            tokens_generated=10,
            tokens_prompt=5,
            tokens_per_second=50.0,
        )
        assert result.text == "Generated text"
        assert result.tokens_generated == 10
        assert result.tokens_prompt == 5
        assert result.tokens_per_second == 50.0

    def test_result_with_empty_text(self):
        """Test result with empty text."""
        result = GenerationResult(
            text="",
            tokens_generated=0,
            tokens_prompt=5,
            tokens_per_second=0.0,
        )
        assert result.text == ""
        assert result.tokens_generated == 0


class TestInferenceEngineInterface:
    """Tests for the InferenceEngine abstract interface."""

    def test_is_abstract_class(self):
        """Test that InferenceEngine is abstract."""
        with pytest.raises(TypeError):
            InferenceEngine()

    def test_required_methods(self):
        """Test that required methods are defined."""
        assert hasattr(InferenceEngine, "load")
        assert hasattr(InferenceEngine, "unload")
        assert hasattr(InferenceEngine, "generate")
        assert hasattr(InferenceEngine, "generate_stream")
        assert hasattr(InferenceEngine, "chat")
        assert hasattr(InferenceEngine, "chat_stream")

    def test_required_properties(self):
        """Test that required properties are defined."""
        assert hasattr(InferenceEngine, "model_name")
        assert hasattr(InferenceEngine, "is_loaded")


class ConcreteEngine(InferenceEngine):
    """Concrete implementation for testing."""

    def __init__(self):
        self._loaded = False
        self._name = ""

    def load(self, model_path, **kwargs):
        self._loaded = True
        self._name = str(model_path)

    def unload(self):
        self._loaded = False
        self._name = ""

    def generate(self, prompt, config=None):
        return GenerationResult(
            text=f"Response to: {prompt}",
            tokens_generated=10,
            tokens_prompt=5,
            tokens_per_second=100.0,
        )

    def generate_stream(self, prompt, config=None):
        yield "Hello "
        yield "world"

    def chat(self, messages, config=None):
        return GenerationResult(
            text="Chat response",
            tokens_generated=5,
            tokens_prompt=3,
            tokens_per_second=50.0,
        )

    def chat_stream(self, messages, config=None):
        yield "Hi"
        yield "!"

    @property
    def model_name(self):
        return self._name

    @property
    def is_loaded(self):
        return self._loaded


class TestConcreteEngineImplementation:
    """Tests for a concrete engine implementation."""

    def test_load_and_unload(self):
        """Test loading and unloading a model."""
        engine = ConcreteEngine()
        assert not engine.is_loaded

        engine.load("/path/to/model")
        assert engine.is_loaded
        assert engine.model_name == "/path/to/model"

        engine.unload()
        assert not engine.is_loaded

    def test_generate(self):
        """Test text generation."""
        engine = ConcreteEngine()
        engine.load("/path/to/model")

        result = engine.generate("Hello")

        assert isinstance(result, GenerationResult)
        assert "Hello" in result.text

    def test_generate_stream(self):
        """Test streaming generation."""
        engine = ConcreteEngine()
        engine.load("/path/to/model")

        tokens = list(engine.generate_stream("Hello"))

        assert tokens == ["Hello ", "world"]

    def test_chat(self):
        """Test chat completion."""
        engine = ConcreteEngine()
        engine.load("/path/to/model")

        messages = [ChatMessage(role="user", content="Hi")]
        result = engine.chat(messages)

        assert isinstance(result, GenerationResult)
        assert result.text == "Chat response"

    def test_chat_stream(self):
        """Test streaming chat completion."""
        engine = ConcreteEngine()
        engine.load("/path/to/model")

        messages = [ChatMessage(role="user", content="Hi")]
        tokens = list(engine.chat_stream(messages))

        assert tokens == ["Hi", "!"]

    def test_generate_with_config(self):
        """Test generation with custom config."""
        engine = ConcreteEngine()
        engine.load("/path/to/model")

        config = GenerationConfig(max_tokens=100, temperature=0.5)
        result = engine.generate("Hello", config=config)

        assert isinstance(result, GenerationResult)

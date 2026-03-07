# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for engine base classes."""

from typing import Iterator

import pytest

from hfl.engine.base import (
    AudioEngine,
    AudioResult,
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
    TTSConfig,
)


class ConcreteInferenceEngine(InferenceEngine):
    """Concrete implementation of InferenceEngine for testing."""

    def __init__(self):
        self._loaded = False
        self._model_name = ""
        self.unload_called = False

    def load(self, model_path: str, **kwargs) -> None:
        self._loaded = True
        self._model_name = model_path

    def unload(self) -> None:
        self._loaded = False
        self.unload_called = True

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        return GenerationResult(text="generated", tokens_generated=10)

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        yield "token"

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            text="chat response", tokens_generated=5,
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        yield "chat"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class ConcreteAudioEngine(AudioEngine):
    """Concrete implementation of AudioEngine for testing."""

    def __init__(self):
        self._loaded = False
        self._model_name = ""
        self.unload_called = False

    def load(self, model_path: str, **kwargs) -> None:
        self._loaded = True
        self._model_name = model_path

    def unload(self) -> None:
        self._loaded = False
        self.unload_called = True

    def synthesize(self, text: str, config: TTSConfig | None = None) -> AudioResult:
        return AudioResult(
            audio=b"audio data",
            sample_rate=22050,
            duration=1.0,
            format="wav",
        )

    def synthesize_stream(self, text: str, config: TTSConfig | None = None) -> Iterator[bytes]:
        yield b"chunk"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_create_chat_message(self):
        """ChatMessage can be created with role and content."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_system_message(self):
        """ChatMessage supports system role."""
        msg = ChatMessage(role="system", content="You are helpful")
        assert msg.role == "system"

    def test_assistant_message(self):
        """ChatMessage supports assistant role."""
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_values(self):
        """GenerationConfig has sensible defaults."""
        config = GenerationConfig()
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.max_tokens == 2048
        assert config.stop is None
        assert config.repeat_penalty == 1.1
        assert config.seed == -1

    def test_custom_values(self):
        """GenerationConfig accepts custom values."""
        config = GenerationConfig(
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
            stop=["END"],
            repeat_penalty=1.2,
            seed=42,
        )
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 50
        assert config.max_tokens == 1024
        assert config.stop == ["END"]
        assert config.repeat_penalty == 1.2
        assert config.seed == 42


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_default_values(self):
        """GenerationResult has default values."""
        result = GenerationResult(text="output")
        assert result.text == "output"
        assert result.tokens_generated == 0
        assert result.tokens_prompt == 0
        assert result.tokens_per_second == 0.0
        assert result.stop_reason == "stop"

    def test_custom_values(self):
        """GenerationResult accepts custom values."""
        result = GenerationResult(
            text="output",
            tokens_generated=100,
            tokens_prompt=50,
            tokens_per_second=25.5,
            stop_reason="length",
        )
        assert result.tokens_generated == 100
        assert result.tokens_prompt == 50
        assert result.tokens_per_second == 25.5
        assert result.stop_reason == "length"


class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_default_values(self):
        """TTSConfig has sensible defaults."""
        config = TTSConfig()
        assert config.voice == "default"
        assert config.speed == 1.0
        assert config.language == "en"
        assert config.sample_rate == 22050
        assert config.format == "wav"

    def test_custom_values(self):
        """TTSConfig accepts custom values."""
        config = TTSConfig(
            voice="alloy",
            speed=1.5,
            language="es",
            sample_rate=44100,
            format="mp3",
        )
        assert config.voice == "alloy"
        assert config.speed == 1.5
        assert config.language == "es"
        assert config.sample_rate == 44100
        assert config.format == "mp3"


class TestAudioResult:
    """Tests for AudioResult dataclass."""

    def test_required_fields(self):
        """AudioResult requires audio, sample_rate, duration, format."""
        result = AudioResult(
            audio=b"audio data",
            sample_rate=22050,
            duration=2.5,
            format="wav",
        )
        assert result.audio == b"audio data"
        assert result.sample_rate == 22050
        assert result.duration == 2.5
        assert result.format == "wav"

    def test_metadata_default(self):
        """AudioResult metadata defaults to empty dict."""
        result = AudioResult(
            audio=b"",
            sample_rate=22050,
            duration=0,
            format="wav",
        )
        assert result.metadata == {}

    def test_custom_metadata(self):
        """AudioResult accepts custom metadata."""
        result = AudioResult(
            audio=b"",
            sample_rate=22050,
            duration=1.0,
            format="wav",
            metadata={"voice": "alloy", "model": "tts-1"},
        )
        assert result.metadata["voice"] == "alloy"


class TestInferenceEngineContextManager:
    """Tests for InferenceEngine context manager support."""

    def test_enter_returns_self(self):
        """__enter__ returns the engine instance."""
        engine = ConcreteInferenceEngine()
        result = engine.__enter__()
        assert result is engine

    def test_exit_unloads_model(self):
        """__exit__ unloads model when loaded."""
        engine = ConcreteInferenceEngine()
        engine.load("/model")
        assert engine.is_loaded

        engine.__exit__(None, None, None)

        assert engine.unload_called

    def test_exit_does_not_unload_when_not_loaded(self):
        """__exit__ does nothing when model not loaded."""
        engine = ConcreteInferenceEngine()
        assert not engine.is_loaded

        engine.__exit__(None, None, None)

        assert not engine.unload_called

    def test_context_manager_usage(self):
        """InferenceEngine works as context manager."""
        with ConcreteInferenceEngine() as engine:
            engine.load("/model")
            assert engine.is_loaded

        assert engine.unload_called

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self):
        """__aenter__ returns the engine instance."""
        engine = ConcreteInferenceEngine()
        result = await engine.__aenter__()
        assert result is engine

    @pytest.mark.asyncio
    async def test_aexit_unloads_model(self):
        """__aexit__ unloads model when loaded."""
        engine = ConcreteInferenceEngine()
        engine.load("/model")
        assert engine.is_loaded

        await engine.__aexit__(None, None, None)

        assert engine.unload_called

    @pytest.mark.asyncio
    async def test_aexit_does_not_unload_when_not_loaded(self):
        """__aexit__ does nothing when model not loaded."""
        engine = ConcreteInferenceEngine()
        assert not engine.is_loaded

        await engine.__aexit__(None, None, None)

        assert not engine.unload_called


class TestAudioEngineContextManager:
    """Tests for AudioEngine context manager support."""

    def test_enter_returns_self(self):
        """__enter__ returns the engine instance."""
        engine = ConcreteAudioEngine()
        result = engine.__enter__()
        assert result is engine

    def test_exit_unloads_model(self):
        """__exit__ unloads model when loaded."""
        engine = ConcreteAudioEngine()
        engine.load("/model")
        assert engine.is_loaded

        engine.__exit__(None, None, None)

        assert engine.unload_called

    def test_exit_does_not_unload_when_not_loaded(self):
        """__exit__ does nothing when model not loaded."""
        engine = ConcreteAudioEngine()
        assert not engine.is_loaded

        engine.__exit__(None, None, None)

        assert not engine.unload_called

    def test_context_manager_usage(self):
        """AudioEngine works as context manager."""
        with ConcreteAudioEngine() as engine:
            engine.load("/model")
            assert engine.is_loaded

        assert engine.unload_called

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self):
        """__aenter__ returns the engine instance."""
        engine = ConcreteAudioEngine()
        result = await engine.__aenter__()
        assert result is engine

    @pytest.mark.asyncio
    async def test_aexit_unloads_model(self):
        """__aexit__ unloads model when loaded."""
        engine = ConcreteAudioEngine()
        engine.load("/model")
        assert engine.is_loaded

        await engine.__aexit__(None, None, None)

        assert engine.unload_called


class TestAudioEngineDefaults:
    """Tests for AudioEngine default property implementations."""

    def test_supported_voices_default(self):
        """supported_voices returns ['default'] by default."""
        engine = ConcreteAudioEngine()
        assert engine.supported_voices == ["default"]

    def test_supported_languages_default(self):
        """supported_languages returns ['en'] by default."""
        engine = ConcreteAudioEngine()
        assert engine.supported_languages == ["en"]


class TestInferenceEngineAbstractMethods:
    """Tests for InferenceEngine abstract method requirements."""

    def test_concrete_engine_methods(self):
        """Concrete engine implements all abstract methods."""
        engine = ConcreteInferenceEngine()

        # Test all methods work
        engine.load("/model")
        assert engine.is_loaded
        assert engine.model_name == "/model"

        result = engine.generate("prompt")
        assert isinstance(result, GenerationResult)

        tokens = list(engine.generate_stream("prompt"))
        assert tokens == ["token"]

        chat_result = engine.chat([ChatMessage(role="user", content="Hi")])
        assert isinstance(chat_result, GenerationResult)

        chat_tokens = list(engine.chat_stream([]))
        assert chat_tokens == ["chat"]

        engine.unload()
        assert not engine.is_loaded


class TestAudioEngineAbstractMethods:
    """Tests for AudioEngine abstract method requirements."""

    def test_concrete_engine_methods(self):
        """Concrete engine implements all abstract methods."""
        engine = ConcreteAudioEngine()

        engine.load("/tts-model")
        assert engine.is_loaded
        assert engine.model_name == "/tts-model"

        result = engine.synthesize("Hello")
        assert isinstance(result, AudioResult)
        assert result.audio == b"audio data"

        chunks = list(engine.synthesize_stream("Hello"))
        assert chunks == [b"chunk"]

        engine.unload()
        assert not engine.is_loaded

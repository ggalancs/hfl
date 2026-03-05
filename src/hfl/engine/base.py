# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Abstract interface for inference engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

# =============================================================================
# LLM (Text Generation) Types
# =============================================================================


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stop: list[str] | None = None
    repeat_penalty: float = 1.1
    seed: int = -1


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int = 0
    tokens_prompt: int = 0
    tokens_per_second: float = 0.0
    stop_reason: str = "stop"


class InferenceEngine(ABC):
    """Interface that all backends must implement."""

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Loads the model into memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Releases the model from memory."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generates text synchronously."""
        ...

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Generates text token by token (streaming)."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Synchronous chat with message format."""
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Streaming chat token by token."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the loaded model."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is in memory."""
        ...


# =============================================================================
# TTS (Text-to-Speech) Types
# =============================================================================


@dataclass
class TTSConfig:
    """Configuration for text-to-speech synthesis."""

    voice: str = "default"
    speed: float = 1.0
    language: str = "en"
    sample_rate: int = 22050
    format: str = "wav"  # wav, mp3, ogg


@dataclass
class AudioResult:
    """Result of audio synthesis."""

    audio: bytes
    sample_rate: int
    duration: float
    format: str
    metadata: dict = field(default_factory=dict)


class AudioEngine(ABC):
    """Abstract base class for audio synthesis engines (TTS)."""

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Loads the TTS model into memory.

        Args:
            model_path: Path to the model directory or file
            **kwargs: Backend-specific options (device, dtype, etc.)
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Releases the model from memory."""
        ...

    @abstractmethod
    def synthesize(self, text: str, config: TTSConfig | None = None) -> AudioResult:
        """Synthesizes text to audio.

        Args:
            text: Text to synthesize
            config: TTS configuration (voice, speed, language, etc.)

        Returns:
            AudioResult with audio bytes and metadata
        """
        ...

    @abstractmethod
    def synthesize_stream(
        self, text: str, config: TTSConfig | None = None
    ) -> Iterator[bytes]:
        """Synthesizes text to audio in streaming chunks.

        Args:
            text: Text to synthesize
            config: TTS configuration

        Yields:
            Audio data chunks (raw PCM or encoded)
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is in memory."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the loaded model."""
        ...

    @property
    def supported_voices(self) -> list[str]:
        """List of supported voice identifiers."""
        return ["default"]

    @property
    def supported_languages(self) -> list[str]:
        """List of supported language codes."""
        return ["en"]

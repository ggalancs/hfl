# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Abstract interface for inference engines."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from types import TracebackType

# =============================================================================
# LLM (Text Generation) Types
# =============================================================================


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    # Tool-calling extensions (only populated when relevant):
    # - ``tool_calls``: assistant turn requesting one or more function calls.
    #   Each entry is the canonical Ollama shape:
    #   ``{"function": {"name": str, "arguments": dict}}``.
    # - ``name``: for role=tool, the function name whose result this carries.
    # - ``tool_call_id``: optional id linking a tool result to a prior call.
    tool_calls: list[dict] | None = None
    name: str | None = None
    tool_call_id: str | None = None


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stop: list[str] | None = None
    repeat_penalty: float = 1.1
    seed: int = -1
    # Structured-output constraint (OLLAMA_PARITY_PLAN P0-5).
    # Backends that support grammar-based constrained decoding
    # (llama-cpp-python GBNF, vLLM GuidedDecodingParams, Transformers
    # + outlines) honour this to force the model's output to conform
    # to a JSON schema or a raw GBNF grammar. ``None`` means
    # unconstrained generation.
    #
    # - ``"json"`` → free-form JSON object (value is the literal
    #   string; backends compile it to their "any JSON" grammar).
    # - ``dict`` → JSON Schema (validated at the route boundary).
    # - ``str`` starting with ``"GBNF:"`` → raw GBNF grammar body
    #   (advanced users; bypasses schema validation).
    response_format: str | dict | None = None


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int = 0
    tokens_prompt: int = 0
    tokens_per_second: float = 0.0
    stop_reason: str = "stop"
    # Populated when the engine (or a downstream parser) produced structured
    # tool calls. Shape: list of ``{"function": {"name", "arguments": dict}}``.
    tool_calls: list[dict] | None = None


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
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        """Synchronous chat with message format.

        Args:
            messages: Chat history, possibly including ``role=tool`` results
                and prior assistant ``tool_calls``.
            config: Sampling configuration.
            tools: Optional list of OpenAI/Ollama-shaped tool definitions
                (``[{"type": "function", "function": {...}}, ...]``). If
                provided and supported by the backend, the model's native
                chat template is applied with tool awareness so that
                structured tool calls can be emitted.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        """Streaming chat token by token.

        Same semantics as :meth:`chat` for the ``tools`` argument.
        """
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

    # Context manager support for automatic resource cleanup
    def __enter__(self) -> "InferenceEngine":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit context manager - automatically unload model."""
        if self.is_loaded:
            self.unload()

    async def __aenter__(self) -> "InferenceEngine":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit async context manager - automatically unload model."""
        if self.is_loaded:
            await asyncio.to_thread(self.unload)


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
    def synthesize_stream(self, text: str, config: TTSConfig | None = None) -> Iterator[bytes]:
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

    # Context manager support for automatic resource cleanup
    def __enter__(self) -> "AudioEngine":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit context manager - automatically unload model."""
        if self.is_loaded:
            self.unload()

    async def __aenter__(self) -> "AudioEngine":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit async context manager - automatically unload model."""
        if self.is_loaded:
            await asyncio.to_thread(self.unload)

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Abstract interface for inference engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


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

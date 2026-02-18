# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Interfaz abstracta para motores de inferencia."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class ChatMessage:
    role: str       # "system", "user", "assistant"
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
    """Interfaz que deben implementar todos los backends."""

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Carga el modelo en memoria."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Libera el modelo de memoria."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Genera texto de forma síncrona."""
        ...

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Genera texto token a token (streaming)."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Chat síncrono con formato de mensajes."""
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Chat streaming token a token."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre del modelo cargado."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Si hay un modelo en memoria."""
        ...

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A minimal HFL engine plugin: it "loads" any model and answers with the
last message it was sent. Copy it to write a real engine: implement the
same methods over your runtime and register the class under the
``hfl.engines`` entry point (see pyproject.toml)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult, InferenceEngine


class EchoEngine(InferenceEngine):
    def __init__(self) -> None:
        self._model: str | None = None

    def load(self, model_path: str, **kwargs) -> None:
        self._model = Path(model_path).name

    def unload(self) -> None:
        self._model = None

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        text = f"echo: {prompt.strip().splitlines()[-1] if prompt.strip() else ''}"
        return GenerationResult(text=text, tokens_generated=len(text.split()), tokens_prompt=0)

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        yield from self.generate(prompt, config).text.split(" ")

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        return self.generate(messages[-1].content if messages else "", config)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        yield from self.generate_stream(messages[-1].content if messages else "", config)

    @property
    def model_name(self) -> str:
        return self._model or ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

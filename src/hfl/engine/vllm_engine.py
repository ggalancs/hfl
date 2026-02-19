# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
vLLM inference engine for production.

vLLM is a high-performance inference engine optimized for NVIDIA GPUs.
Supports PagedAttention, continuous batching, and other optimizations.

WARNING: EXPERIMENTAL
    This module is a basic/placeholder implementation. It lacks:
    - True streaming support (yields complete response at once)
    - Proper chat template handling (uses simple prompt building)
    - Advanced error handling
    - Full configuration options

    For production use, consider using the llama.cpp or Transformers backends
    until this implementation is completed.
"""

from vllm import LLM, SamplingParams

from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)


class VLLMEngine(InferenceEngine):
    """vLLM-based inference engine."""

    def __init__(self):
        self._model = None
        self._model_path = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_path

    def load(self, model_path: str, **kwargs) -> None:
        """Loads a model with vLLM."""
        self._model_path = model_path
        self._model = LLM(model=model_path, **kwargs)

    def unload(self) -> None:
        """Unloads the model."""
        self._model = None

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        """Generates text."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        cfg = config or GenerationConfig()
        sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
        )

        outputs = self._model.generate([prompt], sampling_params)
        output = outputs[0]

        return GenerationResult(
            text=output.outputs[0].text,
            tokens_generated=len(output.outputs[0].token_ids),
            stop_reason="stop",
        )

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None):
        """vLLM streaming not supported in this basic implementation."""
        result = self.generate(prompt, config)
        yield result.text

    def chat(
        self, messages: list[ChatMessage], config: GenerationConfig | None = None
    ) -> GenerationResult:
        """Chat completion."""
        prompt = self._build_prompt(messages)
        return self.generate(prompt, config)

    def chat_stream(self, messages: list[ChatMessage], config: GenerationConfig | None = None):
        """Streaming chat completion."""
        result = self.chat(messages, config)
        yield result.text

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        """Builds a simple prompt from messages."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

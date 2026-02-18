# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 hfl Contributors
"""
Motor de inferencia vLLM para producción.

vLLM es un motor de inferencia de alto rendimiento optimizado para GPUs NVIDIA.
Soporta PagedAttention, batching continuo, y otras optimizaciones.

Este módulo es un placeholder para implementación futura.
"""

from vllm import LLM, SamplingParams

from hfl.engine.base import (
    InferenceEngine, ChatMessage, GenerationConfig, GenerationResult,
)


class VLLMEngine(InferenceEngine):
    """Motor de inferencia basado en vLLM."""

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
        """Carga un modelo con vLLM."""
        self._model_path = model_path
        self._model = LLM(model=model_path, **kwargs)

    def unload(self) -> None:
        """Descarga el modelo."""
        self._model = None

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        """Genera texto."""
        if not self._model:
            raise RuntimeError("Modelo no cargado")

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
        """vLLM streaming no soportado en esta implementación básica."""
        result = self.generate(prompt, config)
        yield result.text

    def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> GenerationResult:
        """Chat completion."""
        prompt = self._build_prompt(messages)
        return self.generate(prompt, config)

    def chat_stream(self, messages: list[ChatMessage], config: GenerationConfig | None = None):
        """Chat completion en streaming."""
        result = self.chat(messages, config)
        yield result.text

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        """Construye un prompt simple a partir de mensajes."""
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

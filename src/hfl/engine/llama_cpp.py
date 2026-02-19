# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Inference backend based on llama-cpp-python.

This is the main backend for GGUF models.
Supports CPU, CUDA, Metal, and Vulkan.
"""

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from llama_cpp import Llama


@contextmanager
def _suppress_stderr():
    """Temporarily suppresses stderr (to silence Metal/CUDA logs)."""
    # Save the original descriptor
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        # Redirect stderr to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restore stderr
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


@contextmanager
def _nullcontext():
    """Context manager that does nothing (for when verbose=True)."""
    yield


from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)


class LlamaCppEngine(InferenceEngine):
    """llama.cpp inference engine."""

    def __init__(self):
        self._model: Llama | None = None
        self._model_path: str = ""

    def load(self, model_path: str, **kwargs) -> None:
        """
        Loads a GGUF model.

        Args:
            model_path: Path to the .gguf file
            **kwargs: Additional parameters:
                n_ctx: Context size (default 4096)
                n_gpu_layers: GPU layers (-1 = all)
                n_threads: CPU threads (0 = auto)
                verbose: Show llama.cpp logs
                flash_attn: Use Flash Attention (default True)
                chat_format: Chat format (auto-detected)
        """
        verbose = kwargs.get("verbose", False)

        # Suppress Metal/CUDA initialization messages if verbose=False
        context = _suppress_stderr if not verbose else _nullcontext
        with context():
            self._model = Llama(
                model_path=model_path,
                n_ctx=kwargs.get("n_ctx", 4096),
                n_gpu_layers=kwargs.get("n_gpu_layers", -1),
                n_threads=kwargs.get("n_threads", 0) or None,
                verbose=verbose,
                flash_attn=kwargs.get("flash_attn", True),
                chat_format=kwargs.get("chat_format", None),  # auto-detect
            )
        self._model_path = model_path

    def unload(self) -> None:
        if self._model:
            del self._model
            self._model = None

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()

        t0 = time.perf_counter()
        output = self._model(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop,
            seed=cfg.seed if cfg.seed >= 0 else None,
        )
        elapsed = time.perf_counter() - t0

        text = output["choices"][0]["text"]
        usage = output.get("usage", {})
        n_gen = usage.get("completion_tokens", 0)

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
            stop_reason=output["choices"][0].get("finish_reason", "stop"),
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        cfg = config or GenerationConfig()

        for chunk in self._model(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()

        msgs = [{"role": m.role, "content": m.content} for m in messages]

        t0 = time.perf_counter()
        output = self._model.create_chat_completion(
            messages=msgs,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop,
        )
        elapsed = time.perf_counter() - t0

        text = output["choices"][0]["message"]["content"]
        usage = output.get("usage", {})
        n_gen = usage.get("completion_tokens", 0)

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        cfg = config or GenerationConfig()
        msgs = [{"role": m.role, "content": m.content} for m in messages]

        for chunk in self._model.create_chat_completion(
            messages=msgs,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                yield text

    @property
    def model_name(self) -> str:
        return self._model_path.split("/")[-1] if self._model_path else ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

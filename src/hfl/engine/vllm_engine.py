# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
vLLM inference engine with true streaming support.

Uses AsyncLLMEngine for token-by-token streaming when available,
with fallback to synchronous LLM for basic generation.

Requires: vLLM installed with GPU support (pip install vllm)

WARNING: EXPERIMENTAL
    For production use, consider using the llama.cpp or Transformers backends
    until this implementation is fully validated on your hardware.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from queue import Empty, Queue
from typing import Iterator

from hfl.config import config as _hfl_config
from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)
from hfl.engine.prompt_builder import PromptBuilder, PromptFormat

logger = logging.getLogger(__name__)


class VLLMEngine(InferenceEngine):
    """vLLM-based inference engine with true async streaming.

    Supports two modes:
    - Async mode (AsyncLLMEngine): True token-by-token streaming
    - Sync mode (LLM): Fallback when AsyncLLMEngine is unavailable
    """

    def __init__(self):
        self._engine = None
        self._model_path = ""
        self._is_async = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._prompt_format = PromptFormat.CHATML

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @property
    def model_name(self) -> str:
        return self._model_path

    def _ensure_loop(self) -> None:
        """Start a background event loop for async operations."""
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, daemon=True, name="vllm-loop"
            )
            self._loop_thread.start()

    def _run_async(self, coro):
        """Run an async coroutine from sync context."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=300)

    def _detect_prompt_format(self, model_path: str) -> PromptFormat:
        """Detect prompt format from model path/name."""
        name = model_path.lower()
        if "llama-3" in name or "llama3" in name:
            return PromptFormat.LLAMA3
        if "llama-2" in name or "llama2" in name:
            return PromptFormat.LLAMA2
        if "vicuna" in name:
            return PromptFormat.VICUNA
        if "alpaca" in name:
            return PromptFormat.ALPACA
        return PromptFormat.CHATML

    def load(self, model_path: str, **kwargs) -> None:
        """Load a model with vLLM.

        Attempts AsyncLLMEngine for streaming; falls back to sync LLM.
        """
        self._model_path = model_path
        self._prompt_format = self._detect_prompt_format(model_path)

        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            self._ensure_loop()
            engine_args = AsyncEngineArgs(model=model_path, **kwargs)
            self._engine = self._run_async(AsyncLLMEngine.from_engine_args(engine_args))
            self._is_async = True
            logger.info("vLLM async engine loaded: %s", model_path)
        except (ImportError, AttributeError):
            from vllm import LLM

            self._engine = LLM(model=model_path, **kwargs)
            self._is_async = False
            logger.info("vLLM sync engine loaded (streaming limited): %s", model_path)

    def unload(self) -> None:
        """Unload the model and clean up resources."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=_hfl_config.vllm_shutdown_join_timeout)
            self._loop = None
            self._loop_thread = None
        self._engine = None
        self._is_async = False

    def _build_sampling_params(self, config: GenerationConfig | None = None):
        """Build vLLM SamplingParams from GenerationConfig."""
        from vllm import SamplingParams

        cfg = config or GenerationConfig()
        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            repetition_penalty=cfg.repeat_penalty,
        )

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        """Generate text completion."""
        if not self._engine:
            raise RuntimeError("Model not loaded")

        sampling_params = self._build_sampling_params(config)

        if self._is_async:
            return self._generate_async(prompt, sampling_params)
        return self._generate_sync(prompt, sampling_params)

    def _generate_async(self, prompt: str, sampling_params) -> GenerationResult:
        """Generate using AsyncLLMEngine."""
        request_id = str(uuid.uuid4())

        async def _gen():
            final = None
            async for output in self._engine.generate(prompt, sampling_params, request_id):
                final = output
            return final

        output = self._run_async(_gen())
        completion = output.outputs[0]

        return GenerationResult(
            text=completion.text,
            tokens_generated=len(completion.token_ids),
            stop_reason=(str(completion.finish_reason) if completion.finish_reason else "stop"),
        )

    def _generate_sync(self, prompt: str, sampling_params) -> GenerationResult:
        """Generate using sync LLM."""
        outputs = self._engine.generate([prompt], sampling_params)
        output = outputs[0]

        return GenerationResult(
            text=output.outputs[0].text,
            tokens_generated=len(output.outputs[0].token_ids),
            stop_reason="stop",
        )

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        """Stream text generation token by token.

        In async mode, yields incremental text deltas as they're generated.
        In sync mode, falls back to generating the full response and yielding it.
        """
        if not self._engine:
            raise RuntimeError("Model not loaded")

        sampling_params = self._build_sampling_params(config)

        if self._is_async:
            yield from self._stream_async(prompt, sampling_params)
        else:
            result = self._generate_sync(prompt, sampling_params)
            yield result.text

    def _stream_async(self, prompt: str, sampling_params) -> Iterator[str]:
        """True token-by-token streaming via AsyncLLMEngine."""
        request_id = str(uuid.uuid4())
        token_queue: Queue[str | None | Exception] = Queue(maxsize=100)

        async def _producer():
            prev_text = ""
            try:
                async for output in self._engine.generate(prompt, sampling_params, request_id):
                    for completion in output.outputs:
                        new_text = completion.text[len(prev_text) :]
                        if new_text:
                            token_queue.put(new_text, timeout=_hfl_config.stream_queue_put_timeout)
                            prev_text = completion.text
            except Exception as e:
                token_queue.put(e, timeout=_hfl_config.vllm_error_put_timeout)
            finally:
                token_queue.put(None, timeout=_hfl_config.vllm_error_put_timeout)

        asyncio.run_coroutine_threadsafe(_producer(), self._loop)

        while True:
            try:
                item = token_queue.get(timeout=_hfl_config.stream_queue_put_timeout)
            except Empty:
                raise TimeoutError("vLLM streaming timed out")
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        """Chat completion using PromptBuilder for format detection.

        ``tools`` is accepted but currently injected into the prompt only
        via PromptBuilder when supported. Structured tool-call parsing for
        vLLM output is handled upstream by the per-family parser, so the
        route layer still gets canonical tool_calls back.
        """
        prompt = PromptBuilder.build(messages, self._prompt_format, tools=tools)
        return self.generate(prompt, config)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        """Streaming chat completion."""
        prompt = PromptBuilder.build(messages, self._prompt_format, tools=tools)
        yield from self.generate_stream(prompt, config)

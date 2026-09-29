# SPDX-License-Identifier: Apache-2.0
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
import contextlib
import inspect
import logging
import threading
import uuid
from queue import Empty, Queue
from typing import Any, Iterator, cast

from hfl.config import config as _hfl_config
from hfl.engine import cancel
from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
    refuse_logprobs,
)
from hfl.engine.prompt_builder import PromptBuilder, PromptFormat

logger = logging.getLogger(__name__)


def _model_max_len(model_path: str) -> int | None:
    """The model's own context limit, from its ``config.json`` (None if
    unreadable): vLLM refuses a ``max_model_len`` above it."""
    import json
    from pathlib import Path

    try:
        cfg = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    for key in ("max_position_embeddings", "max_sequence_length", "seq_length"):
        value = cfg.get(key) if isinstance(cfg, dict) else None
        if isinstance(value, int) and value > 0:
            return value
    return None


def _prompt_tokens(output: Any) -> int:
    """The prompt's tokens as vLLM read them: reported as the request's
    input tokens (they were 0 before, on every API)."""
    return len(getattr(output, "prompt_token_ids", None) or [])


def _own_tools_on_path() -> None:
    """The folder of HFL's own interpreter on ``PATH``, if it is not.

    vLLM's kernels build at first use with ``ninja``, which pip installs next
    to the interpreter; flashinfer looks for it on ``PATH``. Run from a venv
    that is not activated (``uv tool``, pipx, a service calling
    ``venv/bin/hfl``), that folder is not there, and every load failed:
    "No such file or directory: 'ninja'" (measured on an L4).
    """
    import os
    import sys
    from pathlib import Path

    own = str(Path(sys.executable).parent)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if own not in parts:
        os.environ["PATH"] = os.pathsep.join([own, *[p for p in parts if p]])


def _vllm_args(model_path: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """HFL's load options as vLLM's engine arguments.

    Every engine gets the same options (``load_kwargs_for``): ``n_ctx``,
    ``lora_paths``, ``draft_model_path``. vLLM knows none of them, and passed
    as they were every load failed: "AsyncEngineArgs.__init__() got an
    unexpected keyword argument 'n_ctx'" (measured on an L4, vLLM 0.30).
    """
    args = dict(kwargs)
    n_ctx = args.pop("n_ctx", None)
    if isinstance(n_ctx, int) and n_ctx > 0:
        limit = _model_max_len(model_path)
        args["max_model_len"] = min(n_ctx, limit) if limit else n_ctx
    if args.pop("lora_paths", None):
        logger.warning("ADAPTER: HFL does not apply LoRA adapters on vLLM yet; ignored")
    if args.pop("draft_model_path", None):
        logger.warning("DRAFT: HFL does not set up speculative decoding on vLLM yet; ignored")
    return args


class VLLMEngine(InferenceEngine):
    """vLLM-based inference engine with true async streaming.

    Supports two modes:
    - Async mode (AsyncLLMEngine): True token-by-token streaming
    - Sync mode (LLM): Fallback when AsyncLLMEngine is unavailable
    """

    def __init__(self):
        self._engine: Any = None
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

    @property
    def supports_concurrent_inference(self) -> bool:
        """vLLM batches concurrent requests in its own scheduler.

        The only backend HFL ships that may run more than one inference at a
        time: continuous batching is the whole point of vLLM, and overlapping
        requests share the paged KV cache safely. So ``HFL_NUM_PARALLEL > 1``
        is meaningful here, unlike llama.cpp / Transformers.
        """
        return True

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
        loop = cast(asyncio.AbstractEventLoop, self._loop)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
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
        kwargs = _vllm_args(model_path, kwargs)
        _own_tools_on_path()

        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            self._ensure_loop()
            if _hfl_config.vllm_tensor_parallel_size > 1:
                # HFL_TENSOR_PARALLEL_SIZE: the GPUs one model is sharded over.
                kwargs.setdefault("tensor_parallel_size", _hfl_config.vllm_tensor_parallel_size)
            engine_args = AsyncEngineArgs(model=model_path, **kwargs)

            async def _create() -> Any:
                # from_engine_args is a plain classmethod (vLLM 0.6 through
                # 0.30 at least): handing its result to run_coroutine_threadsafe
                # raised TypeError on every real load — the tests faked it as
                # a coroutine. Built inside the engine's own loop, which then
                # runs its requests; awaited only if a version returns one.
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                return await engine if inspect.isawaitable(engine) else engine

            self._engine = self._run_async(_create())
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
            # ENG-10: honour the request seed for reproducible output
            # (vLLM treats None as random).
            seed=cfg.seed if cfg.seed >= 0 else None,
        )

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        """Generate text completion."""
        if not self._engine:
            raise RuntimeError("Model not loaded")
        refuse_logprobs(config, "vLLM")

        sampling_params = self._build_sampling_params(config)

        if self._is_async:
            return self._generate_async(prompt, sampling_params)
        return self._generate_sync(prompt, sampling_params)

    def _generate_async(self, prompt: str, sampling_params) -> GenerationResult:
        """Generate using AsyncLLMEngine."""
        request_id = str(uuid.uuid4())
        # This request's cancellation signal, taken here: the coroutine runs
        # on vLLM's loop thread, whose context does not carry it. A request
        # past its budget used to decode on to max_tokens; now it is aborted
        # in vLLM at its next token (as a closed stream already was).
        signal = cancel.current()

        async def _gen():
            final = None
            async for output in self._engine.generate(prompt, sampling_params, request_id):
                if signal is not None and signal.is_set():
                    abort = getattr(self._engine, "abort", None)
                    if abort is not None:
                        await abort(request_id)
                    raise cancel.GenerationCancelled("request cancelled")
                final = output
            return final

        output = self._run_async(_gen())
        completion = output.outputs[0]

        return GenerationResult(
            text=completion.text,
            tokens_generated=len(completion.token_ids),
            tokens_prompt=_prompt_tokens(output),
            stop_reason=(str(completion.finish_reason) if completion.finish_reason else "stop"),
        )

    def _generate_sync(self, prompt: str, sampling_params) -> GenerationResult:
        """Generate using sync LLM."""
        outputs = self._engine.generate([prompt], sampling_params)
        output = outputs[0]

        completion = output.outputs[0]
        return GenerationResult(
            text=completion.text,
            tokens_generated=len(completion.token_ids),
            tokens_prompt=_prompt_tokens(output),
            # vLLM says why it stopped; "length" = max_tokens cut the reply.
            stop_reason="length" if completion.finish_reason == "length" else "stop",
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
        loop = cast(asyncio.AbstractEventLoop, self._loop)
        # Set when the consumer goes away (client disconnect closes this
        # generator -> GeneratorExit at the ``yield`` below). The producer
        # checks it so it stops pulling from / pushing to vLLM.
        cancelled = threading.Event()

        async def _producer():
            prev_text = ""
            try:
                async for output in self._engine.generate(prompt, sampling_params, request_id):
                    if cancelled.is_set():
                        break
                    for completion in output.outputs:
                        new_text = completion.text[len(prev_text) :]
                        if new_text:
                            token_queue.put(new_text, timeout=_hfl_config.stream_queue_put_timeout)
                            prev_text = completion.text
            except Exception as e:
                if not cancelled.is_set():
                    with contextlib.suppress(Exception):
                        token_queue.put(e, timeout=_hfl_config.vllm_error_put_timeout)
            finally:
                if not cancelled.is_set():
                    with contextlib.suppress(Exception):
                        token_queue.put(None, timeout=_hfl_config.vllm_error_put_timeout)

        producer_future = asyncio.run_coroutine_threadsafe(_producer(), loop)

        try:
            while True:
                try:
                    # The consumer side takes the consumer knob. This read the
                    # PUT timeout, so HFL_STREAM_QUEUE_GET_TIMEOUT governed
                    # nothing and the documented value was never applied here.
                    item = token_queue.get(timeout=_hfl_config.stream_queue_get_timeout)
                except Empty:
                    raise TimeoutError("vLLM streaming timed out") from None
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # RES: on early teardown (client disconnect -> GeneratorExit here,
            # or a timeout/error) stop the producer and abort the in-flight
            # vLLM request, so the GPU stops decoding tokens for a client that
            # is already gone. AsyncLLMEngine.abort(request_id) exists for
            # exactly this — without it every cancelled stream leaks a live
            # generation (and the background coroutine) until it self-terminates.
            cancelled.set()
            abort = getattr(self._engine, "abort", None)
            if abort is not None:
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(abort(request_id), loop)
            producer_future.cancel()

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

    def count_prompt_tokens(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> int:
        """The prompt ``chat`` builds, tokenized by vLLM's own tokenizer as
        vLLM tokenizes a prompt (special tokens added)."""
        if self._engine is None:
            raise RuntimeError("no model loaded")
        prompt = PromptBuilder.build(messages, self._prompt_format, tools=tools)
        tokenizer = self._engine.get_tokenizer()
        if inspect.isawaitable(tokenizer):  # AsyncLLMEngine: a coroutine
            tokenizer = self._run_async(tokenizer)
        return len(tokenizer.encode(prompt))

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        """Streaming chat completion."""
        prompt = PromptBuilder.build(messages, self._prompt_format, tools=tools)
        yield from self.generate_stream(prompt, config)

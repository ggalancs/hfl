# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Shared API helpers for routes.

Consolidates common patterns used across OpenAI, Native, and TTS routes.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, TypeVar
from uuid import uuid4

from fastapi import HTTPException

from hfl.api.state import get_state
from hfl.config import config
from hfl.converter.formats import ModelType, detect_model_type
from hfl.engine.base import GenerationConfig
from hfl.engine.selector import select_engine, select_tts_engine
from hfl.models.registry import get_registry
from hfl.validators import ValidationError, validate_model_name

if TYPE_CHECKING:
    from hfl.engine.base import AudioEngine, InferenceEngine
    from hfl.models.manifest import ModelManifest

T = TypeVar("T")


# =============================================================================
# Timeout Utilities
# =============================================================================


async def run_with_timeout(
    func: Callable[..., T],
    *args: Any,
    timeout: float | None = None,
    operation: str = "operation",
    **kwargs: Any,
) -> T:
    """Run a sync function in thread pool with configurable timeout.

    Uses config.generation_timeout as default, providing consistent
    timeout handling across all API endpoints.

    Args:
        func: Synchronous function to call
        *args: Positional arguments for the function
        timeout: Timeout in seconds (None = use config.generation_timeout)
        operation: Operation name for error messages
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        HTTPException: 504 Gateway Timeout if the operation times out
    """
    effective_timeout = timeout if timeout is not None else config.generation_timeout

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=effective_timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "error": f"{operation} timed out",
                "code": "TIMEOUT",
                "timeout_seconds": effective_timeout,
                "operation": operation,
            },
        )


async def run_dispatched(
    func: Callable[..., T],
    *args: Any,
    timeout: float | None = None,
    operation: str = "operation",
    **kwargs: Any,
) -> T:
    """Run a sync engine call through the inference dispatcher (spec §5.3).

    This is the one-stop entry point for route handlers that want their
    engine call to honour the queue. It:

    1. Acquires a slot on the global :class:`InferenceDispatcher`.
    2. Runs ``func`` in a thread pool (sync backends are not re-entrant
       so the thread simply drives the engine).
    3. Applies the generation timeout from config.

    The 504 timeout path is unchanged: if the engine takes too long
    after having been dispatched, :class:`asyncio.TimeoutError` is
    raised. Dispatcher rejections (``QueueFullError`` /
    ``QueueTimeoutError``) propagate unchanged so the route layer can
    map them to 429 / 503 with the correct headers.
    """
    from hfl.core import get_dispatcher

    dispatcher = get_dispatcher()
    effective_timeout = (
        timeout if timeout is not None else config.generation_timeout
    )

    async def _execute() -> T:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": f"{operation} timed out",
                    "code": "TIMEOUT",
                    "timeout_seconds": effective_timeout,
                    "operation": operation,
                },
            )

    return await dispatcher.run(_execute)


async def acquire_stream_slot() -> Any:
    """Pre-acquire a dispatcher slot for a streaming endpoint.

    Returns either an already-entered async context manager (when
    acquisition succeeded) or a :class:`~fastapi.responses.JSONResponse`
    carrying a structured 429 / 503 envelope (spec §5.3). Route handlers
    check ``hasattr(result, "__aexit__")`` to tell them apart.
    """
    from hfl.api.errors import queue_full, queue_timeout
    from hfl.core import get_dispatcher
    from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

    dispatcher = get_dispatcher()
    cm = dispatcher.slot()
    try:
        await cm.__aenter__()
    except QueueFullError as exc:
        return queue_full(
            retry_after=exc.retry_after_seconds,
            depth=exc.depth,
            max_queued=exc.max_queued,
        )
    except QueueTimeoutError as exc:
        return queue_timeout(waited_seconds=exc.waited_seconds)
    return cm


def queue_response_from_error(exc: BaseException) -> Any | None:
    """Map a dispatcher exception to a pre-built JSONResponse.

    Returns ``None`` when ``exc`` is not a dispatcher rejection, so
    callers can ``return queue_response_from_error(exc) or raise``.
    """
    from hfl.api.errors import queue_full, queue_timeout
    from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

    if isinstance(exc, QueueFullError):
        return queue_full(
            retry_after=exc.retry_after_seconds,
            depth=exc.depth,
            max_queued=exc.max_queued,
        )
    if isinstance(exc, QueueTimeoutError):
        return queue_timeout(waited_seconds=exc.waited_seconds)
    return None


async def run_async_with_timeout(
    coro: Any,
    timeout: float | None = None,
    operation: str = "operation",
) -> Any:
    """Run an async coroutine with configurable timeout.

    Args:
        coro: Async coroutine to await
        timeout: Timeout in seconds (None = use config.generation_timeout)
        operation: Operation name for error messages

    Returns:
        Result of the coroutine

    Raises:
        HTTPException: 504 Gateway Timeout if the operation times out
    """
    effective_timeout = timeout if timeout is not None else config.generation_timeout

    try:
        return await asyncio.wait_for(coro, timeout=effective_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "error": f"{operation} timed out",
                "code": "TIMEOUT",
                "timeout_seconds": effective_timeout,
                "operation": operation,
            },
        )


async def ensure_llm_loaded(model_name: str) -> tuple["InferenceEngine", "ModelManifest"]:
    """Ensure LLM model is loaded and return engine + manifest.

    This is the primary entry point for model loading in API routes.
    Handles validation, registry lookup, type checking, and loading.

    Args:
        model_name: Name, alias, or repo_id of the model

    Returns:
        Tuple of (InferenceEngine, ModelManifest)

    Raises:
        HTTPException: 400 for validation errors, 404 if model not found
    """
    # Validate input
    try:
        validate_model_name(model_name)
    except ValidationError as e:
        raise HTTPException(400, detail=str(e))

    state = get_state()

    # Fast path - already loaded
    if state.current_model and state.current_model.name == model_name:
        if state.engine is None:
            raise HTTPException(503, detail="Model engine not available")
        return state.engine, state.current_model

    # Lookup in registry
    manifest = get_registry().get(model_name)
    if not manifest:
        raise HTTPException(404, detail=f"Model not found: {model_name}")

    # Verify model type
    model_path = Path(manifest.local_path)
    model_type = detect_model_type(model_path)
    if model_type != ModelType.LLM:
        raise HTTPException(
            400,
            detail={
                "error": "Model type mismatch",
                "code": "MODEL_TYPE_MISMATCH",
                "expected": "llm",
                "got": model_type.value,
            },
        )

    # Load model
    engine = select_engine(model_path)
    engine.load(manifest.local_path)
    await state.set_llm_engine(engine, manifest)

    return engine, manifest


async def ensure_tts_loaded(model_name: str) -> tuple["AudioEngine", "ModelManifest"]:
    """Ensure TTS model is loaded and return engine + manifest.

    Args:
        model_name: Name, alias, or repo_id of the TTS model

    Returns:
        Tuple of (AudioEngine, ModelManifest)

    Raises:
        HTTPException: 400 for validation errors, 404 if model not found
    """
    try:
        validate_model_name(model_name)
    except ValidationError as e:
        raise HTTPException(400, detail=str(e))

    state = get_state()

    if state.current_tts_model and state.current_tts_model.name == model_name:
        if state.tts_engine is None:
            raise HTTPException(503, detail="TTS engine not available")
        return state.tts_engine, state.current_tts_model

    manifest = get_registry().get(model_name)
    if not manifest:
        raise HTTPException(404, detail=f"Model not found: {model_name}")

    model_path = Path(manifest.local_path)
    model_type = detect_model_type(model_path)
    if model_type != ModelType.TTS:
        raise HTTPException(
            400,
            detail={
                "error": "Model type mismatch",
                "code": "MODEL_TYPE_MISMATCH",
                "expected": "tts",
                "got": model_type.value,
            },
        )

    engine = select_tts_engine(model_path)
    engine.load(manifest.local_path)
    await state.set_tts_engine(engine, manifest)

    return engine, manifest


def options_to_config(options: dict[str, Any] | None) -> GenerationConfig:
    """Convert Ollama-style options dict to GenerationConfig.

    Args:
        options: Dictionary of generation options (Ollama format)

    Returns:
        GenerationConfig with mapped values
    """
    if not options:
        return GenerationConfig()

    # Validate types at API boundary
    def _validate_float(
        key: str,
        val: Any,
        min_val: float = 0.0,
        max_val: float = float("inf"),
    ) -> float:
        if not isinstance(val, (int, float)):
            raise HTTPException(400, detail=f"'{key}' must be a number, got {type(val).__name__}")
        fval = float(val)
        if not min_val <= fval <= max_val:
            raise HTTPException(400, detail=f"'{key}' must be between {min_val} and {max_val}")
        return fval

    def _validate_int(key: str, val: Any, min_val: int = 0) -> int:
        if not isinstance(val, int):
            raise HTTPException(400, detail=f"'{key}' must be an integer, got {type(val).__name__}")
        if val < min_val:
            raise HTTPException(400, detail=f"'{key}' must be >= {min_val}")
        return val

    config = GenerationConfig()
    if "temperature" in options and options["temperature"] is not None:
        config.temperature = _validate_float("temperature", options["temperature"], 0.0, 2.0)
    if "top_p" in options and options["top_p"] is not None:
        config.top_p = _validate_float("top_p", options["top_p"], 0.0, 1.0)
    if "top_k" in options and options["top_k"] is not None:
        config.top_k = _validate_int("top_k", options["top_k"], 0)
    if "num_predict" in options and options["num_predict"] is not None:
        config.max_tokens = _validate_int("num_predict", options["num_predict"], 1)
    if "repeat_penalty" in options and options["repeat_penalty"] is not None:
        config.repeat_penalty = _validate_float(
            "repeat_penalty",
            options["repeat_penalty"],
            0.0,
            10.0,
        )
    if "stop" in options:
        config.stop = options["stop"]
    if "seed" in options and options["seed"] is not None:
        config.seed = options["seed"]
    return config


def request_to_config(
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | str | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> GenerationConfig:
    """Convert OpenAI-style request params to GenerationConfig.

    Args:
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling
        max_tokens: Maximum tokens to generate
        stop: Stop sequences
        seed: Random seed
        **kwargs: Additional parameters (ignored)

    Returns:
        GenerationConfig
    """
    # Normalize stop to list
    stop_list: list[str] | None = None
    if stop is not None:
        if isinstance(stop, str):
            stop_list = [stop]
        else:
            stop_list = list(stop)

    config = GenerationConfig()
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
            raise HTTPException(400, detail="temperature must be between 0.0 and 2.0")
        config.temperature = float(temperature)
    if top_p is not None:
        if not isinstance(top_p, (int, float)) or not 0.0 <= top_p <= 1.0:
            raise HTTPException(400, detail="top_p must be between 0.0 and 1.0")
        config.top_p = float(top_p)
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise HTTPException(400, detail="max_tokens must be a positive integer")
        config.max_tokens = max_tokens
    if stop_list is not None:
        config.stop = stop_list
    if seed is not None:
        config.seed = seed
    return config


# Optimized streaming helpers


class StreamingContext:
    """Context for optimized streaming responses.

    Generates IDs and timestamps once at start, avoiding repeated
    UUID and time calls for each token.
    """

    def __init__(self, model_name: str, object_type: str = "chat.completion.chunk"):
        self.request_id = f"chatcmpl-{uuid4().hex[:12]}"
        self.created = int(time.time())
        self.model = model_name
        self.object_type = object_type

    def format_chunk(
        self,
        content: str | None = None,
        finish_reason: str | None = None,
        index: int = 0,
    ) -> str:
        """Format a single SSE chunk.

        Args:
            content: Token content (None for final chunk)
            finish_reason: Finish reason for final chunk
            index: Choice index

        Returns:
            SSE-formatted string
        """
        delta = {"content": content} if content is not None else {}

        chunk = {
            "id": self.request_id,
            "object": self.object_type,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": index,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_done(self) -> str:
        """Format the [DONE] marker."""
        return "data: [DONE]\n\n"


async def stream_openai_chat(
    engine: "InferenceEngine",
    messages: list,
    config: GenerationConfig,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream chat completion in OpenAI format.

    Optimized to generate IDs and timestamps once at start.
    Uses simple_stream_async for async-safe iteration.

    Args:
        engine: Loaded inference engine
        messages: List of ChatMessage objects
        config: Generation configuration
        model_name: Model name for response

    Yields:
        SSE-formatted chunks
    """
    from hfl.api.streaming import simple_stream_async

    ctx = StreamingContext(model_name)

    async for chunk in simple_stream_async(
        sync_iterator=engine.chat_stream(messages, config),
        format_item=lambda token: ctx.format_chunk(content=token),
        format_done=lambda: ctx.format_chunk(finish_reason="stop") + ctx.format_done(),
    ):
        yield chunk


async def stream_openai_completion(
    engine: "InferenceEngine",
    prompt: str,
    config: GenerationConfig,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream text completion in OpenAI format.

    Uses simple_stream_async for async-safe iteration.

    Args:
        engine: Loaded inference engine
        prompt: Text prompt
        config: Generation configuration
        model_name: Model name for response

    Yields:
        SSE-formatted chunks
    """
    from hfl.api.streaming import simple_stream_async

    ctx = StreamingContext(model_name, object_type="text_completion")

    async for chunk in simple_stream_async(
        sync_iterator=engine.generate_stream(prompt, config),
        format_item=lambda token: ctx.format_chunk(content=token),
        format_done=lambda: ctx.format_chunk(finish_reason="stop") + ctx.format_done(),
    ):
        yield chunk


def format_ndjson_chunk(
    content: str,
    model: str,
    done: bool = False,
    created_at: str | None = None,
    **extra: Any,
) -> str:
    """Format a chunk in NDJSON format (Ollama style).

    Args:
        content: Response content
        model: Model name
        done: Whether this is the final chunk
        created_at: Pre-computed timestamp (optional, avoids repeated time calls)
        **extra: Additional fields

    Returns:
        NDJSON-formatted string
    """
    chunk = {
        "model": model,
        "created_at": created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": content,
        "done": done,
        **extra,
    }
    return json.dumps(chunk) + "\n"


async def stream_ollama_generate(
    engine: "InferenceEngine",
    prompt: str,
    config: GenerationConfig,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream text generation in Ollama format.

    Uses simple_stream_async for async-safe iteration.

    Args:
        engine: Loaded inference engine
        prompt: Text prompt
        config: Generation configuration
        model_name: Model name for response

    Yields:
        NDJSON-formatted chunks
    """
    from hfl.api.streaming import simple_stream_async

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    async for chunk in simple_stream_async(
        sync_iterator=engine.generate_stream(prompt, config),
        format_item=lambda token: format_ndjson_chunk(token, model_name, created_at=created_at),
        format_done=lambda: format_ndjson_chunk("", model_name, done=True),
    ):
        yield chunk


async def stream_ollama_chat(
    engine: "InferenceEngine",
    messages: list,
    config: GenerationConfig,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream chat in Ollama format.

    Uses simple_stream_async for async-safe iteration.

    Args:
        engine: Loaded inference engine
        messages: List of ChatMessage objects
        config: Generation configuration
        model_name: Model name for response

    Yields:
        NDJSON-formatted chunks
    """
    from hfl.api.streaming import simple_stream_async

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def format_chat_chunk(token: str) -> str:
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "message": {"role": "assistant", "content": token},
            "done": False,
        }
        return json.dumps(chunk) + "\n"

    def format_chat_done() -> str:
        final = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }
        return json.dumps(final) + "\n"

    async for chunk in simple_stream_async(
        sync_iterator=engine.chat_stream(messages, config),
        format_item=format_chat_chunk,
        format_done=format_chat_done,
    ):
        yield chunk

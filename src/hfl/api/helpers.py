# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Shared API helpers for routes.

Consolidates common patterns used across OpenAI, Native, and TTS routes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator
from uuid import uuid4

from fastapi import HTTPException

from hfl.api.state import get_state
from hfl.converter.formats import ModelType, detect_model_type
from hfl.engine.base import GenerationConfig
from hfl.engine.selector import select_engine, select_tts_engine
from hfl.models.registry import get_registry
from hfl.validators import ValidationError, validate_model_name

if TYPE_CHECKING:
    from hfl.engine.base import AudioEngine, InferenceEngine
    from hfl.models.manifest import ModelManifest


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
        assert state.engine is not None
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
        assert state.tts_engine is not None
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

    return GenerationConfig(
        temperature=options.get("temperature"),
        top_p=options.get("top_p"),
        top_k=options.get("top_k"),
        max_tokens=options.get("num_predict"),
        repeat_penalty=options.get("repeat_penalty"),
        stop=options.get("stop"),
        seed=options.get("seed"),
    )


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

    return GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_list,
        seed=seed,
    )


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

    Args:
        engine: Loaded inference engine
        messages: List of ChatMessage objects
        config: Generation configuration
        model_name: Model name for response

    Yields:
        SSE-formatted chunks
    """
    ctx = StreamingContext(model_name)

    async for token in engine.chat_stream(messages, config):
        yield ctx.format_chunk(content=token)

    yield ctx.format_chunk(finish_reason="stop")
    yield ctx.format_done()


async def stream_openai_completion(
    engine: "InferenceEngine",
    prompt: str,
    config: GenerationConfig,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream text completion in OpenAI format.

    Args:
        engine: Loaded inference engine
        prompt: Text prompt
        config: Generation configuration
        model_name: Model name for response

    Yields:
        SSE-formatted chunks
    """
    ctx = StreamingContext(model_name, object_type="text_completion")

    async for token in engine.generate_stream(prompt, config):
        yield ctx.format_chunk(content=token)

    yield ctx.format_chunk(finish_reason="stop")
    yield ctx.format_done()


def format_ndjson_chunk(
    content: str,
    model: str,
    done: bool = False,
    **extra: Any,
) -> str:
    """Format a chunk in NDJSON format (Ollama style).

    Args:
        content: Response content
        model: Model name
        done: Whether this is the final chunk
        **extra: Additional fields

    Returns:
        NDJSON-formatted string
    """
    chunk = {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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

    Args:
        engine: Loaded inference engine
        prompt: Text prompt
        config: Generation configuration
        model_name: Model name for response

    Yields:
        NDJSON-formatted chunks
    """
    async for token in engine.generate_stream(prompt, config):
        yield format_ndjson_chunk(token, model_name)

    yield format_ndjson_chunk("", model_name, done=True)


async def stream_ollama_chat(
    engine: "InferenceEngine",
    messages: list,
    config: GenerationConfig,
    model_name: str,
) -> AsyncIterator[str]:
    """Stream chat in Ollama format.

    Args:
        engine: Loaded inference engine
        messages: List of ChatMessage objects
        config: Generation configuration
        model_name: Model name for response

    Yields:
        NDJSON-formatted chunks
    """
    full_response = []

    async for token in engine.chat_stream(messages, config):
        full_response.append(token)
        chunk = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": token},
            "done": False,
        }
        yield json.dumps(chunk) + "\n"

    # Final chunk with full message
    final = {
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": "".join(full_response)},
        "done": True,
    }
    yield json.dumps(final) + "\n"

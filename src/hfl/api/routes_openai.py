# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatible with the OpenAI API.
Drop-in replacement for applications using the OpenAI SDK.
"""

import contextlib
import json
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Union

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response, StreamingResponse

from hfl.api.converters import openai_to_generation_config
from hfl.api.errors import service_unavailable
from hfl.api.helpers import (
    acquire_stream_slot,
    queue_response_from_error,
    run_dispatched,
)
from hfl.api.schemas import ChatCompletionRequest, CompletionRequest
from hfl.core.container import get_registry
from hfl.engine.base import ChatMessage, GenerationConfig
from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

if TYPE_CHECKING:
    from hfl.api.state import ServerState

router = APIRouter(tags=["OpenAI API"])


# --- Helpers ---


def _get_state() -> "ServerState":
    """Get the singleton server state."""
    from hfl.api.state import get_state

    return get_state()


async def _ensure_model_loaded(model_name: str) -> None:
    """Load the model if it is not already in memory (thread-safe)."""
    from hfl.api.model_loader import load_llm

    await load_llm(model_name)


def _to_gen_config(req: Union[ChatCompletionRequest, CompletionRequest]) -> GenerationConfig:
    """Convert OpenAI request to GenerationConfig."""
    return openai_to_generation_config(req)


# --- Endpoints ---


@router.post(
    "/v1/chat/completions",
    response_model=None,
    tags=["OpenAI"],
    summary="Create chat completion",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def chat_completions(
    req: ChatCompletionRequest,
) -> dict[str, Any] | StreamingResponse | Response:
    await _ensure_model_loaded(req.model)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{req.model}' failed to load")

    messages = [ChatMessage(role=m.role, content=m.content) for m in req.messages]
    gen_config = _to_gen_config(req)

    if req.stream:
        slot_or_response = await acquire_stream_slot()
        if isinstance(slot_or_response, JSONResponse):
            return slot_or_response
        return StreamingResponse(
            _stream_chat(req.model, messages, gen_config, slot_or_response),
            media_type="text/event-stream",
        )

    # Run sync engine call in thread pool with timeout, serialized by
    # the inference dispatcher (spec §5.3).
    try:
        result = await run_dispatched(
            state.engine.chat,
            messages,
            gen_config,
            operation="chat_completion",
        )
    except (QueueFullError, QueueTimeoutError) as exc:
        return queue_response_from_error(exc)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.stop_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.tokens_prompt,
            "completion_tokens": result.tokens_generated,
            "total_tokens": result.tokens_prompt + result.tokens_generated,
        },
    }


async def _stream_chat(
    model: str,
    messages: list[ChatMessage],
    config: GenerationConfig,
    slot_cm: Any | None = None,
) -> AsyncIterator[str]:
    """Generate OpenAI-compatible SSE responses with backpressure.

    ``slot_cm`` is the dispatcher slot held for the entire stream
    (spec §5.3). It is released in ``finally`` regardless of outcome.
    """
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        err = json.dumps(
            {
                "error": "Model not loaded",
                "code": "SERVICE_UNAVAILABLE",
            }
        )
        yield f"data: {err}\n\n"
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)
        return

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())  # Consistent timestamp across all chunks
    first_chunk = True

    def format_chunk(token: str) -> str:
        nonlocal first_chunk
        # OpenAI includes role in first chunk
        delta: dict[str, str] = {"content": token}
        if first_chunk:
            delta["role"] = "assistant"
            first_chunk = False

        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": None,  # OpenAI compatibility
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_done() -> str:
        final = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": None,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=state.engine.chat_stream(messages, config),
            format_item=format_chunk,
            format_done=format_done,
        ):
            yield chunk
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'code': 'STREAM_ERROR'})}\n\n"
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)


@router.post(
    "/v1/completions",
    response_model=None,
    tags=["OpenAI"],
    summary="Create text completion",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def completions(req: CompletionRequest) -> dict[str, Any] | StreamingResponse | Response:
    await _ensure_model_loaded(req.model)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{req.model}' failed to load")

    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    gen_config = _to_gen_config(req)

    if req.stream:
        slot_or_response = await acquire_stream_slot()
        if isinstance(slot_or_response, JSONResponse):
            return slot_or_response
        return StreamingResponse(
            _stream_completion(req.model, prompt, gen_config, slot_or_response),
            media_type="text/event-stream",
        )

    # Run sync engine call in thread pool with timeout, serialized by
    # the inference dispatcher (spec §5.3).
    try:
        result = await run_dispatched(
            state.engine.generate,
            prompt,
            gen_config,
            operation="text_completion",
        )
    except (QueueFullError, QueueTimeoutError) as exc:
        return queue_response_from_error(exc)

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "text": result.text,
                "index": 0,
                "finish_reason": result.stop_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.tokens_prompt,
            "completion_tokens": result.tokens_generated,
            "total_tokens": result.tokens_prompt + result.tokens_generated,
        },
    }


async def _stream_completion(
    model: str,
    prompt: str,
    config: GenerationConfig,
    slot_cm: Any | None = None,
) -> AsyncIterator[str]:
    """Generate text completion SSE responses with backpressure.

    ``slot_cm`` is the dispatcher slot held for the entire stream
    (spec §5.3); it is released in ``finally``.
    """
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        err = json.dumps(
            {
                "error": "Model not loaded",
                "code": "SERVICE_UNAVAILABLE",
            }
        )
        yield f"data: {err}\n\n"
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)
        return

    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())  # Consistent timestamp across all chunks

    def format_chunk(token: str) -> str:
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "system_fingerprint": None,  # OpenAI compatibility
            "choices": [{"text": token, "index": 0, "finish_reason": None}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_done() -> str:
        return "data: [DONE]\n\n"

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=state.engine.generate_stream(prompt, config),
            format_item=format_chunk,
            format_done=format_done,
        ):
            yield chunk
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'code': 'STREAM_ERROR'})}\n\n"
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)


@router.get("/v1/models", tags=["OpenAI"], summary="List available models")
async def list_models() -> dict[str, Any]:
    registry = get_registry()
    models = registry.list_all()
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.repo_id.split("/")[0] if "/" in m.repo_id else "local",
            }
            for m in models
        ],
    }

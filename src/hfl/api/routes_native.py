# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatible with the Ollama native API.
Allows using hfl as a drop-in replacement for Ollama.
"""

import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from hfl.api.converters import ollama_to_generation_config
from hfl.api.errors import service_unavailable
from hfl.api.helpers import run_with_timeout
from hfl.api.schemas import ChatRequest, GenerateRequest
from hfl.core.container import get_registry
from hfl.engine.base import ChatMessage, GenerationConfig

if TYPE_CHECKING:
    from hfl.api.state import ServerState

router = APIRouter(tags=["Ollama API"])


# --- Helpers ---


def _get_state() -> "ServerState":
    """Get the singleton server state."""
    from hfl.api.state import get_state

    return get_state()


async def _ensure_model_loaded(model_name: str) -> None:
    """Load the model if it is not already in memory (thread-safe)."""
    from hfl.api.model_loader import load_llm

    await load_llm(model_name)


def _options_to_config(options: dict | None) -> GenerationConfig:
    """Convert Ollama options dict to GenerationConfig."""
    return ollama_to_generation_config(options)


# --- Endpoints ---


@router.post(
    "/api/generate",
    response_model=None,
    tags=["Ollama"],
    summary="Generate text",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def api_generate(req: GenerateRequest) -> dict[str, Any] | StreamingResponse:
    await _ensure_model_loaded(req.model)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{req.model}' failed to load")

    gen_config = _options_to_config(req.options)

    if req.stream:
        return StreamingResponse(
            _stream_generate(req.model, req.prompt, gen_config),
            media_type="application/x-ndjson",
        )

    # Run sync engine call in thread pool with timeout
    result = await run_with_timeout(
        state.engine.generate, req.prompt, gen_config, operation="generate"
    )
    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "response": result.text,
        "done": True,
        "total_duration": 0,
        "eval_count": result.tokens_generated,
        "eval_duration": 0,
    }


async def _stream_generate(
    model_name: str,
    prompt: str,
    config: GenerationConfig,
) -> AsyncIterator[str]:
    """Stream text generation in Ollama NDJSON format with backpressure."""
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        error = {"model": model_name, "error": "Model not loaded", "done": True}
        yield json.dumps(error) + "\n"
        return

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def format_chunk(token: str) -> str:
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "response": token,
            "done": False,
        }
        return json.dumps(chunk) + "\n"

    def format_done() -> str:
        chunk = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": "",
            "done": True,
        }
        return json.dumps(chunk) + "\n"

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=state.engine.generate_stream(prompt, config),
            format_item=format_chunk,
            format_done=format_done,
        ):
            yield chunk
    except Exception as e:
        error = {"model": model_name, "error": str(e), "done": True}
        yield json.dumps(error) + "\n"


@router.post(
    "/api/chat",
    response_model=None,
    tags=["Ollama"],
    summary="Chat with model",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def api_chat(req: ChatRequest) -> dict[str, Any] | StreamingResponse:
    await _ensure_model_loaded(req.model)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{req.model}' failed to load")

    messages = [ChatMessage(role=m.role, content=m.content) for m in req.messages]
    gen_config = _options_to_config(req.options)

    if req.stream:
        return StreamingResponse(
            _stream_chat(req.model, messages, gen_config),
            media_type="application/x-ndjson",
        )

    # Run sync engine call in thread pool with timeout
    result = await run_with_timeout(
        state.engine.chat, messages, gen_config, operation="chat"
    )
    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": {"role": "assistant", "content": result.text},
        "done": True,
    }


async def _stream_chat(
    model_name: str,
    messages: list[ChatMessage],
    config: GenerationConfig,
) -> AsyncIterator[str]:
    """Stream chat in Ollama NDJSON format with backpressure."""
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        error = {"model": model_name, "error": "Model not loaded", "done": True}
        yield json.dumps(error) + "\n"
        return

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def format_chunk(token: str) -> str:
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "message": {"role": "assistant", "content": token},
            "done": False,
        }
        return json.dumps(chunk) + "\n"

    def format_done() -> str:
        chunk = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }
        return json.dumps(chunk) + "\n"

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=state.engine.chat_stream(messages, config),
            format_item=format_chunk,
            format_done=format_done,
        ):
            yield chunk
    except Exception as e:
        error = {"model": model_name, "error": str(e), "done": True}
        yield json.dumps(error) + "\n"


@router.get("/api/tags", tags=["Ollama"], summary="List local models")
async def api_tags() -> dict[str, Any]:
    """List models (Ollama compatible).

    Returns model list matching Ollama's /api/tags format.
    Uses None instead of empty strings for missing optional fields.
    """
    registry = get_registry()
    return {
        "models": [
            {
                "name": m.name,
                "modified_at": m.created_at,
                "size": m.size_bytes,
                "digest": m.file_hash or "",  # Use actual hash if available
                "details": {
                    "parent_model": m.repo_id,
                    "format": m.format,
                    "family": m.architecture,  # None instead of ""
                    "parameter_size": m.parameters,  # None instead of ""
                    "quantization_level": m.quantization,  # None instead of ""
                },
            }
            for m in registry.list_all()
        ]
    }


@router.get("/api/version", tags=["Ollama"], summary="Get server version")
async def api_version() -> dict[str, str]:
    """Server version (Ollama compatible)."""
    return {"version": "0.1.0"}


@router.head("/")
async def head_root() -> dict[str, Any]:
    """Health check for Ollama compatibility."""
    return {}

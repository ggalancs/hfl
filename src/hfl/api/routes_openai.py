# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatibles con la API de OpenAI.
Drop-in replacement para aplicaciones que usan OpenAI SDK.
"""

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hfl.engine.base import ChatMessage, GenerationConfig
from hfl.models.registry import ModelRegistry

router = APIRouter()


# --- Schemas Pydantic ---


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None


# --- Helpers ---


def _get_state():
    """Import state lazily to avoid circular imports."""
    from hfl.api.server import state

    return state


def _ensure_model_loaded(model_name: str):
    """Carga el modelo si no está ya en memoria."""
    state = _get_state()

    if state.engine and state.engine.is_loaded:
        if state.current_model and state.current_model.name == model_name:
            return
        # Modelo diferente, descargar el actual
        state.engine.unload()

    registry = ModelRegistry()
    manifest = registry.get(model_name)
    if not manifest:
        raise HTTPException(404, f"Modelo no encontrado: {model_name}")

    from pathlib import Path

    from hfl.engine.selector import select_engine

    state.engine = select_engine(Path(manifest.local_path))
    state.engine.load(manifest.local_path, n_ctx=manifest.context_length)
    state.current_model = manifest


def _to_gen_config(req) -> GenerationConfig:
    stop = req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else None)
    return GenerationConfig(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens or 2048,
        stop=stop,
        seed=req.seed or -1,
    )


# --- Endpoints ---


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    _ensure_model_loaded(req.model)
    state = _get_state()

    messages = [ChatMessage(role=m.role, content=m.content) for m in req.messages]
    gen_config = _to_gen_config(req)

    if req.stream:
        return StreamingResponse(
            _stream_chat(req.model, messages, gen_config),
            media_type="text/event-stream",
        )

    result = state.engine.chat(messages, gen_config)

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
) -> AsyncIterator[str]:
    """Genera respuestas SSE compatibles con OpenAI."""
    state = _get_state()
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    for token in state.engine.chat_stream(messages, config):
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Chunk final
    final = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/completions")
async def completions(req: CompletionRequest):
    _ensure_model_loaded(req.model)
    state = _get_state()

    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    gen_config = _to_gen_config(req)

    if req.stream:
        return StreamingResponse(
            _stream_completion(req.model, prompt, gen_config),
            media_type="text/event-stream",
        )

    result = state.engine.generate(prompt, gen_config)

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
) -> AsyncIterator[str]:
    state = _get_state()
    for token in state.engine.generate_stream(prompt, config):
        chunk = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"text": token, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/v1/models")
async def list_models():
    registry = ModelRegistry()
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

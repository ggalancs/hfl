# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatibles con la API nativa de Ollama.
Permite usar hfl como drop-in replacement de Ollama.
"""

import time
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hfl.engine.base import ChatMessage, GenerationConfig
from hfl.models.registry import ModelRegistry

router = APIRouter()


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    options: dict | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = True
    options: dict | None = None


def _get_state():
    """Import state lazily to avoid circular imports."""
    from hfl.api.server import state
    return state


def _options_to_config(options: dict | None) -> GenerationConfig:
    opts = options or {}
    return GenerationConfig(
        temperature=opts.get("temperature", 0.7),
        top_p=opts.get("top_p", 0.9),
        top_k=opts.get("top_k", 40),
        max_tokens=opts.get("num_predict", 2048),
        repeat_penalty=opts.get("repeat_penalty", 1.1),
        seed=opts.get("seed", -1),
        stop=opts.get("stop"),
    )


@router.post("/api/generate")
async def api_generate(req: GenerateRequest):
    from hfl.api.routes_openai import _ensure_model_loaded
    _ensure_model_loaded(req.model)
    state = _get_state()

    gen_config = _options_to_config(req.options)

    if req.stream:
        async def stream():
            for token in state.engine.generate_stream(req.prompt, gen_config):
                yield json.dumps({
                    "model": req.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "response": token,
                    "done": False,
                }) + "\n"

            yield json.dumps({
                "model": req.model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "response": "",
                "done": True,
            }) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    result = state.engine.generate(req.prompt, gen_config)
    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "response": result.text,
        "done": True,
        "total_duration": 0,
        "eval_count": result.tokens_generated,
        "eval_duration": 0,
    }


@router.post("/api/chat")
async def api_chat(req: ChatRequest):
    from hfl.api.routes_openai import _ensure_model_loaded
    _ensure_model_loaded(req.model)
    state = _get_state()

    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]
    gen_config = _options_to_config(req.options)

    if req.stream:
        async def stream():
            for token in state.engine.chat_stream(messages, gen_config):
                yield json.dumps({
                    "model": req.model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "message": {"role": "assistant", "content": token},
                    "done": False,
                }) + "\n"

            yield json.dumps({
                "model": req.model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "message": {"role": "assistant", "content": ""},
                "done": True,
            }) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    result = state.engine.chat(messages, gen_config)
    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": {"role": "assistant", "content": result.text},
        "done": True,
    }


@router.get("/api/tags")
async def api_tags():
    """Lista modelos (compatible con Ollama)."""
    registry = ModelRegistry()
    return {
        "models": [
            {
                "name": m.name,
                "model": m.name,
                "modified_at": m.created_at,
                "size": m.size_bytes,
                "digest": "",
                "details": {
                    "parent_model": m.repo_id,
                    "format": m.format,
                    "family": m.architecture or "",
                    "parameter_size": m.parameters or "",
                    "quantization_level": m.quantization or "",
                },
            }
            for m in registry.list_all()
        ]
    }


@router.get("/api/version")
async def api_version():
    """Versión del servidor (compatible con Ollama)."""
    return {"version": "0.1.0"}


@router.head("/")
async def head_root():
    """Health check para compatibilidad con Ollama."""
    return {}

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatible with the Ollama native API.
Allows using hfl as a drop-in replacement for Ollama.
"""

import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from hfl.api.converters import ollama_to_generation_config
from hfl.api.schemas import ChatRequest, GenerateRequest
from hfl.core.container import get_registry
from hfl.engine.base import ChatMessage, GenerationConfig

router = APIRouter(tags=["Ollama API"])


def _options_to_config(options: dict | None) -> GenerationConfig:
    """Convert Ollama options dict to GenerationConfig."""
    return ollama_to_generation_config(options)


@router.post("/api/generate", response_model=None)
async def api_generate(req: GenerateRequest) -> dict[str, Any] | StreamingResponse:
    from hfl.api.helpers import ensure_llm_loaded, run_with_timeout, stream_ollama_generate

    engine, _ = await ensure_llm_loaded(req.model)
    gen_config = _options_to_config(req.options)

    if req.stream:
        return StreamingResponse(
            stream_ollama_generate(engine, req.prompt, gen_config, req.model),
            media_type="application/x-ndjson",
        )

    # Run sync engine call in thread pool with timeout
    result = await run_with_timeout(
        engine.generate, req.prompt, gen_config, operation="generate"
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


@router.post("/api/chat", response_model=None)
async def api_chat(req: ChatRequest) -> dict[str, Any] | StreamingResponse:
    from hfl.api.helpers import ensure_llm_loaded, run_with_timeout, stream_ollama_chat

    engine, _ = await ensure_llm_loaded(req.model)
    messages = [ChatMessage(role=m.role, content=m.content) for m in req.messages]
    gen_config = _options_to_config(req.options)

    if req.stream:
        return StreamingResponse(
            stream_ollama_chat(engine, messages, gen_config, req.model),
            media_type="application/x-ndjson",
        )

    # Run sync engine call in thread pool with timeout
    result = await run_with_timeout(
        engine.chat, messages, gen_config, operation="chat"
    )
    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": {"role": "assistant", "content": result.text},
        "done": True,
    }


@router.get("/api/tags")
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


@router.get("/api/version")
async def api_version() -> dict[str, str]:
    """Server version (Ollama compatible)."""
    return {"version": "0.1.0"}


@router.head("/")
async def head_root() -> dict[str, Any]:
    """Health check for Ollama compatibility."""
    return {}

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Batched generation endpoint (Phase 15 P2 — V2 row 8).

Runs N prompts against one already-loaded model in a single HTTP
round-trip. Today each request serialises through the existing
dispatcher; Phase 15's row-12 work (continuous batching) upgrades
this to parallel decoding under the hood without changing the
request / response wire shape.

Request::

    POST /api/batch
    {
      "model": "llama3.3",
      "requests": [
        {"prompt": "why is the sky blue?", "options": {...}},
        {"prompt": "2 + 2 = ?",            "options": {...}}
      ],
      "keep_alive": "5m"
    }

Response::

    {
      "model": "llama3.3",
      "results": [
        {"response": "...", "total_duration": ...},
        {"response": "...", "total_duration": ...}
      ]
    }

Every result in the array mirrors ``/api/generate``'s non-streaming
envelope (``response`` / ``total_duration`` / ``prompt_eval_count``
/ etc.), minus the top-level ``model`` (it's the same for every
element and would be redundant).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hfl.api.converters import ollama_to_generation_config
from hfl.api.errors import service_unavailable
from hfl.api.helpers import apply_keep_alive, run_dispatched, unload_after_response
from hfl.api.model_loader import load_llm
from hfl.api.state import get_state
from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ollama"])


class BatchItem(BaseModel):
    prompt: str = Field(..., max_length=2_000_000)
    options: dict[str, Any] | None = Field(None)


class BatchRequest(BaseModel):
    """Body for ``POST /api/batch``."""

    model: str = Field(..., min_length=1, max_length=256)
    requests: list[BatchItem] = Field(..., min_length=1, max_length=256)
    keep_alive: str | int | float | None = Field(None)


@router.post("/api/batch", response_model=None)
async def api_batch(
    req: BatchRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any] | JSONResponse:
    """Run every prompt in ``req.requests`` against ``req.model``.

    Model loads once for the whole batch. Individual failures don't
    abort the rest: they surface as ``{"error": "..."}`` entries at
    the matching index so clients can retry only the broken items.
    """
    unload_after = apply_keep_alive(req.model, req.keep_alive)
    try:
        await load_llm(req.model)
    except Exception:
        logger.exception("model load failed during /api/batch")
        return service_unavailable(f"Model '{req.model}' failed to load")

    state = get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{req.model}' failed to load")

    if unload_after:
        background_tasks.add_task(unload_after_response, req.model)

    results: list[dict[str, Any]] = []
    for item in req.requests:
        cfg = ollama_to_generation_config(item.options)
        try:
            gen_result = await run_dispatched(
                state.engine.generate,
                item.prompt,
                cfg,
                operation="batch.generate",
            )
        except (QueueFullError, QueueTimeoutError):
            # Don't leak ``str(exc)`` into the response envelope —
            # CodeQL py/stack-trace-exposure flags it, and the client
            # only needs the retry semantics anyway.
            logger.warning("batch item rejected by dispatcher (queue full or timeout)")
            results.append({"error": "server busy", "retryable": True})
            continue
        except Exception:
            logger.exception("batch item generate failed")
            results.append({"error": "internal generation error"})
            continue

        item_envelope: dict[str, Any] = {
            "response": gen_result.text,
            "done": True,
            "total_duration": gen_result.total_duration,
            "load_duration": gen_result.load_duration,
            "prompt_eval_count": gen_result.tokens_prompt,
            "prompt_eval_duration": gen_result.prompt_eval_duration,
            "eval_count": gen_result.tokens_generated,
            "eval_duration": gen_result.eval_duration,
        }
        results.append(item_envelope)

    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }


__all__ = ["router", "BatchRequest", "BatchItem"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``POST /api/stop`` — graceful unload without restart.

Ollama supports an undocumented convention where
``POST /api/chat`` with ``{"messages": [], "keep_alive": 0}`` evicts
the model without running inference. HFL's strict ``ChatRequest``
schema (``min_length=1`` on messages) rejects that shape; this
dedicated endpoint provides the same UX — idiomatic for HFL — while
keeping the chat schema's validation intact.

Semantics:

- ``model`` omitted or empty → evict the currently-resident LLM and
  TTS engines (same as SIGTERM handlers would).
- ``model`` set → evict only when it matches the resident model;
  otherwise no-op with ``{"status": "not_loaded"}``.

Always 200 to avoid brittle client error handling; clients key off
the ``status`` field to tell the three outcomes apart:
``stopped``, ``not_loaded``, ``nothing_loaded``.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from hfl.api.state import get_state

router = APIRouter(tags=["Ollama"])


class StopRequest(BaseModel):
    """Body for ``POST /api/stop``."""

    model: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Name of the model to unload. When omitted, every loaded model (LLM + TTS) is evicted."
        ),
    )


async def _unload_llm() -> None:
    """Background helper: unload the resident LLM.

    Routes through ``set_llm_engine(None, None)`` so the unload DRAINS in-flight
    inference (``dispatcher.exclusive()``) and respects the engine pin/unpin
    refcount, instead of freeing the shared non-reentrant model out from under
    a chat that is still running — an in-flight request holds a dispatcher slot,
    NOT ``_llm_lock``, so a raw lock+unload here is a use-after-free. (CON)
    """
    await get_state().set_llm_engine(None, None)


async def _unload_tts() -> None:
    """Background helper: unload the resident TTS engine (serialised via the
    TTS lock, mirroring ``set_tts_engine``)."""
    await get_state().set_tts_engine(None, None)


@router.post(
    "/api/stop",
    tags=["Ollama"],
    summary="Unload a loaded model",
    responses={
        200: {
            "description": (
                "One of: status=stopped (evicted), status=not_loaded "
                "(named model wasn't resident), status=nothing_loaded "
                "(no model at all)."
            ),
        }
    },
)
async def stop_model(
    req: StopRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str | None]:
    """Unload the named model (or all models) from memory.

    The actual ``unload()`` runs in a background task so the HTTP
    response is not gated on teardown time. For a multi-GB model on
    Metal that can be seconds of delay a caller shouldn't have to wait
    for.
    """
    state = get_state()

    # Clear any keep-alive deadlines for the target model so /api/ps
    # doesn't keep showing an expires_at after the unload task has run.
    if req.model:
        state.set_keep_alive_deadline(req.model, None)

    current = state.current_model
    current_tts = state.current_tts_model

    # No model target → evict both if loaded, report what happened.
    if not req.model:
        evicted: list[str] = []
        if current is not None:
            background_tasks.add_task(_unload_llm)
            evicted.append(current.name)
        if current_tts is not None:
            background_tasks.add_task(_unload_tts)
            evicted.append(current_tts.name)
        if not evicted:
            return {"status": "nothing_loaded", "model": None}
        return {"status": "stopped", "model": ",".join(evicted)}

    # Named model — match against LLM first, then TTS.
    if current is not None and current.name == req.model:
        background_tasks.add_task(_unload_llm)
        return {"status": "stopped", "model": req.model}
    if current_tts is not None and current_tts.name == req.model:
        background_tasks.add_task(_unload_tts)
        return {"status": "stopped", "model": req.model}

    return {"status": "not_loaded", "model": req.model}

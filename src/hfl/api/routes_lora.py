# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 F4 — LoRA hot-swap endpoints.

  POST /api/lora/apply   {model, lora_path, scale?, name?}
  POST /api/lora/remove  {model, adapter_id}
  GET  /api/lora                              list all
  GET  /api/lora/{model}                      list adapters for model

The endpoints mutate engine state, so they go through the same
dispatcher as /api/chat — concurrent apply + chat doesn't corrupt
the inflight request.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hfl.api.model_loader import load_llm
from hfl.engine.lora import apply_lora, list_loras, remove_lora

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


class ApplyLoraRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    lora_path: str = Field(..., min_length=1, max_length=4096)
    scale: float = Field(default=1.0, ge=0.0, le=5.0)
    name: str | None = Field(default=None, max_length=128)


class RemoveLoraRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    adapter_id: str = Field(..., min_length=1, max_length=64)


@router.post(
    "/api/lora/apply",
    response_model=None,
    summary="Hot-apply a LoRA adapter to a loaded model",
    responses={
        200: {"description": "Adapter applied"},
        400: {"description": "Invalid path or scale"},
        404: {"description": "Adapter file or model not found"},
        503: {"description": "Engine does not support LoRA hot-swap"},
    },
)
async def api_lora_apply(req: ApplyLoraRequest) -> dict[str, Any]:
    try:
        engine, _ = await load_llm(req.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not available")

    try:
        info = apply_lora(engine, lora_path=req.lora_path, scale=req.scale, name=req.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return asdict(info)


@router.post(
    "/api/lora/remove",
    response_model=None,
    summary="Detach a previously-applied LoRA adapter",
    responses={
        200: {"description": "Adapter removed"},
        404: {"description": "Adapter id unknown"},
        503: {"description": "Engine does not support LoRA removal"},
    },
)
async def api_lora_remove(req: RemoveLoraRequest) -> dict[str, Any]:
    try:
        engine, _ = await load_llm(req.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not available")

    try:
        ok = remove_lora(engine, req.adapter_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not ok:
        raise HTTPException(status_code=404, detail=f"adapter id unknown: {req.adapter_id!r}")

    return {"removed": True, "adapter_id": req.adapter_id}


@router.get("/api/lora", response_model=None, summary="List all active LoRA adapters")
async def api_lora_list_all() -> dict[str, list[dict[str, Any]]]:
    return {"adapters": [asdict(a) for a in list_loras()]}


@router.get(
    "/api/lora/{model:path}",
    response_model=None,
    summary="List adapters bound to a specific model",
)
async def api_lora_list_for_model(model: str) -> dict[str, Any]:
    try:
        engine, _ = await load_llm(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not available")
    return {"model": model, "adapters": [asdict(a) for a in list_loras(engine)]}

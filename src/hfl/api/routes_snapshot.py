# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 F6 — KV cache snapshot endpoints.

  POST /api/snapshot/save    body: {model, name}
  POST /api/snapshot/load    body: {model, name}
  GET  /api/snapshot         list all
  DELETE /api/snapshot/{name}

Snapshots are model-bound: load checks the sidecar's model name
matches before writing tensors, otherwise the operation aborts with
400 (loading a snapshot into the wrong model corrupts memory).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hfl.api.model_loader import load_llm
from hfl.engine.snapshot import (
    delete_snapshot,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


class SnapshotRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    name: str = Field(..., min_length=1, max_length=128)


@router.post(
    "/api/snapshot/save",
    response_model=None,
    summary="Save the KV cache state of a loaded model",
    responses={
        200: {"description": "Snapshot metadata"},
        400: {"description": "Invalid request"},
        404: {"description": "Model not loadable"},
        503: {"description": "Engine unavailable / snapshot unsupported"},
    },
)
async def api_snapshot_save(req: SnapshotRequest) -> dict[str, Any]:
    try:
        engine, _ = await load_llm(req.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not available")

    try:
        meta = save_snapshot(engine, name=req.name, model_name=req.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return asdict(meta)


@router.post(
    "/api/snapshot/load",
    response_model=None,
    summary="Restore a KV cache snapshot into a loaded model",
    responses={
        200: {"description": "Restored snapshot metadata"},
        400: {"description": "Snapshot belongs to a different model"},
        404: {"description": "Snapshot not found"},
        503: {"description": "Engine unavailable"},
    },
)
async def api_snapshot_load(req: SnapshotRequest) -> dict[str, Any]:
    try:
        engine, _ = await load_llm(req.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not available")

    try:
        meta = load_snapshot(engine, name=req.name, model_name=req.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return asdict(meta)


@router.get("/api/snapshot", response_model=None, summary="List all KV snapshots")
async def api_snapshot_list() -> dict[str, list[dict[str, Any]]]:
    return {"snapshots": [asdict(m) for m in list_snapshots()]}


@router.delete(
    "/api/snapshot/{name}",
    response_model=None,
    summary="Delete a KV snapshot",
    responses={
        200: {"description": "Snapshot deleted"},
        400: {"description": "Invalid name"},
        404: {"description": "Snapshot not found"},
    },
)
async def api_snapshot_delete(name: str) -> dict[str, Any]:
    try:
        deleted = delete_snapshot(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"snapshot {name!r} not found")
    return {"deleted": True, "name": name}

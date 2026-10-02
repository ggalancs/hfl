# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``POST /api/train``: train a LoRA adapter on a local model (Apple
Silicon, mlx-lm) and register the result — ``hfl train`` over HTTP, with its
progress streamed as NDJSON like ``/api/pull``.

The machine's owner only, on the machine itself: training takes the GPU
and memory for minutes to hours and reads a data file from this host's
disk, so not even ``HFL_ALLOW_REMOTE_PULL`` opens it to remote clients.
Closing the stream stops the run; ``resume`` continues from the adapter
saved so far.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["HFL"])


class TrainRequest(BaseModel):
    model: str = Field(..., max_length=256, description="The model to train on")
    data: str = Field(..., max_length=4096, description="A JSONL file or a folder, on the server")
    name: str | None = Field(None, max_length=64, description="The trained model's name")
    iters: int = Field(600, ge=1, le=1_000_000)
    batch_size: int = Field(4, ge=1, le=1024)
    num_layers: int = Field(16, ge=-1, le=1024)
    learning_rate: float = Field(1e-5, gt=0, lt=1)
    max_seq_length: int = Field(2048, ge=16, le=1_000_000)
    save_every: int = Field(100, ge=1, le=1_000_000)
    resume: bool = False
    stream: bool = True


def _line(event: dict[str, Any]) -> str:
    return json.dumps(event, separators=(",", ":")) + "\n"


@router.post(
    "/api/train",
    summary="Train a LoRA adapter on a local model (owner, on the host only)",
    response_model=None,
    responses={200: {"description": "NDJSON progress, or the final event with stream=false"}},
)
async def train_route(req: TrainRequest, request: Request) -> StreamingResponse | JSONResponse:
    from hfl.api.admin_guard import require_local_owner
    from hfl.config import config
    from hfl.core.container import get_registry
    from hfl.training import hf_lora
    from hfl.training import mlx_lora as trainer

    # MLX on Apple Silicon, Transformers + PEFT elsewhere (as `hfl train`).
    backend = None if trainer.available() is None else hf_lora

    require_local_owner(request, "train")

    def refuse(message: str, status: int = 400) -> JSONResponse:
        return JSONResponse({"error": message, "code": "train_refused"}, status_code=status)

    why_not = (backend or trainer).available()
    if why_not:
        return refuse(why_not, 501)
    base = get_registry().get(req.model)
    if base is None:
        return refuse(f"model not found: {req.model}", 404)
    name = req.name or f"{base.name}-lora"
    try:
        trainer.check_name(name)
        problem = trainer.trainable(base)
        if problem:
            raise trainer.TrainingError(problem)
        if get_registry().get(name) is not None and not req.resume:
            raise trainer.TrainingError(f"{name} already exists: another name, or resume")
    except trainer.TrainingError as exc:
        return refuse(exc.message)

    options = trainer.Options(
        iters=req.iters,
        batch_size=req.batch_size,
        num_layers=req.num_layers,
        learning_rate=req.learning_rate,
        max_seq_length=req.max_seq_length,
        save_every=req.save_every,
        resume=req.resume,
    )
    stop = threading.Event()
    events = trainer.events_of(
        base, name, Path(req.data), options, Path(config.home_dir), stop, backend=backend
    )

    async def stream() -> AsyncIterator[str]:
        try:
            while (event := await asyncio.to_thread(next, events, None)) is not None:
                yield _line(event)
        finally:
            stop.set()  # the client went away: stop mlx-lm, keep what it saved

    if req.stream:
        return StreamingResponse(stream(), media_type="application/x-ndjson")
    last: dict[str, Any] = {}
    async for line in stream():
        last = json.loads(line)
    status = 200 if last.get("status") == "success" else 400
    return JSONResponse(last, status_code=status)

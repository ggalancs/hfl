# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 ``POST /api/benchmark/{model}`` — TTFT + tok/s harness."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from hfl.api.model_loader import load_llm

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


class BenchmarkRequest(BaseModel):
    runs_per_length: int = Field(default=3, ge=1, le=20)
    max_tokens: int = Field(default=64, ge=1, le=2048)
    prompt_lengths: list[int] = Field(default_factory=lambda: [16, 256, 2048])
    stream: bool = Field(default=True)


async def _stream_events(model: str, req: BenchmarkRequest) -> AsyncIterator[str]:
    from hfl.engine.benchmark import run_benchmark_stream

    try:
        engine, _ = await load_llm(model)
    except FileNotFoundError as exc:
        yield json.dumps({"status": "failed", "error": str(exc)}) + "\n"
        return

    if engine is None:
        yield json.dumps({"status": "failed", "error": "engine not available"}) + "\n"
        return

    try:
        async for event in run_benchmark_stream(
            engine,
            model_name=model,
            runs_per_length=req.runs_per_length,
            max_tokens=req.max_tokens,
            prompt_lengths=tuple(req.prompt_lengths),
        ):
            yield json.dumps(event) + "\n"
    except Exception as exc:
        logger.exception("benchmark failed for %s", model)
        yield json.dumps({"status": "failed", "error": f"benchmark: {exc}"}) + "\n"


@router.post(
    "/api/benchmark/{model:path}",
    response_model=None,
    summary="Benchmark TTFT + tok/s on a registered model",
    responses={
        200: {"description": "Benchmark NDJSON stream or final summary"},
        400: {"description": "Bad request"},
        404: {"description": "Model not found"},
    },
)
async def api_benchmark(
    model: str,
    req: BenchmarkRequest | None = None,
) -> StreamingResponse | JSONResponse:
    """Stream NDJSON benchmark events for ``model``.

    Body (all optional):

    ```
    {
        "runs_per_length": 3,
        "max_tokens": 64,
        "prompt_lengths": [16, 256, 2048],
        "stream": true
    }
    ```
    """
    if req is None:
        req = BenchmarkRequest()

    if not req.stream:
        last: dict[str, Any] = {"status": "starting"}
        async for line in _stream_events(model, req):
            try:
                last = json.loads(line.strip())
            except (json.JSONDecodeError, ValueError):
                continue
        if last.get("status") == "failed":
            raise HTTPException(status_code=400, detail=last.get("error", "unknown"))
        return JSONResponse(content=last)

    return StreamingResponse(
        _stream_events(model, req),
        media_type="application/x-ndjson",
    )

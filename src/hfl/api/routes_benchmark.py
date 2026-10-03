# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 ``POST /api/benchmark/{model}`` — TTFT + tok/s harness."""

from __future__ import annotations

import json
import logging
from types import MethodType
from typing import Annotated, Any, AsyncIterator, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from hfl.api.helpers import run_dispatched
from hfl.api.model_loader import load_llm
from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError
from hfl.exceptions import ModelNotFoundError
from hfl.logging_config import log_internal_failure

logger = logging.getLogger(__name__)

router = APIRouter(tags=["HFL Beyond"])


# Every entry is runs_per_length generations of up to max_tokens: the list
# was unbounded, so one request could queue millions of them.
_MAX_PROMPT_LENGTHS = 8


class BenchmarkRequest(BaseModel):
    runs_per_length: int = Field(default=3, ge=1, le=20)
    max_tokens: int = Field(default=64, ge=1, le=2048)
    prompt_lengths: list[Annotated[int, Field(ge=1, le=1_000_000)]] = Field(
        default_factory=lambda: [16, 256, 2048],
        min_length=1,
        max_length=_MAX_PROMPT_LENGTHS,
    )
    stream: bool = Field(default=True)


def _boxed(engine: Any, fn: Callable[..., Any], *args: Any) -> tuple[Any]:
    # Boxed: a ``BenchmarkRun`` has ``tokens_generated``, and
    # ``run_dispatched`` would account it as a served generation with no
    # timings — a 0 ms sample in the server's latency metrics.
    return (fn(engine, *args),)


async def _dispatched(fn: Callable[..., Any], engine: Any, *args: Any) -> Any:
    """One measurement through the model's queue (bound to the engine so
    ``run_dispatched`` picks its dispatcher), never beside a reply."""
    (run,) = await run_dispatched(MethodType(_boxed, engine), fn, *args, operation="benchmark")
    return run


async def _stream_events(model: str, req: BenchmarkRequest) -> AsyncIterator[str]:
    from hfl.engine.benchmark import run_benchmark_stream

    try:
        engine, _ = await load_llm(model)
    except (ModelNotFoundError, FileNotFoundError):  # load_llm raises the former
        # With HFL_ALLOW_REMOTE_PULL the caller may be remote: name the
        # model they asked for, never the resolver's message — that one
        # spells out where on disk we looked.
        yield json.dumps({"status": "failed", "error": f"model not found: {model}"}) + "\n"
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
            run_call=_dispatched,
        ):
            yield json.dumps(event) + "\n"
    except (QueueFullError, QueueTimeoutError):
        yield json.dumps({"status": "failed", "error": "server busy, retry later"}) + "\n"
    except Exception as exc:
        detail = log_internal_failure(logger, "benchmark", exc)
        yield json.dumps({"status": "failed", "error": detail}) + "\n"


@router.post(
    "/api/benchmark/{model:path}",
    response_model=None,
    summary="Benchmark TTFT + tok/s on a registered model",
    responses={
        200: {"description": "Benchmark NDJSON stream or final summary"},
        400: {"description": "Bad request"},
        403: {"description": "Remote caller (owner-only diagnostic)"},
        404: {"description": "Model not found"},
        429: {"description": "Server busy"},
    },
)
async def api_benchmark(
    model: str,
    request: Request,
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
    from hfl.api.admin_guard import require_owner

    # Minutes of generation on demand: the owner's diagnostic, not a
    # user's way to keep the model busy.
    require_owner(request, "benchmark")
    if req is None:
        req = BenchmarkRequest()

    if not req.stream:
        # The figures are in the "summary" events; the last event ("done")
        # carries none, and it was all this answer returned (local audit B3).
        last: dict[str, Any] = {"status": "starting"}
        summaries: list[dict[str, Any]] = []
        async for line in _stream_events(model, req):
            try:
                last = json.loads(line.strip())
            except (json.JSONDecodeError, ValueError):
                continue
            if last.get("status") == "summary":
                summaries.append({k: v for k, v in last.items() if k != "status"})
        if last.get("status") == "failed":
            error = str(last.get("error", "unknown"))
            if error.startswith("model not found"):
                status = 404
            elif error.startswith("server busy"):
                status = 429
            else:
                status = 400
            raise HTTPException(status_code=status, detail=error)
        return JSONResponse(content={**last, "summaries": summaries})

    return StreamingResponse(
        _stream_events(model, req),
        media_type="application/x-ndjson",
    )

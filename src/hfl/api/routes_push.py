# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``POST /api/push`` endpoint.

Pushes a locally-registered model to a remote registry. HFL targets
the HuggingFace Hub here (Ollama uses its own registry; HFL leverages
the HF Hub we already use for ``pull``). NDJSON progress mirrors
``/api/pull`` so clients can render an upload bar with the same
parser they use for downloads.

Envelope sequence::

    {"status": "preparing", "total": <bytes>}
    {"status": "ensuring repository", "repo": "...", "private": ...}
    {"status": "uploading", "current": 0, "total": <bytes>}
    {"status": "success", "repo": "...", "revision": "...", "commit_url": "..."}

On failure the last event is::

    {"status": "failed", "error": "<short message>"}

Authentication: the request body may include a ``token``. When
absent, the server falls back to the standard ``HF_TOKEN`` env var
(read once at process start by ``hfl.config``); when both are
absent and the target repo is private, ``upload_folder`` raises and
the failure event is emitted.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ollama"])


class PushRequest(BaseModel):
    """Request body for ``POST /api/push``.

    ``model`` is the local registry name; ``destination`` is the
    target ``namespace/model`` on the Hub (Ollama clients pass it
    as ``name``, HFL accepts both spellings).
    """

    model: str = Field(..., min_length=1, max_length=256)
    destination: str | None = Field(default=None, max_length=512)
    name: str | None = Field(
        default=None,
        max_length=512,
        description="Alias for ``destination`` (Ollama clients use this name).",
    )
    revision: str | None = Field(default=None, max_length=128)
    private: bool = Field(default=False)
    token: str | None = Field(default=None, max_length=256)
    stream: bool = Field(default=True)


def _resolve_destination(req: PushRequest) -> str:
    """Pick the target repo from ``destination`` or ``name`` (Ollama)."""
    target = req.destination or req.name
    if not target:
        raise HTTPException(
            status_code=400,
            detail="push requires a 'destination' (or 'name') field",
        )
    return target


def _resolve_token(req_token: str | None) -> str | None:
    """Authentication token: explicit > config > env."""
    if req_token:
        return req_token
    try:
        from hfl.config import config as hfl_config

        if hfl_config.hf_token:
            return hfl_config.hf_token
    except Exception:  # pragma: no cover — config not loaded in some tests
        pass
    return None


def _build_api() -> Any:
    """Lazy import of HfApi so the route module imports cheaply."""
    from huggingface_hub import HfApi

    return HfApi()


async def _stream_ndjson(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        yield json.dumps(event) + "\n"


@router.post(
    "/api/push",
    response_model=None,
    tags=["Ollama"],
    summary="Push a local model to a remote registry",
    responses={
        200: {"description": "Push completed (or NDJSON stream of progress events)"},
        400: {"description": "Invalid request"},
        404: {"description": "Model not found in local registry"},
    },
)
async def api_push(req: PushRequest) -> StreamingResponse | JSONResponse:
    """Ollama-compatible ``POST /api/push``.

    Uploads ``req.model`` (a local registry entry) to the HuggingFace
    Hub at ``req.destination`` (or ``req.name``). Streams NDJSON
    progress events when ``stream=true`` (the Ollama default), or
    returns a single JSON envelope when ``stream=false``.
    """
    # Resolve the local manifest first — fail fast on a missing model.
    from hfl.core.container import get_registry
    from hfl.hub.uploader import build_upload_plan, stream_push

    manifest = get_registry().get(req.model)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"model not found in local registry: {req.model!r}",
        )

    target = _resolve_destination(req)
    token = _resolve_token(req.token)

    try:
        plan = build_upload_plan(
            manifest,
            target_repo_id=target,
            revision=req.revision,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    api = _build_api()
    events = stream_push(plan, api=api, private=req.private, token=token)

    if not req.stream:
        # Block until the iterator drains, return last event as JSON.
        last: dict[str, Any] = {"status": "preparing"}
        async for event in events:
            last = event
        return JSONResponse(content=last)

    return StreamingResponse(
        _stream_ndjson(events),
        media_type="application/x-ndjson",
    )

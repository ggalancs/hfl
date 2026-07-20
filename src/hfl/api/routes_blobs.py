# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``HEAD`` / ``POST /api/blobs/:digest`` endpoints.

Blobs are the transport layer for ``POST /api/create``: the client
uploads the raw bytes of a GGUF (or any other model file) to a
content-addressed slot and then references that slot from the
Modelfile ``FROM`` line. HFL would technically work without this
route (``FROM`` also accepts direct paths and model names), but
Ollama SDKs expect it and the ``ollama create`` CLI always uses it.

Contract (per https://docs.ollama.com/api#push-a-blob):

    HEAD /api/blobs/sha256:<hex>
        → 200 if the blob is already stored
        → 404 if not
        → 400 if the digest is malformed

    POST /api/blobs/sha256:<hex>
        body: raw bytes of the blob (any length, streamed)
        → 201 on successful ingest
        → 400 if the computed digest does not match the path
        → 400 if the digest is malformed
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, Response

from hfl.config import config
from hfl.hub.blobs import (
    DigestMismatchError,
    InvalidBlobDigestError,
    blob_exists,
    parse_digest,
    write_blob_stream,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ollama"])


def _normalize_digest(path_digest: str) -> str:
    """FastAPI hands over the path param with the ``sha256:`` prefix.

    We strip it here so the error message (if any) names the exact
    input the client sent rather than the bare hex form.
    """
    return path_digest


async def _request_chunks(request: Request) -> AsyncIterator[bytes]:
    """Adapt ``Request.stream()`` into a plain async byte iterator."""
    async for chunk in request.stream():
        if chunk:
            yield chunk


@router.head("/api/blobs/{digest:path}")
async def head_blob(digest: str) -> Response:
    """Return 200 if the named blob exists locally, 404 otherwise.

    Malformed digests surface as 400 so clients can distinguish a
    missing blob (404 — upload it) from a request they cannot
    recover from by retrying (400 — fix the digest).
    """
    try:
        parse_digest(digest)
    except InvalidBlobDigestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not blob_exists(digest):
        return Response(status_code=404)
    return Response(status_code=200)


@router.post("/api/blobs/{digest:path}")
async def post_blob(digest: str, request: Request) -> Response:
    """Store the request body at the content-addressed path.

    The stream is hashed as it arrives and the temp file is promoted
    to its final location only if the running SHA-256 equals
    ``digest``. Bodies of any size are accepted; the global body-size
    limit middleware must exempt ``/api/blobs/`` (see
    ``RequestBodyLimitMiddleware.EXCLUDED_PREFIXES``).
    """
    try:
        # SEC-2: bound the upload when an operator has configured a cap;
        # the default (0 → None) preserves the unlimited GGUF create flow.
        bytes_written = await write_blob_stream(
            digest, _request_chunks(request), chunk_limit=config.max_blob_bytes or None
        )
    except InvalidBlobDigestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DigestMismatchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("blob upload failed")
        raise HTTPException(status_code=500, detail="blob upload failed") from None

    return Response(
        status_code=201,
        headers={"X-Blob-Bytes": str(bytes_written)},
    )

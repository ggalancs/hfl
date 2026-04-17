# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Embedding endpoints — Ollama-native + OpenAI-compatible (P0-1).

Three routes, three shapes, one backend:

- ``POST /api/embed`` (Ollama preferred): ``{model, input, truncate,
  options, keep_alive, dimensions}`` → ``{embeddings: [[...], ...],
  total_duration, load_duration, prompt_eval_count}``.
- ``POST /api/embeddings`` (Ollama legacy alias): ``{model, prompt,
  options, keep_alive}`` → ``{embedding: [...]}`` (single vector).
- ``POST /v1/embeddings`` (OpenAI): ``{model, input, encoding_format,
  dimensions, user}`` → ``{object: "list", data: [{object:
  "embedding", embedding: [...], index: 0}], model, usage:
  {prompt_tokens, total_tokens}}``.

Embeddings are stateless and batchable — the router doesn't route
them through the LLM dispatcher (which serialises at max_inflight=1
for chat). Instead requests run on the default thread pool via
``asyncio.to_thread`` so a dozen concurrent RAG queries don't block
each other.

Model loading uses a dedicated path from :mod:`hfl.api.model_loader`
so a future embed-dedicated pool can be plugged in without breaking
LLM loading.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import struct
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from hfl.api.helpers import apply_keep_alive
from hfl.exceptions import (
    ModelNotFoundError,
    ModelNotReadyError,
    ModelTypeMismatchError,
)
from hfl.exceptions import ValidationError as APIValidationError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Embeddings"])


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------


class OllamaEmbedRequest(BaseModel):
    """Body for ``POST /api/embed`` (Ollama preferred)."""

    model: str = Field(..., min_length=1, max_length=256)
    input: str | list[str] = Field(
        ...,
        description="Single string or list of strings to embed.",
    )
    truncate: bool = Field(
        True,
        description=(
            "When True (default), inputs longer than the model's "
            "context are truncated to fit; when False, oversized "
            "inputs raise 400."
        ),
    )
    options: dict[str, Any] | None = Field(None, description="Backend-specific options.")
    keep_alive: str | int | float | None = Field(None)
    dimensions: int | None = Field(
        None,
        ge=1,
        le=8192,
        description=(
            "Matryoshka-style truncation of output vectors. Values "
            "greater than the model's native dimension → 400."
        ),
    )

    @field_validator("input")
    @classmethod
    def _bound_input(cls, v: str | list[str]) -> str | list[str]:
        """Bound input size to prevent DoS via gigantic batches."""
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("input list must be non-empty")
            if len(v) > 1024:
                raise ValueError("input list must contain at most 1024 strings")
            for i, s in enumerate(v):
                if not isinstance(s, str):
                    raise ValueError(f"input[{i}] must be a string")
                if len(s) > 2_000_000:
                    raise ValueError(f"input[{i}] exceeds 2_000_000 characters")
        else:
            if len(v) > 2_000_000:
                raise ValueError("input exceeds 2_000_000 characters")
        return v


class OllamaEmbeddingsLegacyRequest(BaseModel):
    """Body for the legacy ``POST /api/embeddings`` alias."""

    model: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., max_length=2_000_000)
    options: dict[str, Any] | None = Field(None)
    keep_alive: str | int | float | None = Field(None)


class OpenAIEmbeddingsRequest(BaseModel):
    """Body for ``POST /v1/embeddings`` (OpenAI-compatible)."""

    model: str = Field(..., min_length=1, max_length=256)
    input: str | list[str] | list[int] | list[list[int]] = Field(
        ...,
        description=(
            "Single string, list of strings, or pre-tokenised list(s) "
            "of integers. HFL accepts strings and decodes token lists "
            "by joining them as space-separated fallbacks (not lossless)."
        ),
    )
    encoding_format: str = Field(
        "float",
        description="Either 'float' (default) or 'base64'.",
    )
    dimensions: int | None = Field(None, ge=1, le=8192)
    user: str | None = Field(None, max_length=256)

    @field_validator("encoding_format")
    @classmethod
    def _allowed_encoding(cls, v: str) -> str:
        if v not in ("float", "base64"):
            raise ValueError('encoding_format must be "float" or "base64"')
        return v


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _normalize_input(raw: str | list[str] | list[int] | list[list[int]]) -> list[str]:
    """Coerce every OpenAI input variant into ``list[str]``.

    Token-list inputs (``[1, 2, 3]`` or ``[[1,2],[3,4]]``) are
    decoded into space-separated strings — lossy, but matches what
    LangChain/Python-OpenAI do when they fall back from tokens to
    strings. A real token-level detokenizer would need access to
    the specific model's tokeniser; callers that care should pass
    strings directly.
    """
    if isinstance(raw, str):
        return [raw]
    if not raw:
        raise APIValidationError("input must be non-empty")
    first = raw[0]
    if isinstance(first, str):
        return list(raw)  # type: ignore[arg-type]
    if isinstance(first, int):
        return [" ".join(str(x) for x in raw)]
    if isinstance(first, list):
        return [
            " ".join(str(x) for x in tokens)  # type: ignore[union-attr]
            for tokens in raw
        ]
    raise APIValidationError("input has an unsupported shape")


def _encode_base64_vectors(vectors: list[list[float]]) -> list[str]:
    """Encode each vector as OpenAI-style base64 of little-endian floats."""
    out: list[str] = []
    for vec in vectors:
        packed = struct.pack(f"<{len(vec)}f", *vec)
        out.append(base64.b64encode(packed).decode("ascii"))
    return out


async def _load_embedding_model(model_name: str) -> Any:
    """Locate + load an embedding model via the registry.

    Uses the LLM loader's registry lookup but points the loader at
    :class:`LlamaCppEmbeddingEngine` (via :func:`select_embedding_engine`).
    Returns the loaded engine object.

    Kept inline in the route module to avoid extending the general
    ``model_loader`` with an embed-specific branch for this first cut.
    The full embed-pool integration tracked in OLLAMA_PARITY_PLAN P0-1
    subtask 6 will replace this.
    """
    from pathlib import Path

    from hfl.api.state import get_state
    from hfl.converter.formats import ModelType, detect_model_type
    from hfl.models.registry import get_registry
    from hfl.validators import ValidationError, validate_model_name

    try:
        validate_model_name(model_name)
    except ValidationError as e:
        raise APIValidationError(str(e))

    state = get_state()

    # Fast path — if we already have this model resident AS an embed
    # engine, reuse it. HFL doesn't yet have a dedicated embed slot
    # so we piggyback on state.engine when it happens to be an
    # embedding engine.
    existing = getattr(state, "_embed_engine", None)
    existing_name = getattr(state, "_embed_model_name", None)
    if existing is not None and existing_name == model_name:
        if not existing.is_loaded:
            raise ModelNotReadyError(model_name)
        return existing

    manifest = get_registry().get(model_name)
    if manifest is None:
        raise ModelNotFoundError(model_name)

    model_path = Path(manifest.local_path)
    detected = detect_model_type(model_path)
    if detected != ModelType.EMBEDDING:
        raise ModelTypeMismatchError(model_name, expected="embedding", got=detected.value)

    engine = _select_embedding_backend(model_path)
    await asyncio.to_thread(engine.load, manifest.local_path)
    # Stash on state so the next call reuses.
    state._embed_engine = engine  # type: ignore[attr-defined]
    state._embed_model_name = model_name  # type: ignore[attr-defined]
    return engine


def _select_embedding_backend(model_path: Any) -> Any:
    """Pick the right embedding engine for a given model path.

    GGUF → LlamaCpp; directory → Transformers. Mirrors
    ``hfl.engine.selector.select_engine`` for LLMs.
    """
    from hfl.converter.formats import ModelFormat, detect_format
    from hfl.engine.embedding_engine import (
        LlamaCppEmbeddingEngine,
        TransformersEmbeddingEngine,
    )

    fmt = detect_format(model_path)
    if fmt == ModelFormat.GGUF:
        return LlamaCppEmbeddingEngine()
    return TransformersEmbeddingEngine()


# ----------------------------------------------------------------------
# Ollama: POST /api/embed
# ----------------------------------------------------------------------


@router.post(
    "/api/embed",
    tags=["Ollama"],
    summary="Generate embeddings (Ollama)",
    response_model=None,
    responses={
        200: {"description": "Embeddings for each input string."},
        400: {"description": "Validation / type mismatch."},
        404: {"description": "Model not found."},
    },
)
async def ollama_embed(req: OllamaEmbedRequest) -> dict[str, Any]:
    """Ollama-compatible ``POST /api/embed``."""
    start = time.monotonic_ns()
    apply_keep_alive(req.model, req.keep_alive)

    load_start = time.monotonic_ns()
    engine = await _load_embedding_model(req.model)
    load_duration = time.monotonic_ns() - load_start

    inputs = req.input if isinstance(req.input, list) else [req.input]

    result = await asyncio.to_thread(
        engine.embed,
        inputs,
        truncate=req.truncate,
        dimensions=req.dimensions,
    )

    total_duration = time.monotonic_ns() - start

    return {
        "model": req.model,
        "embeddings": result.embeddings,
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": result.total_tokens,
    }


# ----------------------------------------------------------------------
# Ollama: POST /api/embeddings (legacy)
# ----------------------------------------------------------------------


@router.post(
    "/api/embeddings",
    tags=["Ollama"],
    summary="Generate embedding (legacy Ollama alias)",
    response_model=None,
    responses={
        200: {"description": "Single embedding vector for the provided prompt."},
        400: {"description": "Validation / type mismatch."},
        404: {"description": "Model not found."},
    },
)
async def ollama_embeddings_legacy(req: OllamaEmbeddingsLegacyRequest) -> dict[str, Any]:
    """Legacy ``POST /api/embeddings`` — single prompt, single vector.

    Superseded by ``/api/embed`` but kept for client compatibility
    (older ollama-python releases, some LangChain versions).
    """
    apply_keep_alive(req.model, req.keep_alive)
    engine = await _load_embedding_model(req.model)
    result = await asyncio.to_thread(engine.embed, [req.prompt])
    return {"embedding": result.embeddings[0]}


# ----------------------------------------------------------------------
# OpenAI: POST /v1/embeddings
# ----------------------------------------------------------------------


@router.post(
    "/v1/embeddings",
    tags=["OpenAI"],
    summary="Create embeddings (OpenAI-compatible)",
    response_model=None,
    responses={
        200: {"description": "List of embeddings in OpenAI envelope."},
        400: {"description": "Validation / type mismatch."},
        404: {"description": "Model not found."},
    },
)
async def openai_embeddings(req: OpenAIEmbeddingsRequest) -> dict[str, Any]:
    """OpenAI-compatible ``POST /v1/embeddings``.

    Envelope matches the OpenAI spec byte-for-byte so clients using
    the official SDK (or ``langchain_openai.OpenAIEmbeddings``) work
    without patching.
    """
    engine = await _load_embedding_model(req.model)
    inputs = _normalize_input(req.input)
    result = await asyncio.to_thread(
        engine.embed,
        inputs,
        truncate=True,
        dimensions=req.dimensions,
    )

    if req.encoding_format == "base64":
        encoded = _encode_base64_vectors(result.embeddings)
        data = [
            {"object": "embedding", "embedding": enc, "index": i} for i, enc in enumerate(encoded)
        ]
    else:
        data = [
            {"object": "embedding", "embedding": vec, "index": i}
            for i, vec in enumerate(result.embeddings)
        ]

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": result.total_tokens,
            "total_tokens": result.total_tokens,
        },
    }

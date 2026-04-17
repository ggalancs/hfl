# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``POST /api/create`` endpoint.

Creates a derived model from a Modelfile. The response is an NDJSON
progress stream that matches Ollama's byte-for-byte so SDKs that key
off the ``status`` strings (``parsing modelfile``,
``using existing layer``, ``creating model``, ``success``) render the
expected progress bars.

Contract (per https://docs.ollama.com/api#create-a-model):

    POST /api/create
    {
        "model": "my-coder",
        "modelfile": "FROM llama3.3\\nSYSTEM \\"...\\"",
        "stream": true,
        "files": {"llama.gguf": "sha256:..."}
    }

    → 200 application/x-ndjson
    {"status": "parsing modelfile"}
    {"status": "using existing layer sha256:<digest>"}
    {"status": "creating model"}
    {"status": "writing manifest"}
    {"status": "success"}

    or (``stream=false``):
    → 200 application/json
    {"status": "success"}

FROM resolution priority:

    1. ``sha256:<hex>``  → content-addressed blob (must be uploaded
       first via ``POST /api/blobs/:digest``).
    2. Existing model name / alias → clone the manifest.
    3. Absolute path to a local file → use it in place.

Any other shape is rejected with 400. This is a superset of what the
Ollama server does (it always goes through a registry pull); the
registry path is HFL's Modelfile-ingestion analogue.

Errors surface as Ollama-shaped JSON envelopes with ``{"error":
"…"}`` so ``ollama-python`` can display them verbatim.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from hfl.converter.modelfile_parser import (
    ModelfileDocument,
    ModelfileMessage,
    ModelfileParseError,
    parse_modelfile,
)
from hfl.converter.requires_check import (
    InvalidRequiresError,
    RequiresNotSatisfiedError,
    check_requires,
)
from hfl.hub.blobs import blob_exists, blob_path, parse_digest
from hfl.models.manifest import ModelManifest
from hfl.models.registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ollama"])


class CreateRequest(BaseModel):
    """Body for ``POST /api/create``.

    Either ``modelfile`` (the full Modelfile text) **or** the
    structured fields (``from_``, ``system``, ``template``, etc.)
    must be supplied. Passing both forms is allowed: the structured
    fields override whatever was parsed out of ``modelfile``.
    """

    model: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Destination model name for the new manifest.",
    )
    modelfile: str | None = Field(
        None,
        description=("Full Modelfile text. Alternative to the structured fields."),
    )
    # Structured Modelfile fields (Ollama newer shape).
    from_: str | None = Field(
        None,
        alias="from",
        description="FROM source: model name, blob digest, or local path.",
    )
    system: str | None = None
    template: str | None = None
    license: str | None = None
    parameters: dict[str, Any] | None = None
    adapters: list[str] | None = None
    messages: list[dict[str, Any]] | None = None
    files: dict[str, str] | None = Field(
        None,
        description=(
            "Optional map of ``<modelfile-reference>`` → ``sha256:<digest>`` "
            "for blob-backed FROM/ADAPTER values."
        ),
    )
    stream: bool = Field(
        True,
        description="Stream progress as NDJSON (default).",
    )


def _event(status: str, **extra: Any) -> str:
    """Emit a single NDJSON progress chunk."""
    payload: dict[str, Any] = {"status": status}
    payload.update(extra)
    return json.dumps(payload, separators=(",", ":")) + "\n"


def _error_event(message: str) -> str:
    """Emit an Ollama-shaped error envelope."""
    return json.dumps({"error": message}, separators=(",", ":")) + "\n"


def _merge_structured_fields(
    doc: ModelfileDocument,
    req: CreateRequest,
) -> ModelfileDocument:
    """Apply ``req``'s structured overrides on top of ``doc`` in-place."""
    if req.from_:
        doc.from_ = req.from_
    if req.system is not None:
        doc.system = req.system
    if req.template is not None:
        doc.template = req.template
    if req.license is not None:
        doc.license = req.license
    if req.parameters:
        for key, value in req.parameters.items():
            k = key.lower()
            if k == "stop":
                if isinstance(value, list):
                    doc.stop_sequences.extend(str(s) for s in value)
                else:
                    doc.stop_sequences.append(str(value))
            else:
                doc.parameters[k] = value
    if req.adapters:
        doc.adapters.extend(req.adapters)
    if req.messages:
        for msg in req.messages:
            role = msg.get("role") or "user"
            content = msg.get("content") or ""
            doc.messages.append(ModelfileMessage(role=str(role), content=str(content)))
    return doc


def _resolve_from(
    source: str,
    files: dict[str, str] | None,
) -> tuple[str, str | None, str | None]:
    """Resolve a FROM value to ``(local_path, parent_name, parent_digest)``.

    ``parent_name`` is set when FROM references an existing manifest.
    ``parent_digest`` is set when FROM references a blob or when the
    source model has a recorded hash.
    """
    files = files or {}
    # Indirect ref via ``files``: ``FROM llama.gguf`` + files mapping.
    if source in files:
        source = files[source]

    # Blob digest?
    if source.lower().startswith("sha256:") or source.lower().startswith("sha256-"):
        try:
            digest = parse_digest(source)
        except ValueError as exc:
            # Don't interpolate ``exc`` into the user message — CodeQL
            # treats its repr as a taint source for
            # ``py/stack-trace-exposure`` and the parse_digest error
            # already carries enough information on its own.
            raise ValueError("invalid blob digest in FROM") from exc
        if not blob_exists(digest):
            raise ValueError(
                f"FROM references unknown blob sha256:{digest[:12]}… "
                "— upload it first via POST /api/blobs/:digest"
            )
        return str(blob_path(digest)), None, f"sha256:{digest}"

    # Absolute or relative file path?
    path = Path(source)
    if path.exists() and path.is_file():
        return str(path.resolve()), None, None

    # Existing model name / alias?
    registry = get_registry()
    existing = registry.get(source)
    if existing is not None:
        return existing.local_path, existing.name, existing.file_hash

    raise ValueError(f"FROM {source!r} is not a known model, a local file, or an uploaded blob")


async def _create_generator(req: CreateRequest) -> AsyncIterator[str]:
    """Run the create pipeline and yield NDJSON events."""
    # 1. Build / parse the Modelfile.
    yield _event("parsing modelfile")
    try:
        if req.modelfile:
            doc = parse_modelfile(req.modelfile)
        else:
            doc = ModelfileDocument(from_=req.from_ or "")
        doc = _merge_structured_fields(doc, req)
        if not doc.from_:
            raise ModelfileParseError("Modelfile missing FROM instruction")
    except ModelfileParseError as exc:
        yield _error_event(str(exc))
        return

    # 1b. Honour REQUIRES (Phase 7 P3-3). A Modelfile with a
    # version gate is rejected before any side-effects fire.
    if doc.requires:
        try:
            check_requires(doc.requires)
        except (RequiresNotSatisfiedError, InvalidRequiresError) as exc:
            yield _error_event(str(exc))
            return

    # 2. Resolve FROM.
    try:
        local_path, parent_name, parent_digest = await asyncio.to_thread(
            _resolve_from,
            doc.from_,
            req.files,
        )
    except ValueError as exc:
        yield _error_event(str(exc))
        return

    if parent_digest:
        yield _event(f"using existing layer {parent_digest}")

    # 3. Build the manifest.
    yield _event("creating model")
    manifest_fields = doc.to_manifest_fields()
    manifest = ModelManifest(
        name=req.model,
        repo_id=parent_name or req.model,
        local_path=local_path,
        format="gguf" if local_path.lower().endswith(".gguf") else "unknown",
        parent_name=parent_name,
        parent_digest=parent_digest,
        **manifest_fields,
    )
    # If the Modelfile carries MESSAGE instructions, persist them as
    # dicts (the manifest stores ``messages: list[dict]``; the parser
    # exposes dataclasses for type safety).
    if doc.messages:
        manifest.messages = [{"role": m.role, "content": m.content} for m in doc.messages]

    # 4. Persist to registry.
    yield _event("writing manifest")
    try:
        await asyncio.to_thread(get_registry().add, manifest)
    except Exception:
        # Swallow the exception text: ``exc`` may reveal internal
        # paths or library class names (CodeQL
        # ``py/stack-trace-exposure``). The full traceback is in the
        # server log via ``logger.exception``; the client just sees
        # a generic failure event.
        logger.exception("registry add failed during /api/create")
        yield _error_event("failed to persist manifest")
        return

    # 5. Done.
    yield _event("success")


async def _collect_final(gen: AsyncIterator[str]) -> dict[str, Any]:
    """Drain ``gen`` for non-streaming mode; surface the last event."""
    final: dict[str, Any] = {"status": "success"}
    async for line in gen:
        try:
            final = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in final:
            return final
    return final


@router.post("/api/create")
async def api_create(req: CreateRequest) -> Any:
    """Create a new model from a Modelfile.

    Streaming is the default (Ollama parity); pass ``stream=false`` to
    get a single JSON envelope after the pipeline finishes. Errors
    during the pipeline are emitted inline as ``{"error": "..."}``
    events on the stream (HTTP 200) so the connection semantics match
    Ollama exactly.
    """
    generator = _create_generator(req)
    if req.stream:
        return StreamingResponse(
            generator,
            media_type="application/x-ndjson",
        )
    final = await _collect_final(generator)
    status_code = 400 if "error" in final else 200
    return JSONResponse(status_code=status_code, content=final)


__all__ = ["router", "CreateRequest"]

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``POST /api/copy`` endpoint (Phase 5, P1-2).

Creates a second registry entry pointing at the same on-disk blob
as the source. The copy is disk-free — a rename that leaves the
original in place, rather than a byte duplication.

Ollama's contract is strict about status codes:

- ``200 OK`` on success (empty body accepted by clients).
- ``404 Not Found`` when the source model doesn't exist.
- ``400 Bad Request`` when the destination is already taken or
  malformed (HFL maps validation errors from the registry here).

Shape reference: https://docs.ollama.com/api#copy-a-model
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from hfl.exceptions import (
    ModelAlreadyExistsError,
    ModelNotFoundError,
)
from hfl.exceptions import ValidationError as APIValidationError
from hfl.models.registry import get_registry

router = APIRouter(tags=["Ollama"])


class CopyRequest(BaseModel):
    """Body for ``POST /api/copy``."""

    source: str = Field(..., min_length=1, max_length=256)
    destination: str = Field(..., min_length=1, max_length=256)


@router.post(
    "/api/copy",
    tags=["Ollama"],
    summary="Copy a model (Ollama-compatible)",
    response_model=None,
    responses={
        200: {"description": "Copy succeeded (empty body)."},
        400: {"description": "Destination already in use or malformed."},
        404: {"description": "Source model not found."},
    },
)
async def copy_model(req: CopyRequest) -> dict[str, str]:
    """Ollama-compatible ``POST /api/copy``."""
    from hfl.validators import ValidationError as RegistryValidationError
    from hfl.validators import validate_model_name

    # Up-front format validation gives a cleaner 400 message than
    # waiting for the registry to raise.
    try:
        validate_model_name(req.destination)
    except RegistryValidationError as exc:
        raise APIValidationError(f"destination: {exc}")

    registry = get_registry()
    if registry.get(req.source) is None:
        raise ModelNotFoundError(req.source)

    if registry.get(req.destination) is not None:
        raise ModelAlreadyExistsError(req.destination)

    # ``copy`` returns False on destination conflict (we've already
    # checked that above, but the registry is authoritative under
    # concurrent writes).
    if not registry.copy(req.source, req.destination):
        raise ModelAlreadyExistsError(req.destination)

    return {"status": "copied", "source": req.source, "destination": req.destination}

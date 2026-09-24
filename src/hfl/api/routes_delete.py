# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``DELETE /api/delete``.

Left out for a long time so the API could not destroy local models; now
Ollama clients (Open WebUI's model manager, ``ollama.delete``) can use
it, under the strictest guard HFL has: the caller must be on the server
host itself (loopback), no remote exception, and never a web page.

Same rule as ``hfl rm`` (:mod:`hfl.models.removal`): only files inside
HFL's models folder are deleted; a loaded model is unloaded first.

Shape reference: https://docs.ollama.com/api#delete-a-model
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, model_validator

from hfl.exceptions import ModelNotFoundError
from hfl.models.registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ollama"])


class DeleteRequest(BaseModel):
    """Body for ``DELETE /api/delete`` (``name`` is Ollama's older field)."""

    model: str = Field(default="", max_length=256)
    name: str = Field(default="", max_length=256)

    @model_validator(mode="after")
    def _one_name(self) -> "DeleteRequest":
        self.model = self.model or self.name
        if not self.model:
            raise ValueError("model is required")
        return self


@router.delete(
    "/api/delete",
    tags=["Ollama"],
    summary="Delete a model (Ollama-compatible, owner on the host only)",
    response_model=None,
    responses={
        200: {"description": "Deleted."},
        403: {"description": "Not a loopback caller, or called from a web page."},
        404: {"description": "Model not found."},
    },
)
async def delete_model(req: DeleteRequest, request: Request) -> dict[str, str]:
    """Ollama-compatible ``DELETE /api/delete``."""
    from hfl.api.admin_guard import require_local_owner
    from hfl.api.state import get_state
    from hfl.models.removal import remove_model

    require_local_owner(request, "delete")
    registry = get_registry()
    manifest = registry.get(req.model)
    if manifest is None:
        raise ModelNotFoundError(req.model)
    await get_state().evict(manifest.name, reason="deleted")
    result = remove_model(registry, manifest)
    if result.kept_outside is not None:
        logger.info("deleted %s; its file is outside HFL's models folder and was kept", result.name)
    elif result.shared_with:
        logger.info("deleted %s; its file is still used by %s", result.name, result.shared_with)
    else:
        logger.info("deleted %s", result.name)
    return {"status": "success"}

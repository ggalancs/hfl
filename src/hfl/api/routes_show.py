# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``POST /api/show`` endpoint.

Returns the full introspection payload Ollama clients depend on to
render model cards, capability chips and "model info" panels:

- ``modelfile``: the rendered Modelfile text (see
  :mod:`hfl.converter.modelfile`).
- ``parameters``: the PARAMETER block in Ollama's multi-line format.
- ``template``: the chat template (Jinja2 string).
- ``details``: format / family / parameter_size / quantization_level.
- ``model_info``: a dict of architecture-level numeric facts. Keys
  follow the GGUF convention (``<arch>.context_length``, etc.) because
  that's what ``ollama-python``'s typed wrapper looks for.
- ``capabilities``: the list produced by
  :func:`hfl.models.capabilities.detect_capabilities`.

Shape reference: https://docs.ollama.com/api#show-model-information
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from hfl.converter.modelfile import render_modelfile
from hfl.exceptions import ModelNotFoundError
from hfl.models.capabilities import detect_capabilities
from hfl.models.registry import get_registry

router = APIRouter(tags=["Ollama"])


class ShowRequest(BaseModel):
    """Body for ``POST /api/show``."""

    model: str = Field(..., min_length=1, max_length=256, description="Model name")
    verbose: bool = Field(
        False,
        description="When true, includes full tensor info in model_info (larger response).",
    )


def _format_parameters(manifest: Any) -> str:
    """Render the PARAMETER block in Ollama's text format.

    Ollama returns something like ::

        temperature 0.7
        stop "<|im_end|>"
        num_ctx 4096

    one per line, which is what their CLI's ``ollama show
    --parameters`` surfaces. We reuse the logic from ``render_modelfile``
    but strip the PARAMETER prefix since the top-level ``parameters``
    field of ``/api/show`` omits the keyword.
    """
    lines: list[str] = []
    if manifest.context_length and manifest.context_length > 0:
        lines.append(f"num_ctx {manifest.context_length}")

    defaults = getattr(manifest, "default_parameters", None) or {}
    for key in sorted(defaults.keys()):
        value = defaults[key]
        if key == "stop":
            values = value if isinstance(value, (list, tuple)) else [value]
            for s in values:
                # Quote stop strings for readability.
                escaped = str(s).replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'stop "{escaped}"')
        else:
            lines.append(f"{key} {value}")
    return "\n".join(lines)


def _model_info(manifest: Any, verbose: bool) -> dict[str, Any]:
    """Build the ``model_info`` dict.

    Ollama's ``model_info`` is a loose bag of GGUF metadata:
    ``general.architecture``, ``general.parameter_count``,
    ``<arch>.context_length``, ``<arch>.embedding_length``, etc. HFL
    doesn't currently parse all of GGUF's metadata into the manifest,
    so we surface the fields we do track, keyed in the same style.
    Unknown fields are simply absent — ``ollama-python`` treats
    missing keys as ``None``.

    ``verbose=True`` is a pass-through today (we don't store tensor
    statistics). The flag is preserved for forward compatibility so
    future callers can opt in without a schema change.
    """
    info: dict[str, Any] = {}
    arch = manifest.architecture or "unknown"
    info["general.architecture"] = arch
    if manifest.parameters:
        info["general.parameter_count"] = manifest.parameters
    if manifest.quantization:
        info["general.quantization"] = manifest.quantization
    if manifest.size_bytes:
        info["general.size"] = manifest.size_bytes
    if manifest.context_length and manifest.context_length > 0:
        info[f"{arch}.context_length"] = manifest.context_length
    if manifest.file_hash:
        info["general.digest"] = manifest.file_hash

    # Verbose-only extras — stubbed for now, but the key is reserved.
    if verbose:
        info["hfl.verbose"] = True

    return info


def _details(manifest: Any) -> dict[str, Any]:
    """The ``details`` sub-object shared with ``/api/ps``.

    Kept in sync with ``routes_ps._manifest_details`` manually (the
    schemas are intentionally identical per Ollama's contract).
    """
    return {
        "format": manifest.format or "unknown",
        "family": manifest.architecture or "unknown",
        "families": [manifest.architecture] if manifest.architecture else None,
        "parameter_size": manifest.parameters,
        "quantization_level": manifest.quantization,
    }


@router.post(
    "/api/show",
    tags=["Ollama"],
    summary="Show model information",
    responses={
        200: {"description": "Model details with capabilities, modelfile and parameters."},
        404: {"description": "Model not found in the local registry."},
    },
)
async def show_model(req: ShowRequest) -> dict[str, Any]:
    """Ollama-compatible ``POST /api/show``.

    Looks up the registry entry for ``req.model`` and returns the
    rendered Modelfile, chat template, parameters, details,
    architecture info and capabilities list.
    """
    manifest = get_registry().get(req.model)
    if manifest is None:
        raise ModelNotFoundError(req.model)

    return {
        "modelfile": render_modelfile(manifest),
        "parameters": _format_parameters(manifest),
        "template": manifest.chat_template or "",
        "details": _details(manifest),
        "model_info": _model_info(manifest, req.verbose),
        "capabilities": detect_capabilities(manifest),
        "license": manifest.license_name or manifest.license or "",
    }

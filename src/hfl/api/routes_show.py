# SPDX-License-Identifier: Apache-2.0
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

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from hfl.api.state import get_state
from hfl.converter.modelfile import render_modelfile
from hfl.exceptions import ModelNotFoundError
from hfl.models.capabilities import detect_capabilities
from hfl.models.chat_template import model_template
from hfl.models.file_facts import architecture, file_facts
from hfl.models.registry import get_registry
from hfl.models.thinking import thinking_controls

router = APIRouter(tags=["Ollama"])

# ``file_facts`` keys after the context, and their GGUF names under ``<arch>.``.
_LAYOUT_KEYS = (
    ("embedding_length", "embedding_length"),
    ("block_count", "block_count"),
    ("head_count", "attention.head_count"),
    ("head_count_kv", "attention.head_count_kv"),
)


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
    ``<arch>.context_length``, ``<arch>.embedding_length``, etc. The
    architecture and layout come from the model's own files
    (:func:`hfl.models.file_facts.file_facts`), as Ollama's do: clients
    size their prompts from ``<arch>.context_length``. The manifest fills
    in only what the files cannot give. Unknown fields are simply absent
    — ``ollama-python`` treats missing keys as ``None``.

    ``verbose=True`` is a pass-through today (we don't store tensor
    statistics). The flag is preserved for forward compatibility so
    future callers can opt in without a schema change.
    """
    facts = file_facts(manifest.local_path)
    info: dict[str, Any] = {}
    arch = facts.get("architecture") or manifest.architecture or "unknown"
    info["general.architecture"] = arch
    if manifest.parameters:
        info["general.parameter_count"] = manifest.parameters
    if manifest.quantization:
        info["general.quantization"] = manifest.quantization
    if manifest.size_bytes:
        info["general.size"] = manifest.size_bytes
    # The trained window, as the header says; the window the owner pinned
    # only when the files cannot be read (it goes in ``parameters`` too).
    context = facts.get("context_length") or max(manifest.context_length or 0, 0)
    if context:
        info[f"{arch}.context_length"] = context
    for fact, key in _LAYOUT_KEYS:
        if fact in facts:
            info[f"{arch}.{key}"] = facts[fact]
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
    family = architecture(manifest)
    return {
        "format": manifest.format or "unknown",
        "family": family or "unknown",
        "families": [family] if family else None,
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
async def show_model(req: ShowRequest, request: Request) -> dict[str, Any]:
    """Ollama-compatible ``POST /api/show``.

    Looks up the registry entry for ``req.model`` and returns the
    rendered Modelfile, chat template, parameters, details,
    architecture info and capabilities list.
    """
    manifest = get_registry().get(req.model)
    if manifest is None:
        raise ModelNotFoundError(req.model)

    from hfl.api.admin_guard import is_local_request

    body = {
        # Host paths only for the owner, like /api/ps's host figures.
        "modelfile": render_modelfile(manifest, reveal_paths=is_local_request(request)),
        "parameters": _format_parameters(manifest),
        "template": model_template(manifest),
        "details": _details(manifest),
        "model_info": _model_info(manifest, req.verbose),
        "capabilities": detect_capabilities(manifest),
        "license": manifest.license_name or manifest.license or "",
    }
    # A loaded model's engine may know them without jinja2 (llama-server).
    resident = get_state().resident(manifest.name)
    thinking = getattr(resident.engine, "thinking_controls", None) if resident else None
    if not isinstance(thinking, dict):
        thinking = thinking_controls(manifest)
    if thinking is not None:  # Ollama omits it for a model that does not reason
        body["thinking"] = thinking
    return body

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible ``GET /api/ps`` endpoint.

Lists models currently loaded in memory so clients (Open WebUI,
ollama-python, LangChain tooling) can render a "running models" view
identical to Ollama's.

Shape reference: https://docs.ollama.com/api#list-running-models

Response ::

    {"models": [
        {
          "name": "<model_name>",
          "model": "<model_name>",
          "size": <int bytes>,
          "digest": "sha256:...",
          "details": {
              "format": "gguf",
              "family": "qwen",
              "parameter_size": "7B",
              "quantization_level": "Q4_K_M"
          },
          "expires_at": "2026-04-17T15:30:00Z",
          "size_vram": <int bytes>
        },
        ...
    ]}

HFL today keeps at most ONE LLM + ONE TTS engine resident; the array
therefore has 0-2 entries. When the multi-engine ``ModelPool`` is
wired end-to-end the same endpoint will naturally list all residents
without a client-visible schema change.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter

from hfl.api.state import get_state

if TYPE_CHECKING:
    from hfl.models.manifest import ModelManifest

router = APIRouter(tags=["Ollama"])


def _manifest_digest(manifest: "ModelManifest") -> str:
    """Produce a deterministic ``sha256:...`` digest for a loaded model.

    Ollama's ``/api/ps`` always emits a digest to identify the exact
    snapshot resident in memory. We prefer the manifest's stored
    ``file_hash`` (computed at download / verify time); otherwise we
    fall back to hashing the manifest identity so the field is never
    empty — clients use it as an opaque key.
    """
    if manifest.file_hash:
        # file_hash is already the content hash; stamp with the
        # algorithm prefix that Ollama uses.
        if manifest.file_hash.startswith("sha"):
            return manifest.file_hash
        return f"{manifest.hash_algorithm}:{manifest.file_hash}"

    stamp = json.dumps(
        {"name": manifest.name, "repo_id": manifest.repo_id, "path": manifest.local_path},
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(stamp.encode()).hexdigest()


def _manifest_details(manifest: "ModelManifest") -> dict[str, Any]:
    """Compose the Ollama ``details`` sub-object from an HFL manifest."""
    return {
        "format": manifest.format or "unknown",
        "family": manifest.architecture or "unknown",
        "families": [manifest.architecture] if manifest.architecture else None,
        "parameter_size": manifest.parameters,
        "quantization_level": manifest.quantization,
    }


def _size_vram_estimate(manifest: "ModelManifest", engine: Any | None) -> int:
    """Best-effort estimate of VRAM in use for this model (bytes).

    We prefer whatever the engine reports (``engine.memory_used_bytes``
    if available); otherwise fall back to the manifest's disk size as a
    conservative upper bound (the weights themselves must be resident).
    Returns 0 for engines that explicitly run on CPU so that the
    Ollama UI distinguishes "GPU" (non-zero) from "CPU" rows.
    """
    if engine is not None:
        reporter = getattr(engine, "memory_used_bytes", None)
        if callable(reporter):
            try:
                value = reporter()
                if isinstance(value, (int, float)) and value >= 0:
                    return int(value)
            except Exception:  # pragma: no cover — defensive
                pass
        # Engines can also expose a static attribute
        attr = getattr(engine, "size_vram", None)
        if isinstance(attr, (int, float)) and attr >= 0:
            return int(attr)
    return int(manifest.size_bytes or 0)


def _expires_at_iso(manifest: "ModelManifest") -> str | None:
    """Compute the "expires at" ISO-8601 timestamp for this model.

    HFL does not yet serialise a per-model keep-alive deadline to disk;
    R15 introduces one at the ``ServerState`` layer. Until then the
    field is populated from ``ServerState.keep_alive_deadline_for()``
    if set, else ``None`` (Ollama clients interpret null as "infinite"
    / manually controlled).
    """
    state = get_state()
    getter = getattr(state, "keep_alive_deadline_for", None)
    if callable(getter):
        dt = getter(manifest.name)
        if isinstance(dt, datetime):
            # Use UTC ISO-8601 with trailing Z — Ollama's convention.
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return None


def _render_model(manifest: "ModelManifest", engine: Any | None) -> dict[str, Any]:
    """Build one Ollama-shaped model entry."""
    return {
        "name": manifest.name,
        "model": manifest.name,
        "size": int(manifest.size_bytes or 0),
        "digest": _manifest_digest(manifest),
        "details": _manifest_details(manifest),
        "expires_at": _expires_at_iso(manifest),
        "size_vram": _size_vram_estimate(manifest, engine),
    }


@router.get(
    "/api/ps",
    tags=["Ollama"],
    summary="List running models",
    responses={200: {"description": "Currently-loaded models with memory and expiry info"}},
)
async def list_running() -> dict[str, list[dict[str, Any]]]:
    """Ollama-compatible ``GET /api/ps``.

    Returns the set of models HFL currently holds in memory, shaped
    for drop-in replacement of Ollama in UIs like Open WebUI and SDKs
    like ``ollama-python``.

    The list now spans three sources, deduplicated by model name:

    1. ``state.current_model`` — the legacy single-LLM slot.
    2. ``state.current_tts_model`` — the TTS engine slot.
    3. The shared ``ModelPool`` — all multi-model entries when the
       operator opts into ``HFL_MAX_LOADED_MODELS > 1``.

    Order follows the underlying registries (state slots first, then
    pool by recency) so a single-model setup keeps emitting the same
    shape it did before V3.
    """
    state = get_state()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    if state.current_model is not None:
        entries.append(_render_model(state.current_model, state.engine))
        seen.add(state.current_model.name)
    if state.current_tts_model is not None and state.current_tts_model.name not in seen:
        entries.append(_render_model(state.current_tts_model, state.tts_engine))
        seen.add(state.current_tts_model.name)

    # Multi-model: drain the shared pool. We avoid awaiting locks here
    # because /api/ps is hit by liveness probes — read the snapshot
    # via the public attribute and accept best-effort consistency.
    try:
        from hfl.engine.model_pool import get_model_pool

        pool = get_model_pool()
        for name in pool.cached_models:
            if name in seen:
                continue
            cached = pool._models.get(name)
            if cached is None:
                continue
            entries.append(_render_model(cached.manifest, cached.engine))
            seen.add(name)
    except Exception:  # pragma: no cover — pool unavailable in some tests
        pass

    return {"models": entries}

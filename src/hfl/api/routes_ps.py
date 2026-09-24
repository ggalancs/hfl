# SPDX-License-Identifier: Apache-2.0
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

HFL keeps as many LLMs resident as the memory budget allows
(``HFL_MEMORY_BUDGET``), plus one TTS engine; every one is listed, most
recently used first. ``size`` is the estimated memory footprint
(weights + KV cache) when it can be computed, else the files on disk.

HFL extension: a top-level ``memory`` object reports the machine's
memory and the budget, so "how much room is left for the next model?"
is answerable over the API. Ollama clients ignore unknown keys.
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


def _manifest_details(manifest: "ModelManifest", engine: Any | None = None) -> dict[str, Any]:
    """Compose the Ollama ``details`` sub-object from an HFL manifest.

    ``acceleration`` / ``context_size`` are HFL extensions on top of
    Ollama's schema (extra keys, so ollama-python and Open WebUI ignore
    them). They exist so "is this model actually on the GPU, and at what
    context?" is answerable over the API instead of only by re-reading
    the server log.
    """
    details: dict[str, Any] = {
        "format": manifest.format or "unknown",
        "family": manifest.architecture or "unknown",
        "families": [manifest.architecture] if manifest.architecture else None,
        "parameter_size": manifest.parameters,
        "quantization_level": manifest.quantization,
    }
    if engine is not None:
        accel = getattr(engine, "acceleration", None)
        if accel:
            details["acceleration"] = accel
        ctx = getattr(engine, "context_size", 0)
        if isinstance(ctx, int) and ctx > 0:
            details["context_size"] = ctx
    return details


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


def _render_model(
    manifest: "ModelManifest", engine: Any | None, footprint: int = 0
) -> dict[str, Any]:
    """Build one Ollama-shaped model entry."""
    return {
        "name": manifest.name,
        "model": manifest.name,
        "size": int(footprint or manifest.size_bytes or 0),
        "digest": _manifest_digest(manifest),
        "details": _manifest_details(manifest, engine),
        "expires_at": _expires_at_iso(manifest),
        "size_vram": _size_vram_estimate(manifest, engine),
    }


@router.get(
    "/api/ps",
    tags=["Ollama"],
    summary="List running models",
    responses={200: {"description": "Currently-loaded models with memory and expiry info"}},
)
async def list_running() -> dict[str, Any]:
    """Ollama-compatible ``GET /api/ps``.

    Returns every model HFL holds in memory — each resident LLM, most
    recently used first, then the TTS engine — shaped for drop-in
    replacement of Ollama in UIs like Open WebUI and SDKs like
    ``ollama-python``, plus a ``memory`` summary (HFL extension).
    """
    state = get_state()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for resident in state.resident_models():
        entries.append(_render_model(resident.manifest, resident.engine, resident.footprint))
        seen.add(resident.name)
    # A pointer assigned without going through the resident set (older
    # callers, tests) is still a loaded model.
    if state.current_model is not None and state.current_model.name not in seen:
        entries.append(_render_model(state.current_model, state.engine))
        seen.add(state.current_model.name)
    if state.current_tts_model is not None and state.current_tts_model.name not in seen:
        entries.append(_render_model(state.current_tts_model, state.tts_engine))
        seen.add(state.current_tts_model.name)

    memory = _memory_summary(state)
    if memory is not None:
        return {"models": entries, "memory": memory}
    return {"models": entries}


def _memory_summary(state: Any) -> dict[str, Any] | None:
    """The machine's memory and the residency budget, in bytes."""
    from hfl.engine.residency import budget_fraction, current_gpu_memory, current_memory

    memory = current_memory()
    if memory is None:
        return None
    budget = budget_fraction()
    models = sum(r.footprint for r in state.resident_models())
    summary: dict[str, Any] = {
        "total_bytes": memory.total,
        "in_use_bytes": memory.in_use,
        "in_use_percent": round(100.0 * memory.in_use / memory.total, 1) if memory.total else 0.0,
        "budget_percent": round(budget * 100, 1),
        "budget_bytes": int(memory.total * budget),
        "free_within_budget_bytes": max(0, int(memory.total * budget) - memory.in_use),
        "models_bytes": models,
    }
    gpu = current_gpu_memory()
    if gpu is not None:
        summary["gpu"] = {
            "total_bytes": gpu.total,
            "in_use_bytes": gpu.in_use,
            "in_use_percent": round(100.0 * gpu.in_use / gpu.total, 1) if gpu.total else 0.0,
            "budget_bytes": int(gpu.total * budget),
            "free_within_budget_bytes": max(0, int(gpu.total * budget) - gpu.in_use),
            "hfl_bytes": gpu.hfl_rss,
        }
    return summary

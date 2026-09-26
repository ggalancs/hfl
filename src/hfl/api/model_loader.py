# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Centralized model loading logic for API routes.

Consolidates model loading from routes_openai.py and routes_native.py
to avoid code duplication and ensure consistent behavior.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hfl.api.state import get_state
from hfl.converter.formats import ModelType, detect_model_type
from hfl.engine.selector import select_engine, select_tts_engine
from hfl.exceptions import (
    ModelNotFoundError,
    ModelNotReadyError,
    ModelTypeMismatchError,
)
from hfl.exceptions import (
    ValidationError as APIValidationError,
)
from hfl.models.registry import get_registry
from hfl.validators import ValidationError, validate_model_name

if TYPE_CHECKING:
    from hfl.engine.base import AudioEngine, InferenceEngine
    from hfl.models.manifest import ModelManifest

logger = logging.getLogger(__name__)


def _manifest_ctx(manifest: "ModelManifest") -> int:
    """Context length recorded on a manifest, or 0 when unusable.

    Manifests are rehydrated from ``~/.hfl/models.json``, so the field
    can be missing or non-numeric on records written by older versions;
    anything we can't read as a positive int means "auto-detect".
    """
    try:
        value = int(manifest.context_length)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _canonical_model_name(model_name: str) -> str:
    """The local name to load for ``model_name``, validated.

    Clients that learned model names from the Hub's "Use this model" menu,
    or from Ollama, send a Hub reference: ``hf.co/org/model:Q4_K_M``. That
    is not a valid local name (``:`` is not allowed), so it is taken apart
    and each part validated on its own — the repo id with the same rules
    as any model name, the tag only if it is a quantization — and then
    mapped to the copy already on disk. The server never pulls on its own:
    a reference with no local copy is a 404 that names ``hfl pull``. A
    ``@revision`` is not accepted here; pin revisions when pulling.
    """
    from hfl.hub.resolver import parse_model_spec

    try:
        validate_model_name(model_name)
        valid = True
    except ValidationError as exc:
        valid, error = False, exc
    if valid and get_registry().get(model_name) is not None:
        return model_name

    spec = parse_model_spec(model_name)
    if spec.repo_id is not None and spec.revision is None:
        try:
            validate_model_name(spec.repo_id)
        except ValidationError as exc:
            raise APIValidationError(str(exc)) from exc
        local = get_registry().find_pulled(spec.repo_id, spec.quantization)
        if local is not None:
            return str(local.name)
        if not valid:
            raise ModelNotFoundError(model_name)
    if not valid:
        # An Ollama-style tagged short name (``qwen3:8b``) is kept locally
        # under the alias its pull gave it (``qwen3-8b``: ':' is not allowed
        # in a name). Unknown, it is a 404 like any missing model.
        from hfl.hub.shortname import alias_for, is_short_name

        if is_short_name(model_name):
            remembered = get_registry().get(alias_for(model_name))
            if remembered is not None:
                return str(remembered.name)
            raise ModelNotFoundError(model_name)
        raise APIValidationError(str(error)) from error
    return model_name


async def load_llm(
    model_name: str, num_ctx: int | None = None
) -> tuple["InferenceEngine", "ModelManifest"]:
    """Load LLM model with proper async handling.

    This is the primary entry point for model loading in API routes.
    Handles validation, registry lookup, type checking, and loading.

    Args:
        model_name: Name, alias, or repo_id of the model
        num_ctx: Per-request context size (Ollama's ``options.num_ctx``).
            When it differs from the context the resident engine was
            opened with, the model is reloaded — matching Ollama, where
            ``num_ctx`` is a load-time parameter.

    Returns:
        Tuple of (InferenceEngine, ModelManifest)

    Raises:
        APIValidationError: For malformed model names (400).
        ModelNotFoundError: If the model is not in the registry (404).
        ModelTypeMismatchError: If a non-LLM model was requested (400).
        ModelNotReadyError: If the engine slot exists but is None (503).
    """
    model_name = _canonical_model_name(model_name)

    state = get_state()

    requested_ctx = num_ctx if num_ctx and num_ctx > 0 else 0

    # Fast path - already resident with a compatible context window.
    resident = state.resident(model_name)
    candidate: "tuple[InferenceEngine, ModelManifest] | None" = None
    if resident is not None:
        candidate = (resident.engine, resident.manifest)
    elif state.current_model and state.current_model.name == model_name:
        # A preload (CLI/tray) or a test assigned the pointer directly.
        if state.engine is None:
            raise ModelNotReadyError(model_name)
        candidate = (state.engine, state.current_model)
    if candidate is not None:
        resident_engine, resident_manifest = candidate
        # ``0`` from a backend that doesn't track its context window is
        # "unknown", not "mismatched" — don't reload on a guess.
        resident_ctx = getattr(resident_engine, "context_size", 0) if requested_ctx else 0
        # A resident window that is at least as large as the request already
        # satisfies it: a model opened at 32768 serves an 8192-token request
        # perfectly. Only *growing* the window needs a reload.
        #
        # Reloading on any difference (the 0.17.0 behaviour) made a client
        # that varies num_ctx per request thrash the model: observed 16
        # reloads across 86 requests on a 44 GiB model, six of which were
        # shrinks and therefore pure waste — each one evicting the weights
        # and throwing away the KV cache, so the next request had to
        # re-prefill its whole prompt from scratch.
        if not requested_ctx or not resident_ctx or resident_ctx >= requested_ctx:
            state.bind_request(model_name)
            return resident_engine, resident_manifest
        logger.info(
            "Reloading %s: request asked for num_ctx=%d, resident engine has %d",
            model_name,
            requested_ctx,
            resident_ctx,
        )

    # Lookup in registry
    manifest = get_registry().get(model_name)
    if not manifest:
        raise ModelNotFoundError(model_name)

    # Verify model type
    model_path = Path(manifest.local_path)
    model_type = detect_model_type(model_path)
    if model_type != ModelType.LLM:
        raise ModelTypeMismatchError(model_name, expected="llm", got=model_type.value)

    # Context resolution, most specific first:
    #   1. ``options.num_ctx`` on this request (Ollama semantics).
    #   2. ``--ctx`` at server start (context_size_override > 0).
    #   3. The manifest's recorded context_length — what ``hfl run``
    #      already honours via ``load_llm_sync``. Without this the CLI
    #      and the server load the same model with different windows.
    #   4. 0 → let the engine auto-detect from GGUF metadata (clamped
    #      to the model's advertised max and to available memory).
    if requested_ctx:
        n_ctx = requested_ctx
    elif state.context_size_override > 0:
        n_ctx = state.context_size_override
    else:
        n_ctx = _manifest_ctx(manifest)

    async def _loader() -> tuple["InferenceEngine", "ModelManifest"]:
        # Other resident models stay loaded: ``ensure_llm_loaded`` has
        # already made room by memory budget (evicting idle models, least
        # recently used first) before calling this. Load off the event loop;
        # unload on failure so a half-loaded engine never leaks.
        engine = select_engine(model_path)
        started = time.monotonic()
        try:
            await asyncio.to_thread(
                engine.load, manifest.local_path, **load_kwargs_for(manifest, n_ctx)
            )
            _record_load(manifest.name, started)
        except Exception:
            if engine.is_loaded:
                try:
                    await asyncio.to_thread(engine.unload)
                except Exception as cleanup_error:
                    logger.error("Failed to cleanup engine after load error: %s", cleanup_error)
            raise
        return engine, manifest

    from hfl.config import config as _hfl_config
    from hfl.engine.footprint import estimate_footprint

    estimate = estimate_footprint(model_path, n_ctx).total_bytes

    # CON: coalesce concurrent COLD loads of the same model. The unlocked
    # fast-path above lets two simultaneous first-requests both fall through;
    # ensure_llm_loaded holds a per-model lock and re-checks residency inside
    # it, so the second request simply awaits the first's load.
    engine, loaded = await state.ensure_llm_loaded(
        model_name,
        _loader,
        timeout=_hfl_config.model_load_timeout,
        required_ctx=requested_ctx,
        estimate=estimate,
        measure=_measure_loaded,
    )
    state.bind_request(model_name)
    return engine, loaded


def _measure_loaded(engine: "InferenceEngine", manifest: "ModelManifest") -> int:
    """Footprint with the context the engine really opened, plus the MLX
    prompt cache's ceiling when that engine keeps one."""
    from hfl.engine.footprint import footprint_of_loaded

    total = footprint_of_loaded(manifest.local_path, engine).total_bytes
    if getattr(engine, "_prompt_store", None) is not None:
        from hfl.config import config as _cfg

        total += int(getattr(_cfg, "mlx_prompt_cache_bytes", 0) or 0)
    return total


async def load_tts(model_name: str) -> tuple["AudioEngine", "ModelManifest"]:
    """Load TTS model with proper async handling.

    Args:
        model_name: Name, alias, or repo_id of the TTS model

    Returns:
        Tuple of (AudioEngine, ModelManifest)

    Raises:
        APIValidationError: For malformed model names (400).
        ModelNotFoundError: If the model is not in the registry (404).
        ModelTypeMismatchError: If a non-TTS model was requested (400).
        ModelNotReadyError: If the TTS engine slot exists but is None (503).
    """
    try:
        validate_model_name(model_name)
    except ValidationError as e:
        raise APIValidationError(str(e)) from e

    state = get_state()

    # Fast path
    if state.current_tts_model and state.current_tts_model.name == model_name:
        if state.tts_engine is None:
            raise ModelNotReadyError(model_name)
        return state.tts_engine, state.current_tts_model

    manifest = get_registry().get(model_name)
    if not manifest:
        raise ModelNotFoundError(model_name)

    model_path = Path(manifest.local_path)
    model_type = detect_model_type(model_path)
    if model_type != ModelType.TTS:
        raise ModelTypeMismatchError(model_name, expected="tts", got=model_type.value)

    # Load model in thread pool
    engine = select_tts_engine(model_path)
    started = time.monotonic()
    try:
        await asyncio.to_thread(engine.load, manifest.local_path)
        _record_load(manifest.name, started)
        await state.set_tts_engine(engine, manifest)
        return engine, manifest
    except Exception:
        # Cleanup engine if loading succeeded but state update failed
        if engine.is_loaded:
            try:
                await asyncio.to_thread(engine.unload)
            except Exception as cleanup_error:
                logger.error("Failed to cleanup TTS engine after load error: %s", cleanup_error)
        raise


def load_llm_sync(model_name: str) -> tuple["InferenceEngine", "ModelManifest"]:
    """Synchronous version of load_llm for CLI usage.

    Args:
        model_name: Name, alias, or repo_id of the model

    Returns:
        Tuple of (InferenceEngine, ModelManifest)

    Raises:
        ValueError: If model not found or type mismatch
    """
    validate_model_name(model_name)

    manifest = get_registry().get(model_name)
    if not manifest:
        raise ValueError(f"Model not found: {model_name}")

    model_path = Path(manifest.local_path)
    model_type = detect_model_type(model_path)
    if model_type != ModelType.LLM:
        raise ValueError(f"Expected LLM model, got {model_type.value}")

    engine = select_engine(model_path)
    engine.load(manifest.local_path, **load_kwargs_for(manifest, manifest.context_length))

    return engine, manifest


def _record_load(name: str, started: float) -> None:
    """A model load in the metrics (``hfl_model_loads_total`` stayed at 0:
    it was fed by an event nothing emitted)."""
    try:
        from hfl.metrics import get_metrics

        get_metrics().record_model_load(name, (time.monotonic() - started) * 1000)
    except Exception:  # pragma: no cover — metrics must never break a load
        logger.debug("failed to record a model load", exc_info=True)


def load_kwargs_for(manifest: "ModelManifest", n_ctx: int | None) -> dict[str, Any]:
    """What ``engine.load`` gets for ``manifest``: the context, and the LoRA
    adapters of its Modelfile (``ADAPTER``, stored by ``POST /api/create``).

    Every LLM load goes through here — ``hfl run``, the server, ``hfl serve
    --model`` and the tray used to pass only the context, so a model created
    with ``ADAPTER`` ran without its adapter (measured).
    """
    load_kwargs: dict[str, Any] = {"n_ctx": n_ctx}
    if getattr(manifest, "adapter_paths", None):
        # ADAPTER paths come from a Modelfile via POST /api/create — untrusted.
        # Contain each to the HFL data dir so a manifest can't make the engine
        # read arbitrary files (e.g. ../../etc/passwd) as a "LoRA adapter".
        import hfl.config
        from hfl.security import PathTraversalError, sanitize_path

        base = hfl.config.config.home_dir
        safe_adapters: list[str] = []
        for adapter in manifest.adapter_paths:
            try:
                safe_adapters.append(str(sanitize_path(base, str(adapter))))
            except PathTraversalError as exc:
                # Don't echo the containment message: it prints the base
                # directory. ``from exc`` keeps the detail for the log.
                raise ValueError("adapter path rejected: outside the HFL data dir") from exc
        if safe_adapters:
            load_kwargs["lora_paths"] = safe_adapters
    draft = getattr(manifest, "draft_model_path", None)
    if isinstance(draft, str) and draft:
        load_kwargs["draft_model_path"] = _draft_for(draft)
    return load_kwargs


def _draft_for(draft: str) -> str:
    """A Modelfile DRAFT as the engine takes it: ``prompt-lookup`` as is, a
    registered model as its file, anything else a path kept inside the HFL
    data dir (the Modelfile is untrusted, as for ADAPTER)."""
    if draft == "prompt-lookup":
        return draft
    from hfl.core.container import get_registry

    registered = get_registry().get(draft)
    if registered is not None:
        return str(registered.local_path)
    import hfl.config
    from hfl.security import PathTraversalError, sanitize_path

    try:
        return str(sanitize_path(hfl.config.config.home_dir, draft))
    except PathTraversalError as exc:
        raise ValueError("draft path rejected: outside the HFL data dir") from exc

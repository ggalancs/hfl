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
    # Validate input
    try:
        validate_model_name(model_name)
    except ValidationError as e:
        raise APIValidationError(str(e)) from e

    state = get_state()

    requested_ctx = num_ctx if num_ctx and num_ctx > 0 else 0

    # Fast path - already loaded with a compatible context window.
    if state.current_model and state.current_model.name == model_name:
        if state.engine is None:
            raise ModelNotReadyError(model_name)
        if not requested_ctx:
            return state.engine, state.current_model
        # ``0`` from a backend that doesn't track its context window is
        # "unknown", not "mismatched" — don't reload on a guess.
        resident_ctx = state.engine.context_size
        if not resident_ctx or resident_ctx == requested_ctx:
            return state.engine, state.current_model
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
        # Evict the resident model BEFORE allocating the new one. HFL's
        # single-model slot loads-then-swaps, which means the outgoing
        # model's weights are still resident while the incoming ones are
        # allocated. For multi-GB models that either doubles peak memory
        # or — because the llama.cpp preflight measures free memory at
        # load time — rejects a load that would fit perfectly once the
        # old model is gone ("requires ~49.2GB but only 49.5GB are
        # available" while a 47GB model is still loaded). Ollama also
        # unloads before loading. The cost is that a failed load leaves
        # nothing resident instead of the previous model; that is the
        # right trade for a slot that can only hold one model anyway.
        #
        # Runs under ensure_llm_loaded's per-model lock, after its
        # residency re-check, so we only get here once we are committed
        # to loading. set_llm_engine(None, None) is used rather than a
        # bare unload so the dispatcher drain / pinned-engine deferral
        # in ServerState is preserved.
        if state.engine is not None:
            logger.info(
                "Evicting resident model %s before loading %s",
                state.current_model.name if state.current_model else "<unknown>",
                model_name,
            )
            await state.set_llm_engine(None, None)

        # Load off the event loop; unload on load failure so a half-loaded
        # engine never leaks. The state swap is performed by ensure_llm_loaded.
        engine = select_engine(model_path)
        try:
            await asyncio.to_thread(engine.load, manifest.local_path, n_ctx=n_ctx)
        except Exception:
            if engine.is_loaded:
                try:
                    await asyncio.to_thread(engine.unload)
                except Exception as cleanup_error:
                    logger.error("Failed to cleanup engine after load error: %s", cleanup_error)
            raise
        return engine, manifest

    # CON: coalesce concurrent COLD loads of the same model. The unlocked
    # fast-path above lets two simultaneous first-requests both fall through and
    # both run a multi-GB engine.load() (2x VRAM/OOM + A/B load thrash, the
    # second then unloading the first). ensure_llm_loaded holds a per-model lock
    # and re-checks residency inside it, so the second request simply awaits the
    # first's load instead of duplicating it.
    from hfl.config import config as _hfl_config

    return await state.ensure_llm_loaded(
        model_name,
        _loader,
        timeout=_hfl_config.model_load_timeout,
        required_ctx=requested_ctx,
    )


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
    try:
        await asyncio.to_thread(engine.load, manifest.local_path)
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
    # Phase 8 P3-2: pass LoRA adapter paths from the manifest
    # (populated by ``POST /api/create`` when a Modelfile declared
    # ``ADAPTER`` instructions). Engines that don't support LoRA
    # ignore the kwarg.
    load_kwargs: dict[str, Any] = {"n_ctx": manifest.context_length}
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
                raise ValueError(f"adapter path rejected: {exc}") from exc
        load_kwargs["lora_paths"] = safe_adapters
    engine.load(manifest.local_path, **load_kwargs)

    return engine, manifest

# SPDX-License-Identifier: HRUL-1.0
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
from typing import TYPE_CHECKING

from fastapi import HTTPException

from hfl.api.state import get_state
from hfl.converter.formats import ModelType, detect_model_type
from hfl.engine.selector import select_engine, select_tts_engine
from hfl.models.registry import get_registry
from hfl.validators import ValidationError, validate_model_name

if TYPE_CHECKING:
    from hfl.engine.base import AudioEngine, InferenceEngine
    from hfl.models.manifest import ModelManifest

logger = logging.getLogger(__name__)


async def load_llm(model_name: str) -> tuple["InferenceEngine", "ModelManifest"]:
    """Load LLM model with proper async handling.

    This is the primary entry point for model loading in API routes.
    Handles validation, registry lookup, type checking, and loading.

    Args:
        model_name: Name, alias, or repo_id of the model

    Returns:
        Tuple of (InferenceEngine, ModelManifest)

    Raises:
        HTTPException: 400 for validation errors, 404 if model not found
    """
    # Validate input
    try:
        validate_model_name(model_name)
    except ValidationError as e:
        raise HTTPException(400, detail=str(e))

    state = get_state()

    # Fast path - already loaded
    if state.current_model and state.current_model.name == model_name:
        if state.engine is None:
            raise HTTPException(503, detail="Model engine not available")
        return state.engine, state.current_model

    # Lookup in registry
    manifest = get_registry().get(model_name)
    if not manifest:
        raise HTTPException(404, detail=f"Model not found: {model_name}")

    # Verify model type
    model_path = Path(manifest.local_path)
    model_type = detect_model_type(model_path)
    if model_type != ModelType.LLM:
        raise HTTPException(
            400,
            detail={
                "error": "Model type mismatch",
                "code": "MODEL_TYPE_MISMATCH",
                "expected": "llm",
                "got": model_type.value,
            },
        )

    # Load model in thread pool to avoid blocking event loop
    engine = select_engine(model_path)
    try:
        await asyncio.to_thread(engine.load, manifest.local_path, n_ctx=manifest.context_length)
        await state.set_llm_engine(engine, manifest)
        return engine, manifest
    except Exception:
        # Cleanup engine if loading succeeded but state update failed
        if engine.is_loaded:
            try:
                await asyncio.to_thread(engine.unload)
            except Exception as cleanup_error:
                logger.error("Failed to cleanup engine after load error: %s", cleanup_error)
        raise


async def load_tts(model_name: str) -> tuple["AudioEngine", "ModelManifest"]:
    """Load TTS model with proper async handling.

    Args:
        model_name: Name, alias, or repo_id of the TTS model

    Returns:
        Tuple of (AudioEngine, ModelManifest)

    Raises:
        HTTPException: 400 for validation errors, 404 if model not found
    """
    try:
        validate_model_name(model_name)
    except ValidationError as e:
        raise HTTPException(400, detail=str(e))

    state = get_state()

    # Fast path
    if state.current_tts_model and state.current_tts_model.name == model_name:
        if state.tts_engine is None:
            raise HTTPException(503, detail="TTS engine not available")
        return state.tts_engine, state.current_tts_model

    manifest = get_registry().get(model_name)
    if not manifest:
        raise HTTPException(404, detail=f"Model not found: {model_name}")

    model_path = Path(manifest.local_path)
    model_type = detect_model_type(model_path)
    if model_type != ModelType.TTS:
        raise HTTPException(
            400,
            detail={
                "error": "Model type mismatch",
                "code": "MODEL_TYPE_MISMATCH",
                "expected": "tts",
                "got": model_type.value,
            },
        )

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
    engine.load(manifest.local_path, n_ctx=manifest.context_length)

    return engine, manifest

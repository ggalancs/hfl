# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Plugin system for HFL.

Allows dynamic discovery and loading of engine plugins via entry points.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Type

from hfl.logging_config import get_logger

if TYPE_CHECKING:
    from hfl.engine.base import AudioEngine, InferenceEngine

logger = get_logger()


def _lazy_import(module: str, name: str) -> type:
    """Lazy import helper that returns a callable.

    Returns a function that imports and returns the class when called.
    """

    def loader():
        mod = importlib.import_module(module)
        return getattr(mod, name)

    return loader


def discover_engines() -> dict[str, Type["InferenceEngine"] | callable]:
    """Discover available inference engine plugins.

    Includes built-in engines and any plugins registered via entry points.
    Built-in engines are lazily imported to avoid loading unused dependencies.

    Returns:
        Dictionary mapping engine name to class or lazy loader
    """
    engines: dict[str, Type["InferenceEngine"] | callable] = {}

    # Built-in engines (lazy imports)
    engines["llama-cpp"] = _lazy_import("hfl.engine.llama_cpp", "LlamaCppEngine")
    engines["transformers"] = _lazy_import(
        "hfl.engine.transformers_engine", "TransformersEngine"
    )
    engines["vllm"] = _lazy_import("hfl.engine.vllm_engine", "VLLMEngine")

    # Plugin engines via entry points
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="hfl.engines")
        for ep in eps:
            try:
                engine_class = ep.load()
                engines[ep.name] = engine_class
                logger.info(f"Loaded engine plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load engine plugin {ep.name}: {e}")
    except Exception:
        pass  # entry_points not available or error

    return engines


def discover_tts_engines() -> dict[str, Type["AudioEngine"] | callable]:
    """Discover available TTS engine plugins.

    Returns:
        Dictionary mapping engine name to class or lazy loader
    """
    engines: dict[str, Type["AudioEngine"] | callable] = {}

    # Built-in TTS engines
    engines["bark"] = _lazy_import("hfl.engine.bark_engine", "BarkEngine")
    engines["coqui"] = _lazy_import("hfl.engine.coqui_engine", "CoquiEngine")

    # Plugin TTS engines
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="hfl.tts_engines")
        for ep in eps:
            try:
                engines[ep.name] = ep.load()
                logger.info(f"Loaded TTS plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load TTS plugin {ep.name}: {e}")
    except Exception:
        pass

    return engines


def get_engine_class(engine_name: str) -> Type["InferenceEngine"]:
    """Get engine class by name, loading if necessary.

    Args:
        engine_name: Name of the engine

    Returns:
        Engine class

    Raises:
        KeyError: If engine not found
        ImportError: If engine dependencies not available
    """
    engines = discover_engines()

    if engine_name not in engines:
        raise KeyError(f"Unknown engine: {engine_name}. Available: {list(engines.keys())}")

    engine = engines[engine_name]

    # If it's a lazy loader, call it to get the actual class
    if callable(engine) and not isinstance(engine, type):
        engine = engine()

    return engine


def get_tts_engine_class(engine_name: str) -> Type["AudioEngine"]:
    """Get TTS engine class by name, loading if necessary.

    Args:
        engine_name: Name of the TTS engine

    Returns:
        TTS engine class

    Raises:
        KeyError: If engine not found
        ImportError: If engine dependencies not available
    """
    engines = discover_tts_engines()

    if engine_name not in engines:
        raise KeyError(
            f"Unknown TTS engine: {engine_name}. Available: {list(engines.keys())}"
        )

    engine = engines[engine_name]

    if callable(engine) and not isinstance(engine, type):
        engine = engine()

    return engine


def list_available_engines() -> list[str]:
    """List names of available inference engines.

    Returns:
        List of engine names
    """
    return list(discover_engines().keys())


def list_available_tts_engines() -> list[str]:
    """List names of available TTS engines.

    Returns:
        List of TTS engine names
    """
    return list(discover_tts_engines().keys())

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Thread-safe server state management.

Provides atomic access to shared server state using asyncio.Lock
for safe concurrent access in async context.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfl.engine.base import AudioEngine, InferenceEngine
    from hfl.models.registry import ModelManifest


@dataclass
class ServerState:
    """Thread-safe server state container.

    Uses asyncio.Lock for safe concurrent access to mutable state.
    All state modifications should be done through the provided methods.
    """

    # LLM state
    _engine: InferenceEngine | None = None
    _current_model: ModelManifest | None = None

    # TTS state
    _tts_engine: AudioEngine | None = None
    _current_tts_model: ModelManifest | None = None

    # Security
    _api_key: str | None = None

    # Locks for thread-safe access
    _llm_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _tts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Properties with setters for testing compatibility
    @property
    def engine(self) -> InferenceEngine | None:
        """Get current LLM engine."""
        return self._engine

    @engine.setter
    def engine(self, value: InferenceEngine | None) -> None:
        """Set LLM engine (for testing, use set_llm_engine in async context)."""
        self._engine = value

    @property
    def current_model(self) -> ModelManifest | None:
        """Get current LLM model manifest."""
        return self._current_model

    @current_model.setter
    def current_model(self, value: ModelManifest | None) -> None:
        """Set current model (for testing)."""
        self._current_model = value

    @property
    def tts_engine(self) -> AudioEngine | None:
        """Get current TTS engine."""
        return self._tts_engine

    @tts_engine.setter
    def tts_engine(self, value: AudioEngine | None) -> None:
        """Set TTS engine (for testing)."""
        self._tts_engine = value

    @property
    def current_tts_model(self) -> ModelManifest | None:
        """Get current TTS model manifest."""
        return self._current_tts_model

    @current_tts_model.setter
    def current_tts_model(self, value: ModelManifest | None) -> None:
        """Set current TTS model (for testing)."""
        self._current_tts_model = value

    @property
    def api_key(self) -> str | None:
        """Get API key."""
        return self._api_key

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        """Set API key (thread-safe for simple assignment)."""
        self._api_key = value

    # Thread-safe LLM operations
    async def set_llm_engine(
        self,
        engine: InferenceEngine | None,
        model: ModelManifest | None,
    ) -> None:
        """Set LLM engine and model atomically.

        Args:
            engine: New inference engine (or None to unload)
            model: Model manifest for the loaded model
        """
        async with self._llm_lock:
            # Unload previous engine if exists
            if self._engine is not None and self._engine.is_loaded:
                self._engine.unload()
            self._engine = engine
            self._current_model = model

    async def with_llm_engine(self) -> asyncio.Lock:
        """Context manager for exclusive LLM engine access.

        Usage:
            async with state.with_llm_engine():
                result = state.engine.generate(...)
        """
        return self._llm_lock

    def is_llm_loaded(self) -> bool:
        """Check if LLM engine is loaded."""
        return self._engine is not None and self._engine.is_loaded

    # Thread-safe TTS operations
    async def set_tts_engine(
        self,
        engine: AudioEngine | None,
        model: ModelManifest | None,
    ) -> None:
        """Set TTS engine and model atomically.

        Args:
            engine: New audio engine (or None to unload)
            model: Model manifest for the loaded model
        """
        async with self._tts_lock:
            # Unload previous engine if exists
            if self._tts_engine is not None and self._tts_engine.is_loaded:
                self._tts_engine.unload()
            self._tts_engine = engine
            self._current_tts_model = model

    async def with_tts_engine(self) -> asyncio.Lock:
        """Context manager for exclusive TTS engine access."""
        return self._tts_lock

    def is_tts_loaded(self) -> bool:
        """Check if TTS engine is loaded."""
        return self._tts_engine is not None and self._tts_engine.is_loaded

    # Cleanup
    async def cleanup(self) -> None:
        """Cleanup all engines on shutdown."""
        async with self._llm_lock:
            if self._engine is not None and self._engine.is_loaded:
                self._engine.unload()
            self._engine = None
            self._current_model = None

        async with self._tts_lock:
            if self._tts_engine is not None and self._tts_engine.is_loaded:
                self._tts_engine.unload()
            self._tts_engine = None
            self._current_tts_model = None


# Singleton instance
_state: ServerState | None = None


def get_state() -> ServerState:
    """Get the singleton server state instance.

    Creates the instance on first call (lazy initialization).
    """
    global _state
    if _state is None:
        _state = ServerState()
    return _state


def reset_state() -> None:
    """Reset state (for testing purposes)."""
    global _state
    _state = None

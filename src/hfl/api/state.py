# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Thread-safe server state management.

Provides atomic access to shared server state using asyncio.Lock
for safe concurrent access in async context.

Features:
- Model loading serialization (prevents concurrent loads of same model)
- Per-model locks for fine-grained concurrency control
- Timeout support for long-running operations
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable

if TYPE_CHECKING:
    from datetime import datetime  # noqa: F401 — used in type comments / methods

    from hfl.engine.base import AudioEngine, InferenceEngine
    from hfl.engine.dispatcher import InferenceDispatcher
    from hfl.models.manifest import ModelManifest


@dataclass
class ServerState:
    """Thread-safe server state container.

    Uses asyncio.Lock for safe concurrent access to mutable state.
    All state modifications should be done through the provided methods.

    Features:
    - Per-model locks prevent concurrent loads of the same model
    - Tracks loading state for API health checks
    """

    # LLM state
    _engine: InferenceEngine | None = None
    _current_model: ModelManifest | None = None

    # TTS state
    _tts_engine: AudioEngine | None = None
    _current_tts_model: ModelManifest | None = None

    # Security
    _api_key: str | None = None

    # Context size override (0 = use model default)
    context_size_override: int = 0

    # Locks for thread-safe access
    _llm_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _tts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Per-model locks to serialize loading of the same model
    _model_locks: dict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))

    # Track which models are currently loading
    _loading_models: set[str] = field(default_factory=set)

    # Engine hot-swap safety (CON). Inference paths that do NOT hold a
    # dispatcher slot (the WebSocket chat turn) ``pin`` the engine they read
    # for the duration of their use. ``set_llm_engine`` drains the dispatcher
    # (so no slot-holding HTTP request is mid-flight) and then, if the displaced
    # engine is still pinned by such a path, defers its ``unload`` until the
    # last reader ``unpin``s it — so a swap can never free the model out from
    # under an in-flight request (use-after-free of the non-reentrant model).
    _engine_inuse: dict[int, int] = field(default_factory=dict)
    _engine_retired: dict[int, "InferenceEngine"] = field(default_factory=dict)
    _engine_ref_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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

        Unloading the previous engine is delegated to a worker thread
        so that GPU / Metal teardown — which can take several seconds
        for a large model — does not block the asyncio event loop and
        starve other endpoints (``/healthz``, ``/metrics``, etc.).

        Args:
            engine: New inference engine (or None to unload)
            model: Model manifest for the loaded model
        """
        async with self._llm_lock:
            prev = self._engine

            # Fast path: nothing loaded to retire (first load / no-op re-set).
            if prev is None or prev is engine or not prev.is_loaded:
                self._engine = engine
                self._current_model = model
                return

            # There is a loaded previous engine to retire. Retire it BEFORE
            # assigning the new one so an unload *failure* leaves the old engine
            # in place (atomic swap — callers observe "replacement failed", not
            # a half-updated state). The teardown runs off-loop so a slow unload
            # (GPU/Metal context cleanup) doesn't freeze unrelated endpoints.
            async def _retire_and_assign() -> None:
                # A non-dispatcher path (the WS chat turn) may still be reading
                # ``prev``; if so, defer its unload until the last reader unpins.
                async with self._engine_ref_lock:
                    pinned = self._engine_inuse.get(id(prev), 0) > 0
                    if pinned:
                        self._engine_retired[id(prev)] = prev
                if not pinned and prev.is_loaded:
                    # May raise (e.g. GPU fault) — propagate without assigning,
                    # so ``self._engine`` stays the old engine.
                    await asyncio.to_thread(prev.unload)
                self._engine = engine
                self._current_model = model

            # Drain in-flight dispatcher inference first, so no slot-holding
            # HTTP request (stream or non-stream) is mid-read of ``prev`` when
            # it is freed. ``exclusive()`` waits for every inference slot to
            # clear — guarding against a use-after-free of the shared model.
            dispatcher = self._try_get_dispatcher()
            if dispatcher is not None:
                async with dispatcher.exclusive():
                    await _retire_and_assign()
            else:  # pragma: no cover — dispatcher always present in running app
                await _retire_and_assign()

    @staticmethod
    def _try_get_dispatcher() -> "InferenceDispatcher | None":
        """Best-effort handle to the inference dispatcher (None if unavailable,
        e.g. in a unit test that never built the container)."""
        try:
            from hfl.core import get_dispatcher

            return get_dispatcher()
        except Exception:  # pragma: no cover — defensive
            return None

    async def pin_engine(self, engine: "InferenceEngine | None") -> None:
        """Mark ``engine`` as in-use by an in-flight request that does not hold
        a dispatcher slot (the WebSocket chat turn). Paired with
        :meth:`unpin_engine`; while pinned, a hot-swap defers its unload."""
        if engine is None:
            return
        async with self._engine_ref_lock:
            self._engine_inuse[id(engine)] = self._engine_inuse.get(id(engine), 0) + 1

    async def unpin_engine(self, engine: "InferenceEngine | None") -> None:
        """Release a pinned engine. If it was displaced by a hot-swap and this
        was its last in-flight reader, unload it now (off-loop)."""
        if engine is None:
            return
        to_unload: "InferenceEngine | None" = None
        async with self._engine_ref_lock:
            key = id(engine)
            remaining = self._engine_inuse.get(key, 0) - 1
            if remaining > 0:
                self._engine_inuse[key] = remaining
            else:
                self._engine_inuse.pop(key, None)
                to_unload = self._engine_retired.pop(key, None)
        if to_unload is not None and to_unload.is_loaded:
            await asyncio.to_thread(to_unload.unload)

    @asynccontextmanager
    async def with_llm_engine(self) -> AsyncIterator["InferenceEngine"]:
        """Context manager for safe LLM engine access with lock protection.

        Acquires the LLM lock and yields the engine. Ensures the engine
        cannot be unloaded while in use.

        Usage:
            async with state.with_llm_engine() as engine:
                result = engine.generate(...)

        Raises:
            ModelNotLoadedError: If no LLM model is loaded
        """
        from hfl.exceptions import ModelNotLoadedError

        async with self._llm_lock:
            if self._engine is None:
                raise ModelNotLoadedError()
            yield self._engine

    def is_llm_loaded(self) -> bool:
        """Check if LLM engine is loaded."""
        return self._engine is not None and self._engine.is_loaded

    @property
    def is_loading(self) -> bool:
        """Check if any model is currently loading."""
        return len(self._loading_models) > 0

    @property
    def loading_models(self) -> set[str]:
        """Get set of currently loading model names."""
        return self._loading_models.copy()

    async def ensure_llm_loaded(
        self,
        model_name: str,
        loader: Callable[[], Awaitable[tuple["InferenceEngine", "ModelManifest"]]],
        timeout: float = 300.0,
    ) -> tuple["InferenceEngine", "ModelManifest"]:
        """Ensure LLM model is loaded, with serialization per model.

        If the requested model is already loaded, returns immediately.
        If another request is loading the same model, waits for it.
        Uses per-model locks to allow loading different models concurrently.

        Args:
            model_name: Name of the model to load
            loader: Async function that loads the model and returns (engine, manifest)
            timeout: Maximum time to wait for model loading (seconds)

        Returns:
            Tuple of (engine, manifest)

        Raises:
            asyncio.TimeoutError: If loading takes longer than timeout
        """

        async def _load_with_lock() -> tuple["InferenceEngine", "ModelManifest"]:
            async with self._model_locks[model_name]:
                # Check state inside lock to prevent race condition
                # Another thread could clear the engine between check and use
                if (
                    self._current_model
                    and self._current_model.name == model_name
                    and self._engine is not None
                ):
                    return self._engine, self._current_model

                self._loading_models.add(model_name)
                engine = None
                try:
                    engine, manifest = await loader()
                    await self.set_llm_engine(engine, manifest)
                    return engine, manifest
                except Exception:
                    # If the engine loaded but the swap failed (e.g. the previous
                    # engine's unload raised), unload the orphan so a failed swap
                    # doesn't leak its weights/VRAM.
                    if engine is not None and getattr(engine, "is_loaded", False):
                        try:
                            await asyncio.to_thread(engine.unload)
                        except Exception:  # pragma: no cover - best-effort cleanup
                            pass
                    raise
                finally:
                    self._loading_models.discard(model_name)

        # Use asyncio.wait_for for cross-platform timeout (works on Python 3.10+)
        return await asyncio.wait_for(_load_with_lock(), timeout=timeout)

    async def ensure_tts_loaded(
        self,
        model_name: str,
        loader: Callable[[], Awaitable[tuple["AudioEngine", "ModelManifest"]]],
        timeout: float = 300.0,
    ) -> tuple["AudioEngine", "ModelManifest"]:
        """Ensure TTS model is loaded, with serialization per model.

        Similar to ensure_llm_loaded but for TTS models.
        """

        async def _load_with_lock() -> tuple["AudioEngine", "ModelManifest"]:
            async with self._model_locks[f"tts:{model_name}"]:
                # Check state inside lock to prevent race condition
                if (
                    self._current_tts_model
                    and self._current_tts_model.name == model_name
                    and self._tts_engine is not None
                ):
                    return self._tts_engine, self._current_tts_model

                self._loading_models.add(f"tts:{model_name}")
                try:
                    engine, manifest = await loader()
                    await self.set_tts_engine(engine, manifest)
                    return engine, manifest
                finally:
                    self._loading_models.discard(f"tts:{model_name}")

        # Use asyncio.wait_for for cross-platform timeout (works on Python 3.10+)
        return await asyncio.wait_for(_load_with_lock(), timeout=timeout)

    # Thread-safe TTS operations
    async def set_tts_engine(
        self,
        engine: AudioEngine | None,
        model: ModelManifest | None,
    ) -> None:
        """Set TTS engine and model atomically.

        See ``set_llm_engine`` for the rationale behind the off-loop
        unload.

        Args:
            engine: New audio engine (or None to unload)
            model: Model manifest for the loaded model
        """
        async with self._tts_lock:
            # Unload previous engine off-loop; see set_llm_engine.
            if self._tts_engine is not None and self._tts_engine.is_loaded:
                await asyncio.to_thread(self._tts_engine.unload)
            self._tts_engine = engine
            self._current_tts_model = model

    @asynccontextmanager
    async def with_tts_engine(self) -> AsyncIterator["AudioEngine"]:
        """Context manager for safe TTS engine access with lock protection.

        Acquires the TTS lock and yields the engine. Ensures the engine
        cannot be unloaded while in use.

        Usage:
            async with state.with_tts_engine() as engine:
                result = engine.synthesize(...)

        Raises:
            ModelNotLoadedError: If no TTS model is loaded
        """
        from hfl.exceptions import ModelNotLoadedError

        async with self._tts_lock:
            if self._tts_engine is None:
                raise ModelNotLoadedError()
            yield self._tts_engine

    def is_tts_loaded(self) -> bool:
        """Check if TTS engine is loaded."""
        return self._tts_engine is not None and self._tts_engine.is_loaded

    # -- keep_alive tracking -------------------------------------------
    # Per-model keep-alive deadline, populated by request handlers
    # (R15). ``None`` means "managed by the default idle timeout /
    # never auto-expires". The storage is per-name rather than per
    # engine so it survives model hot-swaps.
    def keep_alive_deadline_for(self, model_name: str) -> "datetime | None":
        """Return the keep-alive deadline for ``model_name`` (or None).

        Consulted by ``/api/ps`` to emit the ``expires_at`` field.
        """
        from datetime import datetime as _dt

        deadlines: dict[str, _dt] = getattr(self, "_keep_alive_deadlines", {})
        return deadlines.get(model_name)

    def set_keep_alive_deadline(self, model_name: str, deadline: "datetime | None") -> None:
        """Set / clear the keep-alive deadline for ``model_name``.

        Called from request handlers when they receive a ``keep_alive``
        value (R15). Passing ``None`` clears the deadline.
        """
        from datetime import datetime as _dt

        deadlines: dict[str, _dt] = self.__dict__.setdefault("_keep_alive_deadlines", {})
        if deadline is None:
            deadlines.pop(model_name, None)
        else:
            deadlines[model_name] = deadline

    # Cleanup
    async def cleanup(self) -> None:
        """Cleanup all engines on shutdown.

        Routes through ``set_llm_engine``/``set_tts_engine`` so the LLM unload
        DRAINS in-flight inference via ``dispatcher.exclusive()`` (and defers a
        still-pinned engine) rather than freeing the shared non-reentrant model
        out from under a request still running during uvicorn's graceful-
        shutdown window — the same use-after-free the hot-swap path prevents,
        at the moment (SIGTERM under load) it matters most. ``unload()`` still
        runs off-loop so the event loop stays alive to finish in-flight
        responses. (CON)
        """
        await self.set_llm_engine(None, None)
        await self.set_tts_engine(None, None)


# Singleton access delegated to container for unified management


def get_state() -> ServerState:
    """Get the singleton server state instance.

    Creates the instance on first call (lazy initialization).
    """
    from hfl.core.container import get_state as _get_state

    return _get_state()


def reset_state() -> None:
    """Reset state (for testing purposes)."""
    from hfl.core.container import get_container

    get_container().state.reset()

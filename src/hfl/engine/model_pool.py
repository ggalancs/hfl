# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Model caching and pooling for efficient memory management.

Provides LRU cache for loaded models to avoid repeated loading
and smart eviction based on memory and idle time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from hfl.engine.base import InferenceEngine
    from hfl.models.manifest import ModelManifest

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """Cached model entry with metadata."""

    engine: "InferenceEngine"
    manifest: "ModelManifest"
    last_used: float
    load_time_ms: float
    memory_estimate_mb: float = 0.0

    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used = time.time()


class ModelPool:
    """LRU cache for loaded inference models.

    Manages multiple loaded models in memory, evicting oldest
    when limits are reached.

    Features:
    - LRU eviction based on usage
    - Idle timeout eviction
    - Memory-aware eviction (estimates)
    - Async-safe operations
    """

    def __init__(
        self,
        max_models: int = 3,
        idle_timeout_seconds: float = 3600.0,  # 1 hour
        max_memory_mb: float | None = None,
        background_eviction_interval: float = 60.0,  # Check every minute
    ):
        """Initialize model pool.

        Args:
            max_models: Maximum number of models to keep loaded
            idle_timeout_seconds: Evict models idle longer than this
            max_memory_mb: Maximum total memory for cached models (estimate)
            background_eviction_interval: Seconds between background eviction checks
        """
        self._models: OrderedDict[str, CachedModel] = OrderedDict()
        self._max_models = max_models
        self._idle_timeout = idle_timeout_seconds
        self._max_memory_mb = max_memory_mb
        self._lock = asyncio.Lock()
        self._total_memory_mb = 0.0
        self._background_eviction_interval = background_eviction_interval
        self._shutdown = False
        self._background_task: asyncio.Task | None = None
        # Set of model names currently being loaded (to prevent duplicate loads)
        self._loading: set[str] = set()

    async def get(self, model_name: str) -> CachedModel | None:
        """Get a cached model by name.

        Args:
            model_name: Model name

        Returns:
            CachedModel if found, None otherwise
        """
        async with self._lock:
            if model_name in self._models:
                cached = self._models[model_name]
                cached.touch()
                # Move to end (most recently used)
                self._models.move_to_end(model_name)
                return cached
            return None

    async def get_or_load(
        self,
        model_name: str,
        loader: Callable[[], Awaitable[tuple["InferenceEngine", "ModelManifest", float]]],
    ) -> CachedModel:
        """Get cached model or load it.

        This method is designed to avoid deadlocks by not holding the lock
        during the potentially long-running loader operation.

        Args:
            model_name: Model name
            loader: Async function returning (engine, manifest, memory_mb)

        Returns:
            CachedModel (cached or newly loaded)
        """
        # Check cache first (fast path without lock contention)
        cached = await self.get(model_name)
        if cached is not None:
            return cached

        # Check if we should load (with lock) but don't load yet
        should_load = False
        async with self._lock:
            # Double-check after acquiring lock
            if model_name in self._models:
                cached = self._models[model_name]
                cached.touch()
                self._models.move_to_end(model_name)
                return cached

            # Check if another coroutine is already loading this model
            if model_name in self._loading:
                # Wait for the other coroutine to finish loading
                pass
            else:
                # Mark that we're loading this model
                self._loading.add(model_name)
                should_load = True

                # Evict if necessary before loading
                await self._evict_if_needed_locked()

        # If another coroutine is loading, poll until ready (non-recursive)
        if not should_load:
            for _ in range(3000):  # Max ~5 minutes (3000 * 0.1s)
                await asyncio.sleep(0.1)
                cached = await self.get(model_name)
                if cached is not None:
                    return cached
                # Check if still loading
                async with self._lock:
                    if model_name not in self._loading:
                        break
            # If we get here, either it finished loading or timed out
            cached = await self.get(model_name)
            if cached is not None:
                return cached
            # Retry loading ourselves
            return await self.get_or_load(model_name, loader)

        # Load model OUTSIDE the lock to prevent deadlock
        engine = None
        try:
            start_time = time.time()
            engine, manifest, memory_mb = await loader()
            load_time_ms = (time.time() - start_time) * 1000

            cached = CachedModel(
                engine=engine,
                manifest=manifest,
                last_used=time.time(),
                load_time_ms=load_time_ms,
                memory_estimate_mb=memory_mb,
            )

            # Update cache with lock
            async with self._lock:
                # Check again in case of race (another coroutine loaded it)
                if model_name in self._models:
                    # Another coroutine loaded it while we were loading
                    # Unload our engine and return the cached one
                    if engine.is_loaded:
                        engine.unload()
                    existing = self._models[model_name]
                    existing.touch()
                    self._models.move_to_end(model_name)
                    return existing

                self._models[model_name] = cached
                self._total_memory_mb += memory_mb

            return cached

        except Exception:
            # Cleanup engine on failure
            if engine is not None and engine.is_loaded:
                try:
                    engine.unload()
                except Exception as e:
                    logger.error("Failed to unload engine after load error: %s", e)
            raise

        finally:
            # Always remove from loading set
            async with self._lock:
                self._loading.discard(model_name)

    async def evict(self, model_name: str) -> bool:
        """Evict a specific model from the cache.

        Args:
            model_name: Model to evict

        Returns:
            True if model was evicted, False if not found
        """
        async with self._lock:
            if model_name in self._models:
                cached = self._models.pop(model_name)
                self._total_memory_mb -= cached.memory_estimate_mb
                if cached.engine.is_loaded:
                    cached.engine.unload()
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached models."""
        async with self._lock:
            for cached in self._models.values():
                if cached.engine.is_loaded:
                    cached.engine.unload()
            self._models.clear()
            self._total_memory_mb = 0.0

    async def _evict_if_needed_locked(self) -> None:
        """Evict models if over limits (must hold lock)."""
        now = time.time()

        # Evict idle models first
        to_evict = []
        for name, cached in self._models.items():
            if now - cached.last_used > self._idle_timeout:
                to_evict.append(name)

        for name in to_evict:
            self._evict_locked(name)

        # Evict by count
        while len(self._models) >= self._max_models:
            oldest_name = next(iter(self._models))
            self._evict_locked(oldest_name)

        # Evict by memory if configured
        if self._max_memory_mb is not None:
            while len(self._models) > 0 and self._total_memory_mb > self._max_memory_mb:
                oldest_name = next(iter(self._models))
                self._evict_locked(oldest_name)

    def _evict_locked(self, model_name: str) -> None:
        """Evict a model (must hold lock)."""
        if model_name in self._models:
            cached = self._models.pop(model_name)
            self._total_memory_mb -= cached.memory_estimate_mb
            if cached.engine.is_loaded:
                cached.engine.unload()

    @property
    def cached_models(self) -> list[str]:
        """List of cached model names."""
        return list(self._models.keys())

    @property
    def size(self) -> int:
        """Number of cached models."""
        return len(self._models)

    @property
    def total_memory_mb(self) -> float:
        """Estimated total memory usage in MB."""
        return self._total_memory_mb

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cached_models": len(self._models),
            "max_models": self._max_models,
            "total_memory_mb": self._total_memory_mb,
            "max_memory_mb": self._max_memory_mb,
            "background_eviction_active": self._background_task is not None
            and not self._background_task.done(),
            "models": [
                {
                    "name": name,
                    "last_used_seconds_ago": time.time() - cached.last_used,
                    "memory_mb": cached.memory_estimate_mb,
                    "load_time_ms": cached.load_time_ms,
                }
                for name, cached in self._models.items()
            ],
        }

    def start_background_eviction(self) -> None:
        """Start the background eviction loop.

        Should be called once when the event loop is running.
        The loop checks for idle models periodically and evicts them.
        """
        if self._background_task is not None and not self._background_task.done():
            return  # Already running

        self._shutdown = False
        self._background_task = asyncio.create_task(self._background_eviction_loop())
        logger.info(
            f"Background eviction started (interval: {self._background_eviction_interval}s)"
        )

    async def stop_background_eviction(self) -> None:
        """Stop the background eviction loop."""
        self._shutdown = True
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            logger.info("Background eviction stopped")

    async def _background_eviction_loop(self) -> None:
        """Background task for periodic idle model eviction.

        Features:
        - Exponential backoff on errors
        - Maximum consecutive error limit
        - Graceful degradation instead of infinite retry
        """
        consecutive_errors = 0
        max_consecutive_errors = 5
        backoff_seconds = 1.0
        max_backoff = 60.0

        while not self._shutdown:
            try:
                await asyncio.sleep(self._background_eviction_interval)

                # Find and evict idle models
                async with self._lock:
                    now = time.time()
                    to_evict = [
                        name
                        for name, cached in self._models.items()
                        if now - cached.last_used > self._idle_timeout
                    ]

                    for name in to_evict:
                        self._evict_locked(name)
                        logger.info("Background eviction: evicted idle model '%s'", name)

                # Reset error tracking on success
                consecutive_errors = 0
                backoff_seconds = 1.0

            except asyncio.CancelledError:
                logger.info("Background eviction loop cancelled")
                break

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    "Error in background eviction loop (%s/%s): %s",
                    consecutive_errors, max_consecutive_errors, e
                )

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        "Background eviction loop stopping after %s "
                        "consecutive errors. Manual restart required.",
                        max_consecutive_errors
                    )
                    # Emit event for monitoring if available
                    try:
                        from hfl.events import EventType, emit
                        emit(
                            EventType.ERROR,
                            source="model_pool",
                            error="eviction_loop_failed",
                            consecutive_errors=consecutive_errors,
                        )
                    except ImportError:
                        pass
                    break

                # Exponential backoff
                logger.warning("Retrying eviction loop in %.1fs", backoff_seconds)
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, max_backoff)

    async def shutdown(self) -> None:
        """Shutdown the pool, stopping background tasks and clearing models."""
        await self.stop_background_eviction()
        await self.clear()


# Singleton instance with thread-safe initialization
import threading

_pool: ModelPool | None = None
_pool_lock = threading.Lock()


def get_model_pool(
    max_models: int = 3,
    idle_timeout_seconds: float = 3600.0,
) -> ModelPool:
    """Get or create the singleton model pool.

    Thread-safe singleton pattern using double-checked locking.

    Args:
        max_models: Maximum cached models (only used on creation)
        idle_timeout_seconds: Idle timeout (only used on creation)

    Returns:
        ModelPool instance
    """
    global _pool
    # Fast path: return existing instance without locking
    if _pool is not None:
        return _pool

    # Slow path: acquire lock and create if needed
    with _pool_lock:
        # Double-check after acquiring lock
        if _pool is None:
            _pool = ModelPool(
                max_models=max_models,
                idle_timeout_seconds=idle_timeout_seconds,
            )
        return _pool


def reset_model_pool() -> None:
    """Reset the model pool (for testing)."""
    global _pool
    with _pool_lock:
        _pool = None

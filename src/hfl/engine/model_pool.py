# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Model caching and pooling for efficient memory management.

Provides LRU cache for loaded models to avoid repeated loading
and smart eviction based on memory and idle time.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from hfl.engine.base import InferenceEngine
    from hfl.models.manifest import ModelManifest


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
    ):
        """Initialize model pool.

        Args:
            max_models: Maximum number of models to keep loaded
            idle_timeout_seconds: Evict models idle longer than this
            max_memory_mb: Maximum total memory for cached models (estimate)
        """
        self._models: OrderedDict[str, CachedModel] = OrderedDict()
        self._max_models = max_models
        self._idle_timeout = idle_timeout_seconds
        self._max_memory_mb = max_memory_mb
        self._lock = asyncio.Lock()
        self._total_memory_mb = 0.0

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

        Args:
            model_name: Model name
            loader: Async function returning (engine, manifest, memory_mb)

        Returns:
            CachedModel (cached or newly loaded)
        """
        # Check cache first
        cached = await self.get(model_name)
        if cached is not None:
            return cached

        async with self._lock:
            # Double-check after acquiring lock
            if model_name in self._models:
                cached = self._models[model_name]
                cached.touch()
                self._models.move_to_end(model_name)
                return cached

            # Evict if necessary before loading
            await self._evict_if_needed_locked()

            # Load new model
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

            self._models[model_name] = cached
            self._total_memory_mb += memory_mb

            return cached

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


# Singleton instance
_pool: ModelPool | None = None


def get_model_pool(
    max_models: int = 3,
    idle_timeout_seconds: float = 3600.0,
) -> ModelPool:
    """Get or create the singleton model pool.

    Args:
        max_models: Maximum cached models (only used on creation)
        idle_timeout_seconds: Idle timeout (only used on creation)

    Returns:
        ModelPool instance
    """
    global _pool
    if _pool is None:
        _pool = ModelPool(
            max_models=max_models,
            idle_timeout_seconds=idle_timeout_seconds,
        )
    return _pool


def reset_model_pool() -> None:
    """Reset the model pool (for testing)."""
    global _pool
    _pool = None

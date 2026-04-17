# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Shared-prefix prompt cache (Phase 11 P1 — V2 row 10).

A very small LRU that records the longest common prefix seen for a
given token sequence. Engines that opt in (currently only the
llama-cpp backend — Transformers and vLLM manage their own caches)
query this before running a generation and surface which prefix
length is guaranteed to be prefix-compatible.

The cache does **not** store KV states itself — that is engine
internal state. It only remembers which prefix was last in the
engine's KV cache so the engine can decide whether to reuse it.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Sequence

logger = logging.getLogger(__name__)

__all__ = ["PromptPrefixCache"]


class PromptPrefixCache:
    """Tiny LRU of ``model_key -> tuple(tokens)``.

    Thread-safe (the dispatcher serialises engine calls today but the
    cache may be touched from multiple async tasks for read-only
    queries). Capped by ``max_entries``; the oldest entry is dropped
    on eviction.
    """

    def __init__(self, max_entries: int = 32) -> None:
        self._cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def longest_prefix(self, model_key: str, tokens: Sequence[int]) -> int:
        """Return the length of the longest previously-seen prefix.

        ``0`` means no reusable prefix is known. Callers pass the
        result through to the engine as a hint (e.g. llama-cpp's
        cache-shift mechanism).
        """
        if not tokens:
            return 0
        with self._lock:
            cached = self._cache.get(model_key)
            if cached is None:
                return 0
            # Move to the end so it's marked most-recently-used.
            self._cache.move_to_end(model_key)
        longest = 0
        tokens_tuple = tuple(tokens)
        limit = min(len(cached), len(tokens_tuple))
        for i in range(limit):
            if cached[i] != tokens_tuple[i]:
                break
            longest = i + 1
        return longest

    def record(self, model_key: str, tokens: Sequence[int]) -> None:
        """Record the most recently used token sequence for ``model_key``.

        Replaces any previously-stored sequence for that key — the
        engine has only one live KV cache per model, so storing the
        "latest" one is the semantically correct summary.
        """
        if not tokens:
            return
        payload = tuple(tokens)
        with self._lock:
            self._cache[model_key] = payload
            self._cache.move_to_end(model_key)
            while len(self._cache) > self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("prompt cache evicted %s", evicted_key)

    def drop(self, model_key: str) -> None:
        """Forget ``model_key`` — call on unload so stale prefixes don't linger."""
        with self._lock:
            self._cache.pop(model_key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_singleton: PromptPrefixCache | None = None


def get_prompt_cache() -> PromptPrefixCache:
    """Return the process-wide singleton, creating it lazily."""
    global _singleton
    if _singleton is None:
        from hfl.config import config as _cfg

        _singleton = PromptPrefixCache(max_entries=max(1, _cfg.prompt_cache_max_entries))
    return _singleton


def reset_prompt_cache() -> None:
    """Test hook."""
    global _singleton
    _singleton = None

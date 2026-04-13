# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Failover engine that wraps multiple backends and tries them in order."""

from __future__ import annotations

import logging
import threading
from typing import Iterator

from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)

logger = logging.getLogger(__name__)


class FailoverEngine(InferenceEngine):
    """Wraps multiple InferenceEngine instances with automatic failover.

    Tries the current (sticky) engine first. On failure, iterates through
    the remaining engines in order. The last successful engine becomes the
    new sticky default.

    Thread-safe: the internal ``_current_index`` is guarded by a lock.
    """

    def __init__(self, engines: list[InferenceEngine]) -> None:
        if not engines:
            raise ValueError("FailoverEngine requires at least one engine")
        self._engines = list(engines)
        self._current_index: int = 0
        self._lock = threading.Lock()

    # -- helpers --------------------------------------------------------------

    def _get_current_index(self) -> int:
        with self._lock:
            return self._current_index

    def _set_current_index(self, idx: int) -> None:
        with self._lock:
            self._current_index = idx

    def _ordered_indices(self) -> list[int]:
        """Return engine indices starting from the current sticky one."""
        cur = self._get_current_index()
        n = len(self._engines)
        return [(cur + i) % n for i in range(n)]

    # -- InferenceEngine interface --------------------------------------------

    def load(self, model_path: str, **kwargs) -> None:
        """Load the model on **all** engines."""
        for engine in self._engines:
            engine.load(model_path, **kwargs)

    def unload(self) -> None:
        """Unload the model from **all** engines."""
        for engine in self._engines:
            engine.unload()

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        last_error: Exception | None = None
        for idx in self._ordered_indices():
            engine = self._engines[idx]
            try:
                result = engine.generate(prompt, config)
                self._set_current_index(idx)
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Engine %d (%s) failed on generate, failing over: %s",
                    idx,
                    type(engine).__name__,
                    exc,
                )
        raise last_error  # type: ignore[misc]

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        last_error: Exception | None = None
        for idx in self._ordered_indices():
            engine = self._engines[idx]
            try:
                # Materialise the first token to detect immediate errors,
                # then yield it along with the rest of the stream.
                it = engine.generate_stream(prompt, config)
                first = next(it)
                self._set_current_index(idx)
                yield first
                yield from it
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Engine %d (%s) failed on generate_stream, failing over: %s",
                    idx,
                    type(engine).__name__,
                    exc,
                )
        raise last_error  # type: ignore[misc]

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        last_error: Exception | None = None
        for idx in self._ordered_indices():
            engine = self._engines[idx]
            try:
                if tools is not None:
                    try:
                        result = engine.chat(messages, config, tools=tools)
                    except TypeError:
                        # Legacy engine without ``tools`` kwarg
                        result = engine.chat(messages, config)
                else:
                    result = engine.chat(messages, config)
                self._set_current_index(idx)
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Engine %d (%s) failed on chat, failing over: %s",
                    idx,
                    type(engine).__name__,
                    exc,
                )
        raise last_error  # type: ignore[misc]

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        last_error: Exception | None = None
        for idx in self._ordered_indices():
            engine = self._engines[idx]
            try:
                if tools is not None:
                    try:
                        it = engine.chat_stream(messages, config, tools=tools)
                    except TypeError:
                        it = engine.chat_stream(messages, config)
                else:
                    it = engine.chat_stream(messages, config)
                first = next(it)
                self._set_current_index(idx)
                yield first
                yield from it
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Engine %d (%s) failed on chat_stream, failing over: %s",
                    idx,
                    type(engine).__name__,
                    exc,
                )
        raise last_error  # type: ignore[misc]

    @property
    def model_name(self) -> str:
        return self._engines[self._get_current_index()].model_name

    @property
    def is_loaded(self) -> bool:
        return any(e.is_loaded for e in self._engines)

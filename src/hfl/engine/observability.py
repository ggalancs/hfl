# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Engine observability module.

Provides unified observability hooks for inference engines including:
- Structured logging
- Event emission
- Metrics recording
- Performance tracking
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from hfl.engine.base import GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics for a single inference operation.

    Attributes:
        start_time: Unix timestamp when inference started
        end_time: Unix timestamp when inference completed
        tokens_prompt: Number of tokens in the prompt
        tokens_generated: Number of tokens generated
        model_name: Name of the model used
        operation: Type of operation (generate, chat, etc.)
    """

    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    tokens_prompt: int = 0
    tokens_generated: int = 0
    model_name: str = ""
    operation: str = "generate"

    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        if self.end_time == 0:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        return self.duration_seconds * 1000

    @property
    def tokens_per_second(self) -> float:
        """Tokens generated per second (throughput)."""
        duration = self.duration_seconds
        if duration <= 0 or self.tokens_generated <= 0:
            return 0.0
        return self.tokens_generated / duration

    @property
    def time_to_first_token_ms(self) -> float | None:
        """Time to first token in ms (placeholder for future streaming support)."""
        return None

    def complete(self, result: "GenerationResult | None" = None) -> None:
        """Mark inference as complete and extract result metrics."""
        self.end_time = time.time()
        if result:
            self.tokens_prompt = result.tokens_prompt
            self.tokens_generated = result.tokens_generated

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/export."""
        return {
            "model": self.model_name,
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "tokens_prompt": self.tokens_prompt,
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


class EngineObserver:
    """Observer for engine operations.

    Provides hooks for logging, metrics, and events around engine operations.
    Designed to be used as a context manager for clean resource management.
    """

    def __init__(self, model_name: str = "unknown"):
        """Initialize observer.

        Args:
            model_name: Name of the model being observed
        """
        self.model_name = model_name
        self._enabled = True

    @contextmanager
    def observe_inference(
        self,
        operation: str = "generate",
        emit_events: bool = True,
        record_metrics: bool = True,
    ) -> Generator[InferenceMetrics, None, None]:
        """Context manager for observing inference operations.

        Args:
            operation: Type of operation (generate, chat, etc.)
            emit_events: Whether to emit events
            record_metrics: Whether to record metrics

        Yields:
            InferenceMetrics object to be updated with results

        Example:
            with observer.observe_inference("chat") as metrics:
                result = engine.chat(messages, config)
                metrics.complete(result)
        """
        metrics = InferenceMetrics(
            model_name=self.model_name,
            operation=operation,
        )

        if emit_events and self._enabled:
            self._emit_generation_started(operation)

        try:
            yield metrics
        except Exception as e:
            # Mark completion even on failure to get accurate duration
            if metrics.end_time == 0:
                metrics.end_time = time.time()
            if emit_events and self._enabled:
                self._emit_generation_failed(operation, str(e))
            logger.error(
                f"Inference failed",
                extra={
                    "model": self.model_name,
                    "operation": operation,
                    "error": str(e),
                    "duration_ms": metrics.duration_ms,
                },
            )
            raise
        else:
            # Mark completion if not already done
            if metrics.end_time == 0:
                metrics.end_time = time.time()
            if record_metrics and self._enabled:
                self._record_generation_metrics(metrics)
            if emit_events and self._enabled:
                self._emit_generation_completed(metrics)
            logger.info(
                f"Inference completed",
                extra=metrics.to_dict(),
            )

    @contextmanager
    def observe_model_load(
        self,
        model_path: str,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager for observing model loading.

        Args:
            model_path: Path to the model being loaded

        Yields:
            Dictionary for storing load metadata

        Example:
            with observer.observe_model_load("/path/to/model") as load_info:
                engine.load(model_path)
                load_info["backend"] = "llama-cpp"
        """
        load_info: dict[str, Any] = {
            "model_path": model_path,
            "model_name": self.model_name,
            "start_time": time.time(),
        }

        if self._enabled:
            self._emit_model_loading(model_path)

        try:
            yield load_info
        except Exception as e:
            if self._enabled:
                self._emit_model_load_failed(model_path, str(e))
            logger.error(
                f"Model load failed: {model_path}",
                extra={
                    "model_path": model_path,
                    "error": str(e),
                },
            )
            raise
        else:
            load_info["end_time"] = time.time()
            load_info["duration_ms"] = (load_info["end_time"] - load_info["start_time"]) * 1000

            if self._enabled:
                self._record_model_load_metrics(load_info)
                self._emit_model_loaded(load_info)

            logger.info(
                f"Model loaded: {self.model_name}",
                extra={
                    "model_path": model_path,
                    "duration_ms": round(load_info["duration_ms"], 2),
                    **{k: v for k, v in load_info.items() if k not in ["start_time", "end_time", "model_path"]},
                },
            )

    def observe_model_unload(self) -> None:
        """Record model unload event."""
        if self._enabled:
            self._emit_model_unloaded()
            self._record_model_unload()
        logger.info("Model unloaded: %s", self.model_name)

    def _emit_generation_started(self, operation: str) -> None:
        """Emit generation started event."""
        try:
            from hfl.events import EventType, emit

            emit(
                EventType.GENERATION_STARTED,
                source="engine",
                model=self.model_name,
                operation=operation,
            )
        except ImportError:
            pass

    def _emit_generation_completed(self, metrics: InferenceMetrics) -> None:
        """Emit generation completed event."""
        try:
            from hfl.events import EventType, emit

            emit(
                EventType.GENERATION_COMPLETED,
                source="engine",
                **metrics.to_dict(),
            )
        except ImportError:
            pass

    def _emit_generation_failed(self, operation: str, error: str) -> None:
        """Emit generation failed event."""
        try:
            from hfl.events import EventType, emit

            emit(
                EventType.GENERATION_FAILED,
                source="engine",
                model=self.model_name,
                operation=operation,
                error=error,
            )
        except ImportError:
            pass

    def _emit_model_loading(self, model_path: str) -> None:
        """Emit model loading event."""
        try:
            from hfl.events import EventType, emit

            emit(
                EventType.MODEL_LOADING,
                source="engine",
                model=self.model_name,
                model_path=model_path,
            )
        except ImportError:
            pass

    def _emit_model_loaded(self, load_info: dict[str, Any]) -> None:
        """Emit model loaded event."""
        try:
            from hfl.events import EventType, emit

            emit(
                EventType.MODEL_LOADED,
                source="engine",
                model=self.model_name,
                duration_ms=load_info.get("duration_ms", 0),
            )
        except ImportError:
            pass

    def _emit_model_load_failed(self, model_path: str, error: str) -> None:
        """Emit model load failed event."""
        try:
            from hfl.events import EventType, emit

            emit(
                EventType.MODEL_LOAD_FAILED,
                source="engine",
                model=self.model_name,
                model_path=model_path,
                error=error,
            )
        except ImportError:
            pass

    def _emit_model_unloaded(self) -> None:
        """Emit model unloaded event."""
        try:
            from hfl.events import EventType, emit

            emit(
                EventType.MODEL_UNLOADED,
                source="engine",
                model=self.model_name,
            )
        except ImportError:
            pass

    def _record_generation_metrics(self, metrics: InferenceMetrics) -> None:
        """Record generation metrics."""
        try:
            from hfl.metrics import get_metrics

            m = get_metrics()
            m.record_generation(
                duration_ms=metrics.duration_ms,
                tokens_in=metrics.tokens_prompt,
                tokens_out=metrics.tokens_generated,
            )
        except ImportError:
            pass

    def _record_model_load_metrics(self, load_info: dict[str, Any]) -> None:
        """Record model load metrics."""
        try:
            from hfl.metrics import get_metrics

            m = get_metrics()
            m.record_model_load(
                model_name=self.model_name,
                duration_ms=load_info.get("duration_ms", 0),
            )
        except ImportError:
            pass

    def _record_model_unload(self) -> None:
        """Record model unload metric."""
        try:
            from hfl.metrics import get_metrics

            m = get_metrics()
            m.record_model_unload()
        except ImportError:
            pass


def get_observer(model_name: str = "unknown") -> EngineObserver:
    """Get an engine observer instance.

    Args:
        model_name: Name of the model to observe

    Returns:
        EngineObserver instance
    """
    return EngineObserver(model_name)

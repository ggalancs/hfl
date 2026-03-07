# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Observability setup - connects events to metrics.

This module wires up the event system to automatically record metrics
when important events occur (model loads, generation completions, errors).
"""

import logging
from typing import Any

from hfl.events import EventType, on
from hfl.metrics import get_metrics

logger = logging.getLogger(__name__)

_listeners_registered = False


def setup_event_listeners() -> None:
    """Register event listeners that record metrics.

    This should be called once at server startup to connect
    the event system to the metrics system.

    Safe to call multiple times - will only register once.
    """
    global _listeners_registered

    if _listeners_registered:
        return

    metrics = get_metrics()

    @on(EventType.MODEL_LOADED)
    def on_model_loaded(event: Any) -> None:
        """Record model load in metrics."""
        model_name = event.data.get("model", "unknown")
        duration_ms = event.data.get("duration_ms", 0)
        metrics.record_model_load(model_name, duration_ms)
        logger.debug(f"Recorded model load: {model_name} ({duration_ms:.0f}ms)")

    @on(EventType.MODEL_UNLOADED)
    def on_model_unloaded(event: Any) -> None:
        """Record model unload in metrics."""
        metrics.record_model_unload()
        logger.debug("Recorded model unload")

    @on(EventType.GENERATION_COMPLETED)
    def on_generation_completed(event: Any) -> None:
        """Record generation completion in metrics."""
        duration_ms = event.data.get("duration_ms", 0)
        tokens_in = event.data.get("tokens_prompt", 0)
        tokens_out = event.data.get("tokens_generated", 0)
        metrics.record_generation(duration_ms, tokens_in, tokens_out)
        logger.debug(f"Recorded generation: {tokens_out} tokens in {duration_ms:.0f}ms")

    @on(EventType.GENERATION_FAILED)
    def on_generation_failed(event: Any) -> None:
        """Record generation error in metrics."""
        error = event.data.get("error")
        error_type = type(error).__name__ if error else "Unknown"
        metrics.record_error(error_type)
        logger.debug(f"Recorded generation error: {error_type}")

    _listeners_registered = True
    logger.info("Event listeners registered for metrics")


def reset_event_listeners() -> None:
    """Reset listener registration state (for testing)."""
    global _listeners_registered
    _listeners_registered = False

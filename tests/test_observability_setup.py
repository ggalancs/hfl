# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for observability setup."""

import pytest

from hfl.core.observability_setup import reset_event_listeners, setup_event_listeners
from hfl.events import Event, EventType, emit, get_event_bus
from hfl.metrics import get_metrics


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before each test."""
    reset_event_listeners()
    get_event_bus().clear()
    get_metrics().reset()
    yield
    reset_event_listeners()
    get_event_bus().clear()


class TestSetupEventListeners:
    """Tests for setup_event_listeners function."""

    def test_setup_registers_listeners(self):
        """setup_event_listeners should register without error."""
        # Should not raise
        setup_event_listeners()

    def test_setup_is_idempotent(self):
        """Calling setup multiple times should be safe."""
        setup_event_listeners()
        setup_event_listeners()
        setup_event_listeners()

        # Should still work
        metrics = get_metrics()
        initial_loads = metrics.model_loads

        emit(EventType.MODEL_LOADED, model="test", duration_ms=100)

        assert metrics.model_loads == initial_loads + 1

    def test_model_loaded_event_records_metric(self):
        """MODEL_LOADED event should record model load metric."""
        setup_event_listeners()
        metrics = get_metrics()

        initial_loads = metrics.model_loads

        emit(EventType.MODEL_LOADED, model="test-model", duration_ms=500)

        assert metrics.model_loads == initial_loads + 1

    def test_model_unloaded_event_records_metric(self):
        """MODEL_UNLOADED event should record model unload metric."""
        setup_event_listeners()
        metrics = get_metrics()

        initial_unloads = metrics.model_unloads

        emit(EventType.MODEL_UNLOADED, model="test-model")

        assert metrics.model_unloads == initial_unloads + 1

    def test_generation_completed_event_records_metric(self):
        """GENERATION_COMPLETED event should record generation metric."""
        setup_event_listeners()
        metrics = get_metrics()

        initial_tokens = metrics.tokens_generated

        emit(EventType.GENERATION_COMPLETED, duration_ms=1000, tokens_prompt=50, tokens_generated=100)

        assert metrics.tokens_generated == initial_tokens + 100
        assert metrics.tokens_input >= 50

    def test_generation_failed_event_records_error(self):
        """GENERATION_FAILED event should record error metric."""
        setup_event_listeners()
        metrics = get_metrics()

        emit(EventType.GENERATION_FAILED, error=ValueError("test error"))

        assert "ValueError" in metrics.errors_by_type
        assert metrics.errors_by_type["ValueError"] >= 1

    def test_generation_failed_unknown_error(self):
        """GENERATION_FAILED with no error should record Unknown."""
        setup_event_listeners()
        metrics = get_metrics()

        emit(EventType.GENERATION_FAILED)

        assert "Unknown" in metrics.errors_by_type


class TestResetEventListeners:
    """Tests for reset_event_listeners function."""

    def test_reset_allows_reregistration(self):
        """After reset, listeners can be registered again."""
        setup_event_listeners()
        reset_event_listeners()
        setup_event_listeners()

        # Should work
        metrics = get_metrics()
        emit(EventType.MODEL_LOADED, model="test", duration_ms=100)

        assert metrics.model_loads >= 1

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for event system."""

import pytest

from hfl.events import (
    Event,
    EventBus,
    EventType,
    emit,
    get_event_bus,
    on,
    reset_event_bus,
)


class TestEvent:
    """Tests for Event dataclass."""

    def test_basic_creation(self):
        """Should create basic event."""
        event = Event(type=EventType.MODEL_LOADED)

        assert event.type == EventType.MODEL_LOADED
        assert event.data == {}
        assert event.timestamp > 0
        assert event.source is None

    def test_with_data(self):
        """Should include event data."""
        event = Event(
            type=EventType.MODEL_LOADED,
            data={"model": "llama-7b", "duration_ms": 5000},
            source="engine",
        )

        assert event.data["model"] == "llama-7b"
        assert event.data["duration_ms"] == 5000
        assert event.source == "engine"

    def test_str_representation(self):
        """Should have string representation."""
        event = Event(type=EventType.ERROR, data={"message": "test"})
        s = str(event)

        assert "error" in s
        assert "message" in s


class TestEventBus:
    """Tests for EventBus class."""

    @pytest.fixture
    def bus(self):
        """Create fresh event bus."""
        return EventBus()

    def test_subscribe_and_publish(self, bus):
        """Should deliver events to subscribers."""
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.MODEL_LOADED, handler)
        bus.publish(Event(type=EventType.MODEL_LOADED, data={"test": True}))

        assert len(received) == 1
        assert received[0].data["test"] is True

    def test_multiple_handlers(self, bus):
        """Should deliver to all handlers."""
        results = []

        def handler1(event):
            results.append("h1")

        def handler2(event):
            results.append("h2")

        bus.subscribe(EventType.ERROR, handler1)
        bus.subscribe(EventType.ERROR, handler2)
        bus.publish(Event(type=EventType.ERROR))

        assert "h1" in results
        assert "h2" in results

    def test_global_handler(self, bus):
        """Global handler should receive all events."""
        received = []

        def handler(event):
            received.append(event.type)

        bus.subscribe(None, handler)
        bus.publish(Event(type=EventType.MODEL_LOADED))
        bus.publish(Event(type=EventType.ERROR))

        assert EventType.MODEL_LOADED in received
        assert EventType.ERROR in received

    def test_unsubscribe(self, bus):
        """Should stop receiving after unsubscribe."""
        received = []

        def handler(event):
            received.append(event)

        unsub = bus.subscribe(EventType.MODEL_LOADED, handler)
        bus.publish(Event(type=EventType.MODEL_LOADED))
        assert len(received) == 1

        unsub()
        bus.publish(Event(type=EventType.MODEL_LOADED))
        assert len(received) == 1  # No new events

    def test_handler_error_isolated(self, bus):
        """Handler errors should not affect other handlers."""
        results = []

        def bad_handler(event):
            raise ValueError("oops")

        def good_handler(event):
            results.append("ok")

        bus.subscribe(EventType.ERROR, bad_handler)
        bus.subscribe(EventType.ERROR, good_handler)

        # Should not raise
        bus.publish(Event(type=EventType.ERROR))
        assert "ok" in results

    def test_clear(self, bus):
        """clear() should remove all handlers."""
        called = []

        def handler(event):
            called.append(event)

        bus.subscribe(EventType.MODEL_LOADED, handler)
        bus.subscribe(None, handler)

        bus.clear()
        bus.publish(Event(type=EventType.MODEL_LOADED))

        assert len(called) == 0

    def test_handler_count(self, bus):
        """Should track handler count."""
        assert bus.handler_count == 0

        bus.subscribe(EventType.MODEL_LOADED, lambda e: None)
        bus.subscribe(EventType.ERROR, lambda e: None)
        bus.subscribe(None, lambda e: None)

        assert bus.handler_count == 3


class TestGetEventBus:
    """Tests for get_event_bus singleton."""

    def test_returns_same_instance(self):
        """Should return same instance."""
        reset_event_bus()

        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

    def test_reset_clears_instance(self):
        """reset_event_bus should create new instance."""
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()

        assert bus1 is not bus2


class TestEmitFunction:
    """Tests for emit convenience function."""

    def test_emits_event(self):
        """emit() should publish event."""
        reset_event_bus()
        received = []

        get_event_bus().subscribe(
            EventType.CUSTOM,
            lambda e: received.append(e),
        )

        emit(EventType.CUSTOM, source="test", message="hello")

        assert len(received) == 1
        assert received[0].data["message"] == "hello"
        assert received[0].source == "test"


class TestOnDecorator:
    """Tests for @on decorator."""

    def test_registers_handler(self):
        """@on should register handler."""
        reset_event_bus()
        results = []

        @on(EventType.SERVER_STARTED)
        def handle_start(event):
            results.append("started")

        emit(EventType.SERVER_STARTED)

        assert "started" in results

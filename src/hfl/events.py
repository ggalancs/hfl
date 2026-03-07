# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Event system for HFL.

Provides a simple event bus for decoupled communication between components.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from hfl.logging_config import get_logger

logger = get_logger()


class EventType(Enum):
    """Types of events emitted by HFL."""

    # Model lifecycle
    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"
    MODEL_LOAD_FAILED = "model_load_failed"

    # Generation
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_FAILED = "generation_failed"
    GENERATION_CANCELLED = "generation_cancelled"

    # Server
    SERVER_STARTED = "server_started"
    SERVER_STOPPED = "server_stopped"
    REQUEST_RECEIVED = "request_received"
    REQUEST_COMPLETED = "request_completed"

    # Errors
    ERROR = "error"
    WARNING = "warning"

    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Event data container."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str | None = None

    def __str__(self) -> str:
        return f"Event({self.type.value}, data={self.data})"


EventHandler = Callable[[Event], None]


class EventBus:
    """Simple event bus for decoupled communication.

    Thread-safe implementation for subscribing to and publishing events.
    """

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._lock = threading.Lock()

    def subscribe(
        self,
        event_type: EventType | None,
        handler: EventHandler,
    ) -> Callable[[], None]:
        """Subscribe to events.

        Args:
            event_type: Type of events to subscribe to.
                       If None, handler receives all events.
            handler: Function to call when event occurs

        Returns:
            Unsubscribe function
        """
        with self._lock:
            if event_type is None:
                self._global_handlers.append(handler)

                def unsubscribe():
                    with self._lock:
                        if handler in self._global_handlers:
                            self._global_handlers.remove(handler)

            else:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []
                self._handlers[event_type].append(handler)

                def unsubscribe():
                    with self._lock:
                        handlers = self._handlers.get(event_type, [])
                        if handler in handlers:
                            handlers.remove(handler)

            return unsubscribe

    def publish(self, event: Event) -> None:
        """Publish event to all subscribers.

        Calls handlers synchronously. Errors in handlers are logged
        but don't prevent other handlers from being called.

        Args:
            event: Event to publish
        """
        with self._lock:
            handlers = list(self._global_handlers)
            handlers.extend(self._handlers.get(event.type, []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler failed for {event.type}: {e}")

    def publish_async(self, event: Event) -> None:
        """Publish event asynchronously in a background thread.

        Args:
            event: Event to publish
        """
        thread = threading.Thread(target=self.publish, args=(event,), daemon=True)
        thread.start()

    def clear(self) -> None:
        """Remove all event handlers."""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers."""
        with self._lock:
            count = len(self._global_handlers)
            for handlers in self._handlers.values():
                count += len(handlers)
            return count


# Singleton access delegated to container for unified management


def get_event_bus() -> EventBus:
    """Get the singleton EventBus instance.

    Returns:
        EventBus instance
    """
    from hfl.core.container import get_event_bus as _get_event_bus

    return _get_event_bus()


def reset_event_bus() -> None:
    """Reset the event bus singleton (for testing)."""
    from hfl.core.container import get_container

    get_container().event_bus.reset()


# Convenience functions


def emit(event_type: EventType, source: str | None = None, **data: Any) -> None:
    """Emit an event.

    Args:
        event_type: Type of event
        source: Source component name
        **data: Event data
    """
    event = Event(type=event_type, data=data, source=source)
    get_event_bus().publish(event)


def emit_async(event_type: EventType, source: str | None = None, **data: Any) -> None:
    """Emit an event asynchronously.

    Args:
        event_type: Type of event
        source: Source component name
        **data: Event data
    """
    event = Event(type=event_type, data=data, source=source)
    get_event_bus().publish_async(event)


def on(event_type: EventType | None = None) -> Callable[[EventHandler], EventHandler]:
    """Decorator to subscribe to events.

    Args:
        event_type: Type of events to subscribe to.
                   If None, handler receives all events.

    Returns:
        Decorator function

    Example:
        @on(EventType.MODEL_LOADED)
        def handle_model_loaded(event):
            print(f"Model loaded: {event.data['model']}")
    """

    def decorator(func: EventHandler) -> EventHandler:
        get_event_bus().subscribe(event_type, func)
        return func

    return decorator

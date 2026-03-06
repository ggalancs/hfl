# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Dependency injection container for HFL.

This module provides a unified singleton pattern and dependency injection
container for managing global state throughout the application.

Usage:
    from hfl.core import get_config, get_registry, get_state

    config = get_config()
    registry = get_registry()
    state = get_state()

For testing, use reset_container() to clear all singletons:
    from hfl.core import reset_container

    def test_something():
        reset_container()
        # Test with fresh state
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from hfl.api.state import ServerState
    from hfl.config import HFLConfig
    from hfl.events import EventBus
    from hfl.metrics import Metrics
    from hfl.models.registry import ModelRegistry

T = TypeVar("T")


class Singleton(Generic[T]):
    """Thread-safe lazy singleton.

    This class provides a thread-safe way to create singletons that are
    only instantiated when first accessed.

    Example:
        config_singleton = Singleton(lambda: HFLConfig())
        config = config_singleton.get()  # Creates instance on first call
        config2 = config_singleton.get()  # Returns same instance

        config_singleton.reset()  # Clear the instance
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        """Initialize with a factory function.

        Args:
            factory: Callable that creates the singleton instance.
        """
        self._factory = factory
        self._instance: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        """Get or create the singleton instance.

        Returns:
            The singleton instance.
        """
        if self._instance is None:
            with self._lock:
                # Double-check locking pattern
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def reset(self) -> None:
        """Clear the singleton instance.

        This is primarily useful for testing.
        """
        with self._lock:
            self._instance = None

    @property
    def is_initialized(self) -> bool:
        """Check if the singleton has been initialized."""
        return self._instance is not None


def _create_config() -> "HFLConfig":
    """Factory for HFLConfig."""
    from hfl.config import HFLConfig

    return HFLConfig()


def _create_registry() -> "ModelRegistry":
    """Factory for ModelRegistry."""
    from hfl.models.registry import ModelRegistry

    return ModelRegistry()


def _create_event_bus() -> "EventBus":
    """Factory for EventBus."""
    from hfl.events import EventBus

    return EventBus()


def _create_state() -> "ServerState":
    """Factory for ServerState."""
    from hfl.api.state import ServerState

    return ServerState()


def _create_metrics() -> "Metrics":
    """Factory for Metrics."""
    from hfl.metrics import Metrics

    return Metrics()


@dataclass
class Container:
    """Central dependency injection container for HFL.

    This container holds all singleton instances used throughout the application.
    Each singleton is lazily initialized when first accessed.

    Example:
        container = Container()
        config = container.config.get()
        registry = container.registry.get()

        # Reset all for testing
        container.reset_all()
    """

    config: Singleton["HFLConfig"] = field(
        default_factory=lambda: Singleton(_create_config)
    )
    registry: Singleton["ModelRegistry"] = field(
        default_factory=lambda: Singleton(_create_registry)
    )
    event_bus: Singleton["EventBus"] = field(
        default_factory=lambda: Singleton(_create_event_bus)
    )
    state: Singleton["ServerState"] = field(
        default_factory=lambda: Singleton(_create_state)
    )
    metrics: Singleton["Metrics"] = field(
        default_factory=lambda: Singleton(_create_metrics)
    )

    def reset_all(self) -> None:
        """Reset all singletons.

        This is primarily useful for testing to ensure clean state.
        """
        self.config.reset()
        self.registry.reset()
        self.event_bus.reset()
        self.state.reset()
        self.metrics.reset()


# Global container instance
_container: Container | None = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """Get the global container instance.

    Returns:
        The global Container instance.
    """
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container()
    return _container


def reset_container() -> None:
    """Reset the global container and all singletons.

    This is primarily useful for testing.
    """
    global _container
    with _container_lock:
        if _container is not None:
            _container.reset_all()
        _container = None


# Convenience functions for accessing common singletons


def get_config() -> "HFLConfig":
    """Get the global config instance.

    Returns:
        The global HFLConfig instance.
    """
    return get_container().config.get()


def get_registry() -> "ModelRegistry":
    """Get the global model registry instance.

    Returns:
        The global ModelRegistry instance.
    """
    return get_container().registry.get()


def get_event_bus() -> "EventBus":
    """Get the global event bus instance.

    Returns:
        The global EventBus instance.
    """
    return get_container().event_bus.get()


def get_state() -> "ServerState":
    """Get the global server state instance.

    Returns:
        The global ServerState instance.
    """
    return get_container().state.get()


def get_metrics() -> "Metrics":
    """Get the global metrics collector instance.

    Returns:
        The global Metrics instance.
    """
    return get_container().metrics.get()

# SPDX-License-Identifier: Apache-2.0
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
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from hfl.api.rate_limit import RateLimiter
    from hfl.api.state import ServerState
    from hfl.config import HFLConfig
    from hfl.engine.dispatcher import CpuTurn, DispatcherSnapshot, InferenceDispatcher
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
    """Factory for HFLConfig.

    Returns the global config instance from hfl.config module
    to ensure consistency across the application.
    """
    from hfl.config import config

    return config


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


def _create_rate_limiter() -> "RateLimiter":
    """Factory for RateLimiter: the in-memory one, sized by config (HFL is
    a single process by design, so the limiter lives in it)."""
    from hfl.api.rate_limit import InMemoryRateLimiter
    from hfl.config import config

    return InMemoryRateLimiter(
        requests_per_window=config.rate_limit_requests,
        window_seconds=config.rate_limit_window,
    )


def _create_dispatcher() -> "InferenceDispatcher":
    """Factory for the in-server inference dispatcher (spec §5.3)."""
    from hfl.engine.dispatcher import build_default_dispatcher

    return build_default_dispatcher()


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

    config: Singleton["HFLConfig"] = field(default_factory=lambda: Singleton(_create_config))
    registry: Singleton["ModelRegistry"] = field(
        default_factory=lambda: Singleton(_create_registry)
    )
    event_bus: Singleton["EventBus"] = field(default_factory=lambda: Singleton(_create_event_bus))
    state: Singleton["ServerState"] = field(default_factory=lambda: Singleton(_create_state))
    metrics: Singleton["Metrics"] = field(default_factory=lambda: Singleton(_create_metrics))
    rate_limiter: Singleton["RateLimiter"] = field(
        default_factory=lambda: Singleton(_create_rate_limiter)
    )
    dispatcher: Singleton["InferenceDispatcher"] = field(
        default_factory=lambda: Singleton(_create_dispatcher)
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
        self.rate_limiter.reset()
        self.dispatcher.reset()


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
    global _container, _CPU_TURN
    with _container_lock:
        if _container is not None:
            _container.reset_all()
        _container = None
        _CPU_TURN = None


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


def get_rate_limiter() -> "RateLimiter":
    """Get the global rate limiter instance.

    Returns:
        The global RateLimiter instance.
    """
    return get_container().rate_limiter.get()


def get_dispatcher() -> "InferenceDispatcher":
    """Get the global inference dispatcher instance (spec §5.3).

    Returns:
        The global :class:`InferenceDispatcher` singleton.
    """
    return get_container().dispatcher.get()


def dispatcher_for(engine: object | None) -> "InferenceDispatcher":
    """The dispatcher that schedules inference on ``engine``.

    Engines that keep one non-reentrant model instance (llama-cpp-python,
    Transformers, MLX) share the global dispatcher, clamped to one request
    at a time. An engine that serves requests concurrently
    (``supports_concurrent_inference``: llama-server, vLLM) gets its own,
    sized to its ``parallel_slots`` (or ``HFL_NUM_PARALLEL``), so its
    requests neither wait behind another model's nor are held to one.
    """
    # ``is True``: a mock (or any truthy non-bool) must not route an engine
    # away from the serialized queue.
    if engine is None:
        return get_dispatcher()
    concurrent = getattr(engine, "supports_concurrent_inference", False) is True
    independent = getattr(engine, "independent_instances", False) is True
    if not (concurrent or independent):
        return get_dispatcher()
    own = getattr(engine, "_hfl_dispatcher", None)
    if own is None:
        from hfl.engine.dispatcher import InferenceDispatcher

        cfg = get_config()
        # An engine whose instances are independent (llama.cpp) still runs
        # one call at a time on each: a queue of its own with one slot, so a
        # model no longer waits behind another one.
        slots = (
            int(getattr(engine, "parallel_slots", 0) or cfg.queue_max_inflight or 1)
            if concurrent
            else 1
        )
        # Models that each generate on every CPU core take turns (CpuTurn).
        cpu_bound = getattr(engine, "generates_on_all_cpu_cores", False) is True
        own = InferenceDispatcher(
            max_inflight=max(1, slots),
            max_queued=cfg.queue_max_size,
            acquire_timeout=cfg.queue_acquire_timeout_seconds,
            turn=_cpu_turn() if cpu_bound else None,
        )
        engine._hfl_dispatcher = own  # type: ignore[attr-defined]
        _ENGINE_DISPATCHERS.add(own)
    return own


# The one turn on the CPU cores all CPU-bound models share (created on
# first use; it is bound to no event loop).
_CPU_TURN: "CpuTurn | None" = None


def _cpu_turn() -> "CpuTurn":
    global _CPU_TURN
    if _CPU_TURN is None:
        from hfl.engine.dispatcher import CpuTurn

        _CPU_TURN = CpuTurn()
    return _CPU_TURN


# Every per-engine queue alive, for the totals /healthz and /metrics report.
_ENGINE_DISPATCHERS: "weakref.WeakSet[InferenceDispatcher]" = weakref.WeakSet()


def dispatcher_totals() -> "DispatcherSnapshot":
    """The global queue and every per-engine queue, added up: what the
    server as a whole is doing (``/healthz``, ``X-Queue-Depth``,
    ``/metrics``). The global queue alone under-reported the load once
    models had queues of their own."""
    from hfl.engine.dispatcher import DispatcherSnapshot

    snaps = [get_dispatcher().snapshot(), *(d.snapshot() for d in list(_ENGINE_DISPATCHERS))]
    return DispatcherSnapshot(
        max_inflight=sum(s.max_inflight for s in snaps),
        max_queued=sum(s.max_queued for s in snaps),
        in_flight=sum(s.in_flight for s in snaps),
        depth=sum(s.depth for s in snaps),
        accepted_total=sum(s.accepted_total for s in snaps),
        rejected_full_total=sum(s.rejected_full_total for s in snaps),
        rejected_timeout_total=sum(s.rejected_timeout_total for s in snaps),
    )

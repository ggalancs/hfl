# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for dependency injection container."""

import threading
from unittest.mock import MagicMock

from hfl.core.container import (
    Container,
    Singleton,
    get_config,
    get_container,
    get_event_bus,
    get_metrics,
    get_rate_limiter,
    get_registry,
    get_state,
    reset_container,
)


class TestSingleton:
    """Tests for Singleton class."""

    def test_lazy_initialization(self):
        """Singleton is not created until get() is called."""
        factory_called = False

        def factory():
            nonlocal factory_called
            factory_called = True
            return "instance"

        singleton = Singleton(factory)
        assert not factory_called

        singleton.get()
        assert factory_called

    def test_returns_same_instance(self):
        """get() returns the same instance every time."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"id": call_count}

        singleton = Singleton(factory)

        instance1 = singleton.get()
        instance2 = singleton.get()
        instance3 = singleton.get()

        assert instance1 is instance2
        assert instance2 is instance3
        assert call_count == 1

    def test_reset_clears_instance(self):
        """reset() clears the singleton instance."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"id": call_count}

        singleton = Singleton(factory)

        instance1 = singleton.get()
        singleton.reset()
        instance2 = singleton.get()

        assert instance1 is not instance2
        assert call_count == 2

    def test_is_initialized_property(self):
        """is_initialized returns correct state."""
        singleton = Singleton(lambda: "value")

        assert not singleton.is_initialized
        singleton.get()
        assert singleton.is_initialized

        singleton.reset()
        assert not singleton.is_initialized

    def test_thread_safety(self):
        """Singleton is thread-safe."""
        call_count = 0
        results = []

        def factory():
            nonlocal call_count
            call_count += 1
            return {"id": call_count}

        singleton = Singleton(factory)

        def get_instance():
            instance = singleton.get()
            results.append(instance)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert all(r is results[0] for r in results)
        assert call_count == 1

    def test_reset_thread_safety(self):
        """reset() is thread-safe."""
        singleton = Singleton(lambda: MagicMock())

        def reset_and_get():
            singleton.reset()
            singleton.get()

        threads = [threading.Thread(target=reset_and_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not raise any errors


class TestContainer:
    """Tests for Container class."""

    def test_has_all_singletons(self):
        """Container has all expected singletons."""
        container = Container()

        assert hasattr(container, "config")
        assert hasattr(container, "registry")
        assert hasattr(container, "event_bus")
        assert hasattr(container, "state")
        assert hasattr(container, "metrics")
        assert hasattr(container, "rate_limiter")

    def test_singletons_are_lazy(self):
        """Container singletons are not initialized until accessed."""
        container = Container()

        assert not container.config.is_initialized
        assert not container.registry.is_initialized
        assert not container.event_bus.is_initialized
        assert not container.state.is_initialized
        assert not container.metrics.is_initialized

    def test_reset_all_clears_all_singletons(self):
        """reset_all() clears all singletons."""
        container = Container()

        # Initialize all
        container.config.get()
        container.registry.get()
        container.event_bus.get()
        container.state.get()
        container.metrics.get()

        assert container.config.is_initialized
        assert container.registry.is_initialized

        container.reset_all()

        assert not container.config.is_initialized
        assert not container.registry.is_initialized
        assert not container.event_bus.is_initialized
        assert not container.state.is_initialized
        assert not container.metrics.is_initialized


class TestGetContainer:
    """Tests for get_container function."""

    def test_returns_container(self):
        """get_container returns a Container instance."""
        reset_container()
        container = get_container()
        assert isinstance(container, Container)

    def test_returns_same_instance(self):
        """get_container returns the same instance."""
        reset_container()

        container1 = get_container()
        container2 = get_container()

        assert container1 is container2

    def test_reset_creates_new_container(self):
        """reset_container creates new container on next get."""
        container1 = get_container()
        reset_container()
        container2 = get_container()

        assert container1 is not container2


class TestResetContainer:
    """Tests for reset_container function."""

    def test_resets_singletons(self):
        """reset_container resets all singletons."""
        container = get_container()
        container.config.get()

        assert container.config.is_initialized

        reset_container()
        new_container = get_container()

        assert not new_container.config.is_initialized

    def test_handles_none_container(self):
        """reset_container handles case where container is None."""
        reset_container()  # Ensure container is None
        reset_container()  # Should not raise


class TestConvenienceFunctions:
    """Tests for convenience access functions."""

    def test_get_config_returns_config(self):
        """get_config returns HFLConfig instance."""
        reset_container()
        config = get_config()

        from hfl.config import HFLConfig

        assert isinstance(config, HFLConfig)

    def test_get_config_returns_same_instance(self):
        """get_config returns same instance."""
        reset_container()

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_registry_returns_registry(self):
        """get_registry returns ModelRegistry instance."""
        reset_container()
        registry = get_registry()

        from hfl.models.registry import ModelRegistry

        assert isinstance(registry, ModelRegistry)

    def test_get_registry_returns_same_instance(self):
        """get_registry returns same instance."""
        reset_container()

        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_get_event_bus_returns_event_bus(self):
        """get_event_bus returns EventBus instance."""
        reset_container()
        bus = get_event_bus()

        from hfl.events import EventBus

        assert isinstance(bus, EventBus)

    def test_get_event_bus_returns_same_instance(self):
        """get_event_bus returns same instance."""
        reset_container()

        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

    def test_get_state_returns_state(self):
        """get_state returns ServerState instance."""
        reset_container()
        state = get_state()

        from hfl.api.state import ServerState

        assert isinstance(state, ServerState)

    def test_get_state_returns_same_instance(self):
        """get_state returns same instance."""
        reset_container()

        state1 = get_state()
        state2 = get_state()

        assert state1 is state2

    def test_get_metrics_returns_metrics(self):
        """get_metrics returns Metrics instance."""
        reset_container()
        metrics = get_metrics()

        from hfl.metrics import Metrics

        assert isinstance(metrics, Metrics)

    def test_get_metrics_returns_same_instance(self):
        """get_metrics returns same instance."""
        reset_container()

        metrics1 = get_metrics()
        metrics2 = get_metrics()

        assert metrics1 is metrics2


class TestFactoryFunctions:
    """Tests for internal factory functions."""

    def test_config_factory_creates_config(self):
        """_create_config creates HFLConfig."""
        from hfl.core.container import _create_config

        config = _create_config()

        from hfl.config import HFLConfig

        assert isinstance(config, HFLConfig)

    def test_registry_factory_creates_registry(self):
        """_create_registry creates ModelRegistry."""
        from hfl.core.container import _create_registry

        registry = _create_registry()

        from hfl.models.registry import ModelRegistry

        assert isinstance(registry, ModelRegistry)

    def test_event_bus_factory_creates_event_bus(self):
        """_create_event_bus creates EventBus."""
        from hfl.core.container import _create_event_bus

        bus = _create_event_bus()

        from hfl.events import EventBus

        assert isinstance(bus, EventBus)

    def test_state_factory_creates_state(self):
        """_create_state creates ServerState."""
        from hfl.core.container import _create_state

        state = _create_state()

        from hfl.api.state import ServerState

        assert isinstance(state, ServerState)

    def test_metrics_factory_creates_metrics(self):
        """_create_metrics creates Metrics."""
        from hfl.core.container import _create_metrics

        metrics = _create_metrics()

        from hfl.metrics import Metrics

        assert isinstance(metrics, Metrics)

    def test_rate_limiter_factory_creates_rate_limiter(self):
        """_create_rate_limiter creates RateLimiter."""
        from hfl.core.container import _create_rate_limiter

        limiter = _create_rate_limiter()

        from hfl.api.rate_limit import RateLimiter

        assert isinstance(limiter, RateLimiter)


class TestRateLimiterSingleton:
    """Tests for rate limiter singleton."""

    def test_get_rate_limiter_returns_limiter(self):
        """get_rate_limiter returns RateLimiter instance."""
        reset_container()
        limiter = get_rate_limiter()

        from hfl.api.rate_limit import RateLimiter

        assert isinstance(limiter, RateLimiter)

    def test_get_rate_limiter_returns_same_instance(self):
        """get_rate_limiter returns same instance."""
        reset_container()

        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()

        assert limiter1 is limiter2


class TestConfigCentralization:
    """Tests for config centralization."""

    def test_container_config_is_global_config(self):
        """Container config should be the same as global config."""
        reset_container()

        from hfl.config import config as global_config

        container_config = get_config()

        # Should be the exact same instance
        assert container_config is global_config

    def test_config_singleton_returns_global_instance(self):
        """Config singleton should return global instance."""
        reset_container()

        from hfl.config import config

        container = get_container()
        container_config = container.config.get()

        assert container_config is config

    def test_config_changes_reflected_in_container(self):
        """Changes to global config should be reflected in container."""
        reset_container()

        from hfl.config import config

        original_port = config.port
        try:
            config.port = 9999
            container_config = get_config()
            assert container_config.port == 9999
        finally:
            config.port = original_port

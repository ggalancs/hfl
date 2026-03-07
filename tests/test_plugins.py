# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for plugin system."""

from unittest.mock import MagicMock, patch

import pytest

import hfl.plugins as plugins_module
from hfl.plugins import (
    _CACHE_TTL,
    _lazy_import,
    clear_engine_cache,
    discover_engines,
    discover_tts_engines,
    get_engine_class,
    get_tts_engine_class,
    list_available_engines,
    list_available_tts_engines,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear engine cache before each test."""
    clear_engine_cache()
    yield
    clear_engine_cache()


class TestLazyImport:
    """Tests for _lazy_import helper."""

    def test_returns_callable(self):
        """_lazy_import returns a callable."""
        loader = _lazy_import("os.path", "join")
        assert callable(loader)

    def test_callable_imports_module(self):
        """Calling the loader imports the module."""
        loader = _lazy_import("os.path", "join")
        result = loader()

        import os.path

        assert result is os.path.join

    def test_callable_raises_on_missing_module(self):
        """Loader raises ImportError for missing module."""
        loader = _lazy_import("nonexistent.module", "class")

        with pytest.raises(ModuleNotFoundError):
            loader()

    def test_callable_raises_on_missing_attribute(self):
        """Loader raises AttributeError for missing attribute."""
        loader = _lazy_import("os", "nonexistent_attribute")

        with pytest.raises(AttributeError):
            loader()


class TestDiscoverEngines:
    """Tests for discover_engines function."""

    def test_returns_dict(self):
        """discover_engines returns a dictionary."""
        engines = discover_engines()
        assert isinstance(engines, dict)

    def test_includes_builtin_engines(self):
        """discover_engines includes built-in engines."""
        engines = discover_engines()

        assert "llama-cpp" in engines
        assert "transformers" in engines
        assert "vllm" in engines

    def test_builtin_engines_are_lazy(self):
        """Built-in engines are lazy loaders, not classes."""
        engines = discover_engines()

        # They should be callables but not types yet
        llama_cpp = engines["llama-cpp"]
        assert callable(llama_cpp)

    @patch("importlib.metadata.entry_points")
    def test_loads_entry_point_plugins(self, mock_entry_points):
        """discover_engines loads entry point plugins."""
        mock_ep = MagicMock()
        mock_ep.name = "custom-engine"
        mock_ep.load.return_value = MagicMock()
        mock_entry_points.return_value = [mock_ep]

        engines = discover_engines()

        assert "custom-engine" in engines
        mock_ep.load.assert_called_once()

    @patch("importlib.metadata.entry_points")
    def test_handles_failed_plugin_load(self, mock_entry_points):
        """discover_engines handles failed plugin loads gracefully."""
        mock_ep = MagicMock()
        mock_ep.name = "bad-plugin"
        mock_ep.load.side_effect = ImportError("Missing dependency")
        mock_entry_points.return_value = [mock_ep]

        # Should not raise
        engines = discover_engines()

        # Built-in engines should still be present
        assert "llama-cpp" in engines
        # Bad plugin should not be present
        assert "bad-plugin" not in engines

    @patch("importlib.metadata.entry_points")
    def test_handles_entry_points_exception(self, mock_entry_points):
        """discover_engines handles ImportError from entry_points."""
        mock_entry_points.side_effect = ImportError("entry_points error")

        # Should not raise
        engines = discover_engines()

        # Built-in engines should still be present
        assert "llama-cpp" in engines


class TestDiscoverTTSEngines:
    """Tests for discover_tts_engines function."""

    def test_returns_dict(self):
        """discover_tts_engines returns a dictionary."""
        engines = discover_tts_engines()
        assert isinstance(engines, dict)

    def test_includes_builtin_tts_engines(self):
        """discover_tts_engines includes built-in TTS engines."""
        engines = discover_tts_engines()

        assert "bark" in engines
        assert "coqui" in engines

    def test_builtin_tts_engines_are_lazy(self):
        """Built-in TTS engines are lazy loaders."""
        engines = discover_tts_engines()

        bark = engines["bark"]
        assert callable(bark)

    @patch("importlib.metadata.entry_points")
    def test_loads_tts_entry_point_plugins(self, mock_entry_points):
        """discover_tts_engines loads entry point plugins."""
        mock_ep = MagicMock()
        mock_ep.name = "custom-tts"
        mock_ep.load.return_value = MagicMock()
        mock_entry_points.return_value = [mock_ep]

        engines = discover_tts_engines()

        assert "custom-tts" in engines

    @patch("importlib.metadata.entry_points")
    def test_handles_failed_tts_plugin_load(self, mock_entry_points):
        """discover_tts_engines handles failed plugin loads gracefully."""
        mock_ep = MagicMock()
        mock_ep.name = "bad-tts-plugin"
        mock_ep.load.side_effect = ImportError("Missing dependency")
        mock_entry_points.return_value = [mock_ep]

        engines = discover_tts_engines()

        assert "bark" in engines
        assert "bad-tts-plugin" not in engines


class TestGetEngineClass:
    """Tests for get_engine_class function."""

    def test_raises_key_error_for_unknown_engine(self):
        """get_engine_class raises KeyError for unknown engine."""
        with pytest.raises(KeyError) as exc_info:
            get_engine_class("nonexistent-engine")

        assert "Unknown engine" in str(exc_info.value)
        assert "nonexistent-engine" in str(exc_info.value)

    @patch("hfl.plugins.discover_engines")
    def test_calls_lazy_loader(self, mock_discover):
        """get_engine_class calls lazy loader to get class."""
        mock_class = MagicMock()
        mock_loader = MagicMock(return_value=mock_class)
        mock_discover.return_value = {"test-engine": mock_loader}

        result = get_engine_class("test-engine")

        mock_loader.assert_called_once()
        assert result is mock_class

    @patch("hfl.plugins.discover_engines")
    def test_returns_class_directly_if_not_callable(self, mock_discover):
        """get_engine_class returns class directly if already a type."""

        class MockEngineClass:
            pass

        mock_discover.return_value = {"direct-engine": MockEngineClass}

        result = get_engine_class("direct-engine")

        assert result is MockEngineClass


class TestGetTTSEngineClass:
    """Tests for get_tts_engine_class function."""

    def test_raises_key_error_for_unknown_tts_engine(self):
        """get_tts_engine_class raises KeyError for unknown engine."""
        with pytest.raises(KeyError) as exc_info:
            get_tts_engine_class("nonexistent-tts")

        assert "Unknown TTS engine" in str(exc_info.value)
        assert "nonexistent-tts" in str(exc_info.value)

    @patch("hfl.plugins.discover_tts_engines")
    def test_calls_lazy_loader(self, mock_discover):
        """get_tts_engine_class calls lazy loader to get class."""
        mock_class = MagicMock()
        mock_loader = MagicMock(return_value=mock_class)
        mock_discover.return_value = {"test-tts": mock_loader}

        result = get_tts_engine_class("test-tts")

        mock_loader.assert_called_once()
        assert result is mock_class

    @patch("hfl.plugins.discover_tts_engines")
    def test_returns_class_directly_if_not_callable(self, mock_discover):
        """get_tts_engine_class returns class directly if already a type."""

        class MockTTSEngineClass:
            pass

        mock_discover.return_value = {"direct-tts": MockTTSEngineClass}

        result = get_tts_engine_class("direct-tts")

        assert result is MockTTSEngineClass


class TestListAvailableEngines:
    """Tests for list_available_engines function."""

    def test_returns_list(self):
        """list_available_engines returns a list."""
        engines = list_available_engines()
        assert isinstance(engines, list)

    def test_includes_builtin_engines(self):
        """list_available_engines includes built-in engines."""
        engines = list_available_engines()

        assert "llama-cpp" in engines
        assert "transformers" in engines
        assert "vllm" in engines

    @patch("hfl.plugins.discover_engines")
    def test_lists_all_discovered_engines(self, mock_discover):
        """list_available_engines lists all discovered engines."""
        mock_discover.return_value = {
            "engine-a": MagicMock(),
            "engine-b": MagicMock(),
            "engine-c": MagicMock(),
        }

        engines = list_available_engines()

        assert set(engines) == {"engine-a", "engine-b", "engine-c"}


class TestListAvailableTTSEngines:
    """Tests for list_available_tts_engines function."""

    def test_returns_list(self):
        """list_available_tts_engines returns a list."""
        engines = list_available_tts_engines()
        assert isinstance(engines, list)

    def test_includes_builtin_tts_engines(self):
        """list_available_tts_engines includes built-in TTS engines."""
        engines = list_available_tts_engines()

        assert "bark" in engines
        assert "coqui" in engines

    @patch("hfl.plugins.discover_tts_engines")
    def test_lists_all_discovered_tts_engines(self, mock_discover):
        """list_available_tts_engines lists all discovered TTS engines."""
        mock_discover.return_value = {
            "tts-a": MagicMock(),
            "tts-b": MagicMock(),
        }

        engines = list_available_tts_engines()

        assert set(engines) == {"tts-a", "tts-b"}


class TestEngineDiscoveryCache:
    """Tests for engine discovery caching."""

    def test_cache_ttl_is_5_minutes(self):
        """Cache TTL should be 5 minutes (300 seconds)."""
        assert _CACHE_TTL == 300.0

    def test_returns_cached_result(self):
        """discover_engines returns cached result on second call."""
        engines1 = discover_engines()
        engines2 = discover_engines()

        # Should return same object (cached)
        assert engines1 is engines2

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True bypasses the cache."""
        engines1 = discover_engines()
        engines2 = discover_engines(force_refresh=True)

        # Should return new dict (not cached)
        assert engines1 is not engines2
        # But should have same content
        assert engines1.keys() == engines2.keys()

    def test_cache_expires_after_ttl(self):
        """Cache expires after TTL."""
        engines1 = discover_engines()

        # Simulate time passing beyond TTL
        original_time = plugins_module._engine_cache_timestamp
        plugins_module._engine_cache_timestamp = original_time - _CACHE_TTL - 1

        engines2 = discover_engines()

        # Should return new dict (cache expired)
        assert engines1 is not engines2

    def test_clear_cache_resets_state(self):
        """clear_engine_cache resets all cache state."""
        # Populate caches
        discover_engines()
        discover_tts_engines()

        assert plugins_module._engine_cache is not None
        assert plugins_module._tts_cache is not None

        clear_engine_cache()

        assert plugins_module._engine_cache is None
        assert plugins_module._engine_cache_timestamp == 0
        assert plugins_module._tts_cache is None
        assert plugins_module._tts_cache_timestamp == 0


class TestTTSEngineDiscoveryCache:
    """Tests for TTS engine discovery caching."""

    def test_returns_cached_result(self):
        """discover_tts_engines returns cached result on second call."""
        engines1 = discover_tts_engines()
        engines2 = discover_tts_engines()

        # Should return same object (cached)
        assert engines1 is engines2

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True bypasses the cache."""
        engines1 = discover_tts_engines()
        engines2 = discover_tts_engines(force_refresh=True)

        # Should return new dict (not cached)
        assert engines1 is not engines2
        # But should have same content
        assert engines1.keys() == engines2.keys()

    def test_cache_expires_after_ttl(self):
        """TTS cache expires after TTL."""
        engines1 = discover_tts_engines()

        # Simulate time passing beyond TTL
        original_time = plugins_module._tts_cache_timestamp
        plugins_module._tts_cache_timestamp = original_time - _CACHE_TTL - 1

        engines2 = discover_tts_engines()

        # Should return new dict (cache expired)
        assert engines1 is not engines2

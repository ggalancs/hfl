# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for plugin system."""

from hfl.plugins import (
    discover_engines,
    discover_tts_engines,
    list_available_engines,
    list_available_tts_engines,
)


class TestDiscoverEngines:
    """Tests for discover_engines function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        engines = discover_engines()
        assert isinstance(engines, dict)

    def test_includes_builtin_engines(self):
        """Should include built-in engines."""
        engines = discover_engines()

        assert "llama-cpp" in engines
        assert "transformers" in engines
        assert "vllm" in engines

    def test_lazy_loaders_are_callable(self):
        """Built-in engines should be callable lazy loaders."""
        engines = discover_engines()

        for name, loader in engines.items():
            assert callable(loader), f"{name} should be callable"


class TestDiscoverTTSEngines:
    """Tests for discover_tts_engines function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        engines = discover_tts_engines()
        assert isinstance(engines, dict)

    def test_includes_builtin_tts(self):
        """Should include built-in TTS engines."""
        engines = discover_tts_engines()

        assert "bark" in engines
        assert "coqui" in engines


class TestListAvailableEngines:
    """Tests for list_available_engines function."""

    def test_returns_list(self):
        """Should return a list of engine names."""
        names = list_available_engines()

        assert isinstance(names, list)
        assert "llama-cpp" in names
        assert "transformers" in names


class TestListAvailableTTSEngines:
    """Tests for list_available_tts_engines function."""

    def test_returns_list(self):
        """Should return a list of TTS engine names."""
        names = list_available_tts_engines()

        assert isinstance(names, list)
        assert "bark" in names
        assert "coqui" in names

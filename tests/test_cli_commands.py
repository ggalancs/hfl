# SPDX-License-Identifier: HRUL-1.0
"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hfl.cli.main import app


runner = CliRunner()


class TestServeCommand:
    """Tests for serve command."""

    def test_serve_default_options(self):
        """Test serve command with default options."""
        with patch("hfl.api.server.start_server") as mock_start:
            result = runner.invoke(app, ["serve"])

            assert result.exit_code == 0
            mock_start.assert_called_once()
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 11434
            assert call_kwargs["api_key"] is None

    def test_serve_with_api_key(self):
        """Test serve command with API key."""
        with patch("hfl.api.server.start_server") as mock_start:
            result = runner.invoke(app, ["serve", "--api-key", "secret-key"])

            assert result.exit_code == 0
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["api_key"] == "secret-key"

    def test_serve_network_exposure_warning_declined(self):
        """Test serve command network warning when user declines."""
        with patch("hfl.api.server.start_server") as mock_start:
            result = runner.invoke(app, ["serve", "--host", "0.0.0.0"], input="n\n")

            assert result.exit_code == 0
            mock_start.assert_not_called()


class TestListCommand:
    """Tests for list command."""

    def test_list_no_models(self):
        """Test list command with no models."""
        with patch("hfl.models.registry.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.list_all.return_value = []
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["list"])

            assert result.exit_code == 0

    def test_list_with_models(self):
        """Test list command with models."""
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.alias = "test"
        mock_model.format = "GGUF"
        mock_model.quantization = "Q4_K_M"
        mock_model.license = "apache-2.0"
        mock_model.display_size = "5.0 GB"

        with patch("hfl.models.registry.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.list_all.return_value = [mock_model]
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["list"])

            assert result.exit_code == 0
            assert "test-model" in result.output


class TestRemoveCommand:
    """Tests for remove/rm command."""

    def test_remove_model_not_found(self):
        """Test remove command when model not found."""
        with patch("hfl.models.registry.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["rm", "nonexistent-model"])

            assert result.exit_code == 1

    def test_remove_model_cancelled(self):
        """Test remove command when user cancels."""
        mock_manifest = MagicMock()
        mock_manifest.name = "test-model"
        mock_manifest.local_path = "/path/to/model"
        mock_manifest.display_size = "5.0 GB"

        with patch("hfl.models.registry.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = mock_manifest
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["rm", "test-model"], input="n\n")

            assert result.exit_code == 0


class TestSearchCommand:
    """Tests for search command."""

    def test_search_query_too_short(self):
        """Test search with query too short."""
        result = runner.invoke(app, ["search", "ab"])

        assert result.exit_code == 1


class TestInspectCommand:
    """Tests for inspect command."""

    def test_inspect_model_not_found(self):
        """Test inspect when model not found."""
        with patch("hfl.models.registry.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["inspect", "nonexistent-model"])

            assert result.exit_code == 1


class TestAliasCommand:
    """Tests for alias command."""

    def test_alias_model_not_found(self):
        """Test alias when model not found."""
        with patch("hfl.models.registry.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["alias", "nonexistent-model", "my-alias"])

            assert result.exit_code == 1


class TestAuthCommands:
    """Tests for auth commands."""

    def test_logout(self):
        """Test logout command."""
        with patch("huggingface_hub.logout") as mock_logout:
            result = runner.invoke(app, ["logout"])

            assert result.exit_code == 0
            mock_logout.assert_called_once()

    def test_version(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0

    def test_help(self):
        """Test help command."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0


class TestRunCommand:
    """Tests for run command."""

    def test_run_model_not_found(self):
        """Test run with non-existent model."""
        with patch("hfl.models.registry.ModelRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.get.return_value = None
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["run", "nonexistent-model"])

            assert result.exit_code == 1

# SPDX-License-Identifier: HRUL-1.0
"""Extended CLI tests for improved coverage."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hfl.cli.main import _format_size, app

runner = CliRunner()


class TestFormatSize:
    """Tests for _format_size helper function."""

    def test_format_size_gigabytes(self):
        """Test formatting gigabyte sizes."""
        result = _format_size(2 * 1024 * 1024 * 1024)  # 2 GB
        assert "GB" in result

    def test_format_size_megabytes(self):
        """Test formatting megabyte sizes."""
        result = _format_size(500 * 1024 * 1024)  # 500 MB
        assert "MB" in result


class TestLogoutCommand:
    """Tests for the logout command."""

    def test_logout_success(self):
        """Test successful logout."""
        with patch("huggingface_hub.logout") as mock_logout:
            result = runner.invoke(app, ["logout"])
            mock_logout.assert_called_once()
            assert result.exit_code == 0


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_output(self):
        """Test version command output."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output or "hfl" in result.output.lower()


class TestSearchMinChars:
    """Test search minimum character requirement."""

    def test_search_with_two_chars_fails(self):
        """Test that search with less than 3 chars fails."""
        result = runner.invoke(app, ["search", "ab"])
        assert result.exit_code == 1


class TestHelpCommands:
    """Test help output for commands."""

    def test_alias_help(self):
        """Test alias command help."""
        result = runner.invoke(app, ["alias", "--help"])
        assert result.exit_code == 0

    def test_login_help(self):
        """Test login command help."""
        result = runner.invoke(app, ["login", "--help"])
        assert result.exit_code == 0

    def test_logout_help(self):
        """Test logout command help."""
        result = runner.invoke(app, ["logout", "--help"])
        assert result.exit_code == 0

    def test_inspect_help(self):
        """Test inspect command help."""
        result = runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0

    def test_rm_help(self):
        """Test rm command help."""
        result = runner.invoke(app, ["rm", "--help"])
        assert result.exit_code == 0

    def test_list_help(self):
        """Test list command help."""
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_version_help(self):
        """Test version command help."""
        result = runner.invoke(app, ["version", "--help"])
        assert result.exit_code == 0

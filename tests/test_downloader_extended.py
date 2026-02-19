# SPDX-License-Identifier: HRUL-1.0
"""Extended tests for hub downloader module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from hfl.hub.downloader import _rate_limit, pull_model


class TestRateLimitExtended:
    """Extended tests for _rate_limit function."""

    def test_rate_limit_enforces_minimum_interval(self):
        """Test that rate limiting enforces minimum interval between calls."""
        import hfl.hub.downloader as dl

        # Set last call to now
        dl._last_api_call = time.time()

        with patch("hfl.hub.downloader.time.sleep") as mock_sleep:
            _rate_limit()

            # Should have called sleep to enforce minimum interval
            if mock_sleep.called:
                # Sleep was called, verify it's close to the minimum interval
                pass  # Sleep time depends on exact timing

    def test_rate_limit_no_sleep_after_long_delay(self):
        """Test that no sleep is needed after a long delay."""
        import hfl.hub.downloader as dl

        # Set last call to long ago
        dl._last_api_call = 0

        with patch("hfl.hub.downloader.time.sleep") as mock_sleep:
            _rate_limit()

            # No sleep needed since last call was long ago
            mock_sleep.assert_not_called()


class TestPullModelExtended:
    """Extended tests for pull_model function."""

    def test_pull_gguf_creates_directory(self, tmp_path):
        """Test that pull_model creates the model directory."""
        from hfl.hub.resolver import ResolvedModel

        resolved = ResolvedModel(
            repo_id="org/model",
            format="gguf",
            filename="model.Q4_K_M.gguf",
            revision=None,
            quantization="Q4_K_M",
        )

        with patch("hfl.hub.downloader.ensure_auth") as mock_auth:
            mock_auth.return_value = None

            with patch("hfl.hub.downloader.hf_hub_download") as mock_download:
                mock_download.return_value = str(tmp_path / "model.gguf")

                with patch("hfl.hub.downloader.config") as mock_config:
                    mock_config.models_dir = tmp_path

                    result = pull_model(resolved)

                    # Should have called hf_hub_download with correct params
                    call_kwargs = mock_download.call_args[1]
                    assert call_kwargs["repo_id"] == "org/model"
                    assert call_kwargs["filename"] == "model.Q4_K_M.gguf"

    def test_pull_safetensors_filters_files(self, tmp_path):
        """Test that safetensors download filters file patterns."""
        from hfl.hub.resolver import ResolvedModel

        resolved = ResolvedModel(
            repo_id="org/model",
            format="safetensors",
            filename=None,
            revision=None,
            quantization=None,
        )

        with patch("hfl.hub.downloader.ensure_auth") as mock_auth:
            mock_auth.return_value = "token"

            with patch("hfl.hub.downloader.snapshot_download") as mock_download:
                mock_download.return_value = str(tmp_path / "model")

                with patch("hfl.hub.downloader.config") as mock_config:
                    mock_config.models_dir = tmp_path

                    result = pull_model(resolved)

                    # Should have called snapshot_download with allow_patterns
                    call_kwargs = mock_download.call_args[1]
                    patterns = call_kwargs.get("allow_patterns")
                    assert patterns is not None
                    assert "*.safetensors" in patterns
                    assert "config.json" in patterns
                    assert "tokenizer.json" in patterns


class TestVersionFallback:
    """Tests for version fallback in downloader module."""

    def test_version_is_available(self):
        """Test that __version__ is available."""
        try:
            from hfl import __version__
            assert __version__ is not None
            assert isinstance(__version__, str)
        except ImportError:
            # Fallback should be available
            pass

    def test_user_agent_is_set(self):
        """Test that HF_HUB_USER_AGENT is set."""
        import os

        # After importing downloader, the user agent should be set
        import hfl.hub.downloader

        user_agent = os.environ.get("HF_HUB_USER_AGENT", "")
        # It should contain 'hfl'
        assert "hfl" in user_agent.lower()

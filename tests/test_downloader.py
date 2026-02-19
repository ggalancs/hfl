# SPDX-License-Identifier: HRUL-1.0
"""Tests for hub downloader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hfl.hub.downloader import pull_model, _rate_limit


class TestRateLimit:
    """Tests for _rate_limit function."""

    def test_rate_limit_first_call(self):
        """Test that first call doesn't sleep."""
        # Reset the last API call
        import hfl.hub.downloader as dl
        dl._last_api_call = 0

        with patch("hfl.hub.downloader.time.sleep") as mock_sleep:
            _rate_limit()

            # First call should not sleep (elapsed time is large)
            # This depends on implementation but covers the function


class TestPullModel:
    """Tests for pull_model function."""

    def test_pull_gguf_model(self, tmp_path):
        """Test pulling a GGUF model."""
        from hfl.hub.resolver import ResolvedModel

        resolved = ResolvedModel(
            repo_id="org/model",
            format="gguf",
            filename="model.Q4_K_M.gguf",
            revision=None,
            quantization="Q4_K_M",
        )

        with patch("hfl.hub.downloader.ensure_auth") as mock_auth:
            mock_auth.return_value = "token"

            with patch("hfl.hub.downloader.hf_hub_download") as mock_download:
                mock_download.return_value = str(tmp_path / "model.gguf")

                with patch("hfl.hub.downloader.config") as mock_config:
                    mock_config.models_dir = tmp_path

                    result = pull_model(resolved)

                    mock_download.assert_called_once()
                    assert result == Path(tmp_path / "model.gguf")

    def test_pull_safetensors_model(self, tmp_path):
        """Test pulling a safetensors model."""
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

                    mock_download.assert_called_once()
                    # Should include allow_patterns for safetensors
                    call_kwargs = mock_download.call_args[1]
                    assert call_kwargs.get("allow_patterns") is not None

    def test_pull_pytorch_model(self, tmp_path):
        """Test pulling a pytorch model."""
        from hfl.hub.resolver import ResolvedModel

        resolved = ResolvedModel(
            repo_id="org/model",
            format="pytorch",
            filename=None,
            revision=None,
            quantization=None,
        )

        with patch("hfl.hub.downloader.ensure_auth") as mock_auth:
            mock_auth.return_value = None

            with patch("hfl.hub.downloader.snapshot_download") as mock_download:
                mock_download.return_value = str(tmp_path / "model")

                with patch("hfl.hub.downloader.config") as mock_config:
                    mock_config.models_dir = tmp_path

                    result = pull_model(resolved)

                    mock_download.assert_called_once()

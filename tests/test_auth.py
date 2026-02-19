# SPDX-License-Identifier: HRUL-1.0
"""Tests for HuggingFace Hub authentication."""

from unittest.mock import MagicMock, patch

import pytest


class TestGetHfToken:
    """Tests for get_hf_token function."""

    def test_token_from_env_variable(self):
        """Test that token is retrieved from HF_TOKEN env var."""
        with patch("hfl.hub.auth.config") as mock_config:
            mock_config.hf_token = "env_token_123"
            from hfl.hub.auth import get_hf_token

            assert get_hf_token() == "env_token_123"

    def test_token_from_huggingface_hub(self):
        """Test fallback to huggingface_hub token."""
        with patch("hfl.hub.auth.config") as mock_config:
            mock_config.hf_token = None
            with patch("hfl.hub.auth.get_token") as mock_get_token:
                mock_get_token.return_value = "hf_token_456"
                from hfl.hub.auth import get_hf_token

                assert get_hf_token() == "hf_token_456"

    def test_no_token_available(self):
        """Test returns None when no token is available."""
        with patch("hfl.hub.auth.config") as mock_config:
            mock_config.hf_token = None
            with patch("hfl.hub.auth.get_token") as mock_get_token:
                mock_get_token.return_value = None
                from hfl.hub.auth import get_hf_token

                assert get_hf_token() is None

    def test_huggingface_hub_exception(self):
        """Test returns None when huggingface_hub raises an exception."""
        with patch("hfl.hub.auth.config") as mock_config:
            mock_config.hf_token = None
            with patch("hfl.hub.auth.get_token") as mock_get_token:
                mock_get_token.side_effect = Exception("Token error")
                from hfl.hub.auth import get_hf_token

                assert get_hf_token() is None


class TestEnsureAuth:
    """Tests for ensure_auth function."""

    def test_public_model_no_token_needed(self):
        """Test that public models work without a token."""
        with patch("hfl.hub.auth.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.model_info.return_value = MagicMock()
            mock_api_class.return_value = mock_api

            with patch("hfl.hub.auth.get_hf_token") as mock_get_token:
                mock_get_token.return_value = None

                from hfl.hub.auth import ensure_auth

                result = ensure_auth("public/model")

                assert result is None

    def test_model_with_existing_token(self):
        """Test that existing token is used successfully."""
        with patch("hfl.hub.auth.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.model_info.return_value = MagicMock()
            mock_api_class.return_value = mock_api

            with patch("hfl.hub.auth.get_hf_token") as mock_get_token:
                mock_get_token.return_value = "existing_token"

                from hfl.hub.auth import ensure_auth

                result = ensure_auth("meta-llama/Llama-3.1-8B")

                assert result == "existing_token"

    def test_gated_model_requires_token_interactive(self):
        """Test that gated models prompt for token when none available."""
        with patch("hfl.hub.auth.HfApi") as mock_api_class:
            mock_api = MagicMock()
            # First call fails (no token), second call succeeds (with user-provided token)
            mock_api.model_info.side_effect = [
                Exception("Gated model requires auth"),
                MagicMock(),
            ]
            mock_api_class.return_value = mock_api

            with patch("hfl.hub.auth.get_hf_token") as mock_get_token:
                mock_get_token.return_value = None

                with patch("rich.console.Console") as mock_console_class:
                    mock_console = MagicMock()
                    mock_console_class.return_value = mock_console

                    with patch("rich.prompt.Prompt.ask") as mock_prompt:
                        mock_prompt.return_value = "user_provided_token"

                        from hfl.hub.auth import ensure_auth

                        result = ensure_auth("meta-llama/Llama-3.1-8B")

                        assert result == "user_provided_token"
                        mock_prompt.assert_called_once()

    def test_gated_model_invalid_token_raises(self):
        """Test that invalid token raises RuntimeError."""
        with patch("hfl.hub.auth.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.model_info.side_effect = Exception("Invalid token")
            mock_api_class.return_value = mock_api

            with patch("hfl.hub.auth.get_hf_token") as mock_get_token:
                mock_get_token.return_value = None

                with patch("rich.console.Console"):
                    with patch("rich.prompt.Prompt.ask") as mock_prompt:
                        mock_prompt.return_value = "invalid_token"

                        from hfl.hub.auth import ensure_auth

                        with pytest.raises(RuntimeError) as exc_info:
                            ensure_auth("meta-llama/Llama-3.1-8B")

                        assert "Cannot access" in str(exc_info.value)
                        assert "meta-llama/Llama-3.1-8B" in str(exc_info.value)

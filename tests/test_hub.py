# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the hub module (auth, resolver, downloader)."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


class TestAuth:
    """Tests for hub/auth.py."""

    def test_ensure_auth_public_model(self, mock_hf_api):
        """Verifies that public models work without token."""
        mock_hf_api.model_info.return_value = MagicMock()

        from hfl.hub.auth import ensure_auth

        with patch("hfl.hub.auth.HfApi", return_value=mock_hf_api):
            with patch("hfl.hub.auth.get_hf_token", return_value=None):
                result = ensure_auth("public-org/public-model")

        assert result is None

    def test_ensure_auth_gated_model_with_token(self, mock_hf_api, temp_config, monkeypatch):
        """Verifies authentication with token for gated models."""
        # First call fails (no token), second succeeds
        mock_hf_api.model_info.side_effect = [
            Exception("Gated model"),
            MagicMock(),
        ]
        monkeypatch.setenv("HF_TOKEN", "valid-token")

        from hfl.hub.auth import ensure_auth

        with patch("hfl.hub.auth.HfApi", return_value=mock_hf_api):
            with patch("hfl.hub.auth.config") as mock_config:
                mock_config.hf_token = "valid-token"
                result = ensure_auth("meta-llama/Llama-3")

        assert result == "valid-token"

    def test_ensure_auth_raises_on_invalid_token(self, mock_hf_api, temp_config):
        """Verifies that error is raised with invalid token."""
        mock_hf_api.model_info.side_effect = Exception("Unauthorized")

        from hfl.hub.auth import ensure_auth

        with patch("hfl.hub.auth.HfApi", return_value=mock_hf_api):
            with patch("hfl.hub.auth.config") as mock_config:
                mock_config.hf_token = "invalid-token"
                with pytest.raises(RuntimeError, match="Cannot access"):
                    ensure_auth("private-org/private-model")


class TestResolver:
    """Tests for hub/resolver.py."""

    def test_resolved_model_dataclass(self):
        """Verifies ResolvedModel creation."""
        from hfl.hub.resolver import ResolvedModel

        resolved = ResolvedModel(
            repo_id="org/model",
            revision="main",
            filename="model.gguf",
            format="gguf",
            quantization="Q4_K_M",
        )

        assert resolved.repo_id == "org/model"
        assert resolved.revision == "main"
        assert resolved.filename == "model.gguf"
        assert resolved.format == "gguf"
        assert resolved.quantization == "Q4_K_M"

    def test_resolved_model_defaults(self):
        """Verifies ResolvedModel default values."""
        from hfl.hub.resolver import ResolvedModel

        resolved = ResolvedModel(repo_id="org/model")

        assert resolved.revision == "main"
        assert resolved.filename is None
        assert resolved.format == "auto"
        assert resolved.quantization is None

    def test_resolve_direct_repo_id_gguf(self, mock_hf_api, sample_gguf_model_info):
        """Verifies resolution of direct repo_id with GGUF."""
        mock_hf_api.model_info.return_value = sample_gguf_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            result = resolve("test-org/test-model-gguf")

        assert result.repo_id == "test-org/test-model-gguf"
        assert result.format == "gguf"
        assert "Q4_K_M" in result.filename

    def test_resolve_direct_repo_id_safetensors(self, mock_hf_api, sample_model_info):
        """Verifies resolution of direct repo_id with safetensors."""
        mock_hf_api.model_info.return_value = sample_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            result = resolve("test-org/test-model")

        assert result.repo_id == "test-org/test-model"
        assert result.format == "safetensors"

    def test_resolve_search_by_name(self, mock_hf_api, sample_model_info):
        """Verifies resolution by name search."""
        mock_model = MagicMock()
        mock_model.id = "found-org/found-model"
        mock_hf_api.list_models.return_value = [mock_model]

        # Adjust sample_model_info to match the found model
        sample_model_info.id = "found-org/found-model"
        mock_hf_api.model_info.return_value = sample_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            result = resolve("llama")

        mock_hf_api.list_models.assert_called_once()
        assert result.repo_id == "found-org/found-model"

    def test_resolve_not_found(self, mock_hf_api):
        """Verifies error when model is not found."""
        mock_hf_api.list_models.return_value = []

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            with pytest.raises(ValueError, match="Model not found"):
                resolve("nonexistent")

    def test_select_gguf_with_specific_quant(self):
        """Verifies GGUF selection with specific quantization."""
        from hfl.hub.resolver import _select_gguf

        files = ["model-Q4_K_M.gguf", "model-Q5_K_M.gguf", "model-Q8_0.gguf"]

        result = _select_gguf(files, "Q5_K_M")
        assert result == "model-Q5_K_M.gguf"

        result = _select_gguf(files, "Q8_0")
        assert result == "model-Q8_0.gguf"

    def test_select_gguf_default_priority(self):
        """Verifies default GGUF selection priority."""
        from hfl.hub.resolver import _select_gguf

        files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q3_K_M.gguf"]

        result = _select_gguf(files, None)
        assert result == "model-Q4_K_M.gguf"  # Q4_K_M has priority

    def test_select_gguf_fallback_to_first(self):
        """Verifies fallback to first file if no match."""
        from hfl.hub.resolver import _select_gguf

        files = ["model-custom.gguf", "model-other.gguf"]

        result = _select_gguf(files, None)
        assert result == "model-custom.gguf"

    def test_detect_quant(self):
        """Verifies quantization level detection."""
        from hfl.hub.resolver import _detect_quant

        assert _detect_quant("model-Q4_K_M.gguf") == "Q4_K_M"
        assert _detect_quant("model-Q5_K_S.gguf") == "Q5_K_S"
        assert _detect_quant("model-F16.gguf") == "F16"
        assert _detect_quant("model-Q8_0.gguf") == "Q8_0"
        assert _detect_quant("model.gguf") is None

    def test_is_quantization(self):
        """Verifies quantization string detection."""
        from hfl.hub.resolver import _is_quantization

        # Valid quantizations
        assert _is_quantization("Q4_K_M") is True
        assert _is_quantization("q4_k_m") is True  # case insensitive
        assert _is_quantization("Q5_K_S") is True
        assert _is_quantization("Q8_0") is True
        assert _is_quantization("F16") is True
        assert _is_quantization("F32") is True
        assert _is_quantization("IQ4_NL") is True
        assert _is_quantization("IQ2_XS") is True

        # Not quantizations
        assert _is_quantization("main") is False
        assert _is_quantization("v1.0") is False
        assert _is_quantization("latest") is False
        assert _is_quantization("") is False
        assert _is_quantization("model-name") is False

    def test_resolve_with_colon_quantization(self, mock_hf_api, sample_gguf_model_info):
        """Verifies resolution with repo:quantization format (Ollama style)."""
        mock_hf_api.model_info.return_value = sample_gguf_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            result = resolve("test-org/test-model-gguf:Q5_K_M")

        assert result.repo_id == "test-org/test-model-gguf"
        assert result.quantization == "Q5_K_M"
        assert "Q5_K_M" in result.filename

    def test_resolve_colon_not_quantization(self, mock_hf_api, sample_gguf_model_info):
        """Verifies that : without valid quantization is not parsed incorrectly."""
        # Simulate a repo that has : in the name (rare but possible in searches)
        mock_model = MagicMock()
        mock_model.id = "org/model"
        mock_hf_api.list_models.return_value = [mock_model]
        mock_hf_api.model_info.return_value = sample_gguf_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            # "model:latest" - latest is not a quantization, should search "model:latest"
            result = resolve("model:latest")

        # Should have searched for the model
        mock_hf_api.list_models.assert_called()

    def test_resolve_quantization_priority(self, mock_hf_api, sample_gguf_model_info):
        """Verifies that quantization in spec has priority over parameter."""
        mock_hf_api.model_info.return_value = sample_gguf_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            # Q5_K_M in spec should have priority over Q4_K_M in parameter
            result = resolve("test-org/test-model-gguf:Q5_K_M", quantization="Q4_K_M")

        assert result.quantization == "Q5_K_M"


class TestLicenseChecker:
    """Tests for hub/license_checker.py."""

    def test_check_license_with_license_name(self, mock_hf_api):
        """Verifies that it uses license_name when license is 'other'."""
        from hfl.hub.license_checker import check_model_license, LicenseRisk

        # Simulate model with license: other but license_name: qwen2
        mock_info = MagicMock()
        mock_card_data = MagicMock()
        mock_card_data.license = "other"
        mock_card_data.license_name = "qwen2"
        mock_card_data.license_link = "https://example.com/LICENSE"
        mock_info.card_data = mock_card_data
        mock_info.tags = []
        mock_info.gated = False
        mock_hf_api.model_info.return_value = mock_info

        with patch("hfl.hub.license_checker.HfApi", return_value=mock_hf_api):
            result = check_model_license("test/model")

        assert result.license_id == "qwen2"
        assert result.risk == LicenseRisk.CONDITIONAL
        assert result.url == "https://example.com/LICENSE"

    def test_check_license_permissive(self, mock_hf_api):
        """Verifies permissive license detection."""
        from hfl.hub.license_checker import check_model_license, LicenseRisk

        mock_info = MagicMock()
        mock_card_data = MagicMock(spec=["license"])
        mock_card_data.license = "apache-2.0"
        mock_info.card_data = mock_card_data
        mock_info.tags = []
        mock_info.gated = False
        mock_hf_api.model_info.return_value = mock_info

        with patch("hfl.hub.license_checker.HfApi", return_value=mock_hf_api):
            result = check_model_license("test/model")

        assert result.license_id == "apache-2.0"
        assert result.risk == LicenseRisk.PERMISSIVE

    def test_check_license_non_commercial(self, mock_hf_api):
        """Verifies non-commercial license detection."""
        from hfl.hub.license_checker import check_model_license, LicenseRisk

        mock_info = MagicMock()
        mock_card_data = MagicMock(spec=["license"])
        mock_card_data.license = "cc-by-nc-4.0"
        mock_info.card_data = mock_card_data
        mock_info.tags = []
        mock_info.gated = False
        mock_hf_api.model_info.return_value = mock_info

        with patch("hfl.hub.license_checker.HfApi", return_value=mock_hf_api):
            result = check_model_license("test/model")

        assert result.risk == LicenseRisk.NON_COMMERCIAL
        assert "non-commercial-only" in result.restrictions

    def test_check_license_unknown_fallback(self, mock_hf_api):
        """Verifies fallback to unknown for unrecognized licenses."""
        from hfl.hub.license_checker import check_model_license, LicenseRisk

        mock_info = MagicMock()
        mock_card_data = MagicMock(spec=["license"])
        mock_card_data.license = "custom-proprietary"
        mock_info.card_data = mock_card_data
        mock_info.tags = []
        mock_info.gated = False
        mock_hf_api.model_info.return_value = mock_info

        with patch("hfl.hub.license_checker.HfApi", return_value=mock_hf_api):
            result = check_model_license("test/model")

        assert result.risk == LicenseRisk.UNKNOWN


class TestDownloader:
    """Tests for hub/downloader.py."""

    def test_pull_model_gguf(self, mock_hf_api, temp_config):
        """Verifies GGUF model download."""
        from hfl.hub.resolver import ResolvedModel
        from hfl.hub.downloader import pull_model

        resolved = ResolvedModel(
            repo_id="test/model",
            filename="model.gguf",
            format="gguf",
        )

        with patch("hfl.hub.downloader.ensure_auth", return_value=None):
            with patch("hfl.hub.downloader.hf_hub_download") as mock_download:
                mock_download.return_value = str(temp_config.models_dir / "model.gguf")

                result = pull_model(resolved)

                mock_download.assert_called_once()
                assert isinstance(result, Path)

    def test_pull_model_safetensors(self, mock_hf_api, temp_config):
        """Verifies safetensors model download (snapshot)."""
        from hfl.hub.resolver import ResolvedModel
        from hfl.hub.downloader import pull_model

        resolved = ResolvedModel(
            repo_id="test/model",
            format="safetensors",
        )

        with patch("hfl.hub.downloader.ensure_auth", return_value=None):
            with patch("hfl.hub.downloader.snapshot_download") as mock_download:
                mock_download.return_value = str(temp_config.models_dir / "test--model")

                result = pull_model(resolved)

                mock_download.assert_called_once()
                # Verify allow_patterns for safetensors
                call_kwargs = mock_download.call_args[1]
                assert "*.safetensors" in call_kwargs["allow_patterns"]
                assert "config.json" in call_kwargs["allow_patterns"]

    def test_pull_model_with_auth(self, mock_hf_api, temp_config):
        """Verifies download with authentication."""
        from hfl.hub.resolver import ResolvedModel
        from hfl.hub.downloader import pull_model

        resolved = ResolvedModel(
            repo_id="gated/model",
            filename="model.gguf",
            format="gguf",
        )

        with patch("hfl.hub.downloader.ensure_auth", return_value="my-token"):
            with patch("hfl.hub.downloader.hf_hub_download") as mock_download:
                mock_download.return_value = str(temp_config.models_dir / "model.gguf")

                pull_model(resolved)

                call_kwargs = mock_download.call_args[1]
                assert call_kwargs["token"] == "my-token"

    def test_pull_model_creates_directory(self, mock_hf_api, temp_config):
        """Verifies that model directory is created."""
        from hfl.hub.resolver import ResolvedModel
        from hfl.hub.downloader import pull_model

        resolved = ResolvedModel(
            repo_id="new-org/new-model",
            filename="model.gguf",
            format="gguf",
        )

        expected_dir = temp_config.models_dir / "new-org--new-model"

        with patch("hfl.hub.downloader.ensure_auth", return_value=None):
            with patch("hfl.hub.downloader.hf_hub_download") as mock_download:
                mock_download.return_value = str(expected_dir / "model.gguf")

                pull_model(resolved)

                assert expected_dir.exists()

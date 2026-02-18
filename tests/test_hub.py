# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests para el módulo hub (auth, resolver, downloader)."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


class TestAuth:
    """Tests para hub/auth.py."""

    def test_ensure_auth_public_model(self, mock_hf_api):
        """Verifica que modelos públicos no requieren token."""
        mock_hf_api.model_info.return_value = MagicMock()

        from hfl.hub.auth import ensure_auth

        with patch("hfl.hub.auth.HfApi", return_value=mock_hf_api):
            result = ensure_auth("public-org/public-model")

        assert result is None

    def test_ensure_auth_gated_model_with_token(self, mock_hf_api, temp_config, monkeypatch):
        """Verifica autenticación con token para modelos gated."""
        # Primera llamada falla (sin token), segunda tiene éxito
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
        """Verifica que se lanza error con token inválido."""
        mock_hf_api.model_info.side_effect = Exception("Unauthorized")

        from hfl.hub.auth import ensure_auth

        with patch("hfl.hub.auth.HfApi", return_value=mock_hf_api):
            with patch("hfl.hub.auth.config") as mock_config:
                mock_config.hf_token = "invalid-token"
                with pytest.raises(RuntimeError, match="No se puede acceder"):
                    ensure_auth("private-org/private-model")


class TestResolver:
    """Tests para hub/resolver.py."""

    def test_resolved_model_dataclass(self):
        """Verifica la creación de ResolvedModel."""
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
        """Verifica valores por defecto de ResolvedModel."""
        from hfl.hub.resolver import ResolvedModel

        resolved = ResolvedModel(repo_id="org/model")

        assert resolved.revision == "main"
        assert resolved.filename is None
        assert resolved.format == "auto"
        assert resolved.quantization is None

    def test_resolve_direct_repo_id_gguf(self, mock_hf_api, sample_gguf_model_info):
        """Verifica resolución de repo_id directo con GGUF."""
        mock_hf_api.model_info.return_value = sample_gguf_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            result = resolve("test-org/test-model-gguf")

        assert result.repo_id == "test-org/test-model-gguf"
        assert result.format == "gguf"
        assert "Q4_K_M" in result.filename

    def test_resolve_direct_repo_id_safetensors(self, mock_hf_api, sample_model_info):
        """Verifica resolución de repo_id directo con safetensors."""
        mock_hf_api.model_info.return_value = sample_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            result = resolve("test-org/test-model")

        assert result.repo_id == "test-org/test-model"
        assert result.format == "safetensors"

    def test_resolve_search_by_name(self, mock_hf_api, sample_model_info):
        """Verifica resolución por búsqueda de nombre."""
        mock_model = MagicMock()
        mock_model.id = "found-org/found-model"
        mock_hf_api.list_models.return_value = [mock_model]

        # Ajustar sample_model_info para que coincida con el modelo encontrado
        sample_model_info.id = "found-org/found-model"
        mock_hf_api.model_info.return_value = sample_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            result = resolve("llama")

        mock_hf_api.list_models.assert_called_once()
        assert result.repo_id == "found-org/found-model"

    def test_resolve_not_found(self, mock_hf_api):
        """Verifica error cuando no se encuentra el modelo."""
        mock_hf_api.list_models.return_value = []

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            with pytest.raises(ValueError, match="No se encontró modelo"):
                resolve("nonexistent")

    def test_select_gguf_with_specific_quant(self):
        """Verifica selección de GGUF con cuantización específica."""
        from hfl.hub.resolver import _select_gguf

        files = ["model-Q4_K_M.gguf", "model-Q5_K_M.gguf", "model-Q8_0.gguf"]

        result = _select_gguf(files, "Q5_K_M")
        assert result == "model-Q5_K_M.gguf"

        result = _select_gguf(files, "Q8_0")
        assert result == "model-Q8_0.gguf"

    def test_select_gguf_default_priority(self):
        """Verifica prioridad por defecto de selección GGUF."""
        from hfl.hub.resolver import _select_gguf

        files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q3_K_M.gguf"]

        result = _select_gguf(files, None)
        assert result == "model-Q4_K_M.gguf"  # Q4_K_M tiene prioridad

    def test_select_gguf_fallback_to_first(self):
        """Verifica fallback al primer archivo si no hay match."""
        from hfl.hub.resolver import _select_gguf

        files = ["model-custom.gguf", "model-other.gguf"]

        result = _select_gguf(files, None)
        assert result == "model-custom.gguf"

    def test_detect_quant(self):
        """Verifica detección de nivel de cuantización."""
        from hfl.hub.resolver import _detect_quant

        assert _detect_quant("model-Q4_K_M.gguf") == "Q4_K_M"
        assert _detect_quant("model-Q5_K_S.gguf") == "Q5_K_S"
        assert _detect_quant("model-F16.gguf") == "F16"
        assert _detect_quant("model-Q8_0.gguf") == "Q8_0"
        assert _detect_quant("model.gguf") is None

    def test_is_quantization(self):
        """Verifica detección de cadenas de cuantización."""
        from hfl.hub.resolver import _is_quantization

        # Cuantizaciones válidas
        assert _is_quantization("Q4_K_M") is True
        assert _is_quantization("q4_k_m") is True  # case insensitive
        assert _is_quantization("Q5_K_S") is True
        assert _is_quantization("Q8_0") is True
        assert _is_quantization("F16") is True
        assert _is_quantization("F32") is True
        assert _is_quantization("IQ4_NL") is True
        assert _is_quantization("IQ2_XS") is True

        # No son cuantizaciones
        assert _is_quantization("main") is False
        assert _is_quantization("v1.0") is False
        assert _is_quantization("latest") is False
        assert _is_quantization("") is False
        assert _is_quantization("model-name") is False

    def test_resolve_with_colon_quantization(self, mock_hf_api, sample_gguf_model_info):
        """Verifica resolución con formato repo:quantization (estilo Ollama)."""
        mock_hf_api.model_info.return_value = sample_gguf_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            result = resolve("test-org/test-model-gguf:Q5_K_M")

        assert result.repo_id == "test-org/test-model-gguf"
        assert result.quantization == "Q5_K_M"
        assert "Q5_K_M" in result.filename

    def test_resolve_colon_not_quantization(self, mock_hf_api, sample_gguf_model_info):
        """Verifica que : sin cuantización válida no se parsea incorrectamente."""
        # Simular un repo que tiene : en el nombre (raro pero posible en búsquedas)
        mock_model = MagicMock()
        mock_model.id = "org/model"
        mock_hf_api.list_models.return_value = [mock_model]
        mock_hf_api.model_info.return_value = sample_gguf_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            # "model:latest" - latest no es cuantización, debería buscar "model:latest"
            result = resolve("model:latest")

        # Debería haber buscado el modelo
        mock_hf_api.list_models.assert_called()

    def test_resolve_quantization_priority(self, mock_hf_api, sample_gguf_model_info):
        """Verifica que cuantización en spec tiene prioridad sobre parámetro."""
        mock_hf_api.model_info.return_value = sample_gguf_model_info

        from hfl.hub.resolver import resolve

        with patch("hfl.hub.resolver.HfApi", return_value=mock_hf_api):
            # Q5_K_M en spec debería tener prioridad sobre Q4_K_M en parámetro
            result = resolve("test-org/test-model-gguf:Q5_K_M", quantization="Q4_K_M")

        assert result.quantization == "Q5_K_M"


class TestDownloader:
    """Tests para hub/downloader.py."""

    def test_pull_model_gguf(self, mock_hf_api, temp_config):
        """Verifica descarga de modelo GGUF."""
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
        """Verifica descarga de modelo safetensors (snapshot)."""
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
                # Verificar allow_patterns para safetensors
                call_kwargs = mock_download.call_args[1]
                assert "*.safetensors" in call_kwargs["allow_patterns"]
                assert "config.json" in call_kwargs["allow_patterns"]

    def test_pull_model_with_auth(self, mock_hf_api, temp_config):
        """Verifica descarga con autenticación."""
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
        """Verifica que se crea el directorio del modelo."""
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

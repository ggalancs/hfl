# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests para el módulo exceptions."""

import pytest


class TestHFLError:
    """Tests para la excepción base."""

    def test_basic_message(self):
        """Mensaje básico sin detalles."""
        from hfl.exceptions import HFLError

        err = HFLError("Test error")
        assert str(err) == "Test error"
        assert err.message == "Test error"
        assert err.details is None

    def test_message_with_details(self):
        """Mensaje con detalles adicionales."""
        from hfl.exceptions import HFLError

        err = HFLError("Test error", "Additional details")
        assert "Test error" in str(err)
        assert "Additional details" in str(err)
        assert err.details == "Additional details"


class TestModelErrors:
    """Tests para errores de modelos."""

    def test_model_not_found(self):
        """ModelNotFoundError tiene mensaje útil."""
        from hfl.exceptions import ModelNotFoundError

        err = ModelNotFoundError("test-model")
        assert "test-model" in str(err)
        assert "hfl list" in str(err)
        assert err.model_name == "test-model"

    def test_model_already_exists(self):
        """ModelAlreadyExistsError tiene nombre del modelo."""
        from hfl.exceptions import ModelAlreadyExistsError

        err = ModelAlreadyExistsError("existing-model")
        assert "existing-model" in str(err)
        assert err.model_name == "existing-model"


class TestDownloadErrors:
    """Tests para errores de descarga."""

    def test_download_error(self):
        """DownloadError básico."""
        from hfl.exceptions import DownloadError

        err = DownloadError("org/model", "Connection timeout")
        assert "org/model" in str(err)
        assert "Connection timeout" in str(err)
        assert err.repo_id == "org/model"

    def test_network_error(self):
        """NetworkError hereda de DownloadError."""
        from hfl.exceptions import NetworkError, DownloadError

        err = NetworkError("org/model", "DNS resolution failed")
        assert isinstance(err, DownloadError)
        assert "Error de red" in str(err)


class TestConversionErrors:
    """Tests para errores de conversión."""

    def test_conversion_error(self):
        """ConversionError tiene formatos."""
        from hfl.exceptions import ConversionError

        err = ConversionError("safetensors", "gguf", "Out of memory")
        assert "safetensors" in str(err)
        assert "gguf" in str(err)
        assert err.source_format == "safetensors"
        assert err.target_format == "gguf"

    def test_tool_not_found(self):
        """ToolNotFoundError hereda de ConversionError."""
        from hfl.exceptions import ToolNotFoundError, ConversionError

        err = ToolNotFoundError("llama-quantize")
        assert isinstance(err, ConversionError)
        assert "llama-quantize" in str(err)
        assert err.tool_name == "llama-quantize"


class TestLicenseErrors:
    """Tests para errores de licencia."""

    def test_license_not_accepted(self):
        """LicenseNotAcceptedError tiene info de licencia."""
        from hfl.exceptions import LicenseNotAcceptedError, LicenseError

        err = LicenseNotAcceptedError("meta-llama/Llama-3", "llama3")
        assert isinstance(err, LicenseError)
        assert "meta-llama/Llama-3" in str(err)
        assert "llama3" in str(err)
        assert err.repo_id == "meta-llama/Llama-3"
        assert err.license_type == "llama3"

    def test_gated_model_error(self):
        """GatedModelError tiene URL de aceptación."""
        from hfl.exceptions import GatedModelError

        err = GatedModelError("meta-llama/Llama-3")
        assert "huggingface.co/meta-llama/Llama-3" in str(err)
        assert err.repo_id == "meta-llama/Llama-3"


class TestEngineErrors:
    """Tests para errores de motor."""

    def test_model_not_loaded(self):
        """ModelNotLoadedError mensaje claro."""
        from hfl.exceptions import ModelNotLoadedError, EngineError

        err = ModelNotLoadedError()
        assert isinstance(err, EngineError)
        assert "no cargado" in str(err).lower()

    def test_missing_dependency(self):
        """MissingDependencyError tiene comando de instalación."""
        from hfl.exceptions import MissingDependencyError

        err = MissingDependencyError(
            "llama-cpp", "llama-cpp-python", "pip install llama-cpp-python"
        )
        assert "llama-cpp" in str(err)
        assert "pip install" in str(err)
        assert err.engine_name == "llama-cpp"
        assert err.package == "llama-cpp-python"

    def test_out_of_memory(self):
        """OutOfMemoryError tiene info de memoria."""
        from hfl.exceptions import OutOfMemoryError

        err = OutOfMemoryError(required_gb=16.0, available_gb=8.0)
        assert "16" in str(err)
        assert "8" in str(err)
        assert err.required_gb == 16.0
        assert err.available_gb == 8.0


class TestAuthErrors:
    """Tests para errores de autenticación."""

    def test_invalid_token(self):
        """InvalidTokenError tiene URL de tokens."""
        from hfl.exceptions import InvalidTokenError, AuthenticationError

        err = InvalidTokenError()
        assert isinstance(err, AuthenticationError)
        assert "huggingface.co/settings/tokens" in str(err)

    def test_token_required(self):
        """TokenRequiredError tiene repo_id."""
        from hfl.exceptions import TokenRequiredError

        err = TokenRequiredError("private/model")
        assert "private/model" in str(err)
        assert "hfl login" in str(err)
        assert err.repo_id == "private/model"


class TestConfigErrors:
    """Tests para errores de configuración."""

    def test_invalid_config_basic(self):
        """InvalidConfigError mensaje básico."""
        from hfl.exceptions import InvalidConfigError, ConfigurationError

        err = InvalidConfigError("quantization", "INVALID")
        assert isinstance(err, ConfigurationError)
        assert "quantization" in str(err)
        assert "INVALID" in str(err)

    def test_invalid_config_with_valid_values(self):
        """InvalidConfigError con valores válidos."""
        from hfl.exceptions import InvalidConfigError

        err = InvalidConfigError(
            "quantization", "INVALID", ["Q4_K_M", "Q5_K_M", "Q8_0"]
        )
        assert "Q4_K_M" in str(err)
        assert "Q5_K_M" in str(err)
        assert err.key == "quantization"
        assert err.value == "INVALID"


class TestExceptionHierarchy:
    """Tests para verificar la jerarquía de excepciones."""

    def test_all_inherit_from_hfl_error(self):
        """Todas las excepciones heredan de HFLError."""
        from hfl.exceptions import (
            HFLError,
            ModelNotFoundError,
            DownloadError,
            ConversionError,
            LicenseError,
            EngineError,
            AuthenticationError,
            ConfigurationError,
        )

        exceptions = [
            ModelNotFoundError("x"),
            DownloadError("x", "y"),
            ConversionError("a", "b", "c"),
            LicenseError("x"),
            EngineError("x"),
            AuthenticationError("x"),
            ConfigurationError("x"),
        ]

        for exc in exceptions:
            assert isinstance(exc, HFLError)
            assert isinstance(exc, Exception)

    def test_can_catch_by_base_class(self):
        """Se pueden capturar por clase base."""
        from hfl.exceptions import HFLError, ModelNotFoundError

        try:
            raise ModelNotFoundError("test")
        except HFLError as e:
            assert e.model_name == "test"

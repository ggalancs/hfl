# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the exceptions module."""


class TestHFLError:
    """Tests for the base exception."""

    def test_basic_message(self):
        """Basic message without details."""
        from hfl.exceptions import HFLError

        err = HFLError("Test error")
        assert str(err) == "Test error"
        assert err.message == "Test error"
        assert err.details is None

    def test_message_with_details(self):
        """Message with additional details."""
        from hfl.exceptions import HFLError

        err = HFLError("Test error", "Additional details")
        assert "Test error" in str(err)
        assert "Additional details" in str(err)
        assert err.details == "Additional details"


class TestModelErrors:
    """Tests for model errors."""

    def test_model_not_found(self):
        """ModelNotFoundError has useful message."""
        from hfl.exceptions import ModelNotFoundError

        err = ModelNotFoundError("test-model")
        assert "test-model" in str(err)
        assert "hfl list" in str(err)
        assert err.model_name == "test-model"

    def test_model_already_exists(self):
        """ModelAlreadyExistsError has model name."""
        from hfl.exceptions import ModelAlreadyExistsError

        err = ModelAlreadyExistsError("existing-model")
        assert "existing-model" in str(err)
        assert err.model_name == "existing-model"


class TestDownloadErrors:
    """Tests for download errors."""

    def test_download_error(self):
        """Basic DownloadError."""
        from hfl.exceptions import DownloadError

        err = DownloadError("org/model", "Connection timeout")
        assert "org/model" in str(err)
        assert "Connection timeout" in str(err)
        assert err.repo_id == "org/model"

    def test_network_error(self):
        """NetworkError inherits from DownloadError."""
        from hfl.exceptions import DownloadError, NetworkError

        err = NetworkError("org/model", "DNS resolution failed")
        assert isinstance(err, DownloadError)
        assert "Network error" in str(err)


class TestConversionErrors:
    """Tests for conversion errors."""

    def test_conversion_error(self):
        """ConversionError has formats."""
        from hfl.exceptions import ConversionError

        err = ConversionError("safetensors", "gguf", "Out of memory")
        assert "safetensors" in str(err)
        assert "gguf" in str(err)
        assert err.source_format == "safetensors"
        assert err.target_format == "gguf"

    def test_tool_not_found(self):
        """ToolNotFoundError inherits from ConversionError."""
        from hfl.exceptions import ConversionError, ToolNotFoundError

        err = ToolNotFoundError("llama-quantize")
        assert isinstance(err, ConversionError)
        assert "llama-quantize" in str(err)
        assert err.tool_name == "llama-quantize"


class TestLicenseErrors:
    """Tests for license errors."""

    def test_license_not_accepted(self):
        """LicenseNotAcceptedError has license info."""
        from hfl.exceptions import LicenseError, LicenseNotAcceptedError

        err = LicenseNotAcceptedError("meta-llama/Llama-3", "llama3")
        assert isinstance(err, LicenseError)
        assert "meta-llama/Llama-3" in str(err)
        assert "llama3" in str(err)
        assert err.repo_id == "meta-llama/Llama-3"
        assert err.license_type == "llama3"

    def test_gated_model_error(self):
        """GatedModelError has acceptance URL."""
        from hfl.exceptions import GatedModelError

        err = GatedModelError("meta-llama/Llama-3")
        assert "huggingface.co/meta-llama/Llama-3" in str(err)
        assert err.repo_id == "meta-llama/Llama-3"


class TestEngineErrors:
    """Tests for engine errors."""

    def test_model_not_loaded(self):
        """ModelNotLoadedError clear message."""
        from hfl.exceptions import EngineError, ModelNotLoadedError

        err = ModelNotLoadedError()
        assert isinstance(err, EngineError)
        assert "not loaded" in str(err).lower()

    def test_missing_dependency(self):
        """MissingDependencyError has install command."""
        from hfl.exceptions import MissingDependencyError

        err = MissingDependencyError(
            "llama-cpp", "llama-cpp-python", "pip install llama-cpp-python"
        )
        assert "llama-cpp" in str(err)
        assert "pip install" in str(err)
        assert err.engine_name == "llama-cpp"
        assert err.package == "llama-cpp-python"

    def test_out_of_memory(self):
        """OutOfMemoryError has memory info."""
        from hfl.exceptions import OutOfMemoryError

        err = OutOfMemoryError(required_gb=16.0, available_gb=8.0)
        assert "16" in str(err)
        assert "8" in str(err)
        assert err.required_gb == 16.0
        assert err.available_gb == 8.0


class TestAuthErrors:
    """Tests for authentication errors."""

    def test_invalid_token(self):
        """InvalidTokenError has tokens URL."""
        from hfl.exceptions import AuthenticationError, InvalidTokenError

        err = InvalidTokenError()
        assert isinstance(err, AuthenticationError)
        assert "huggingface.co/settings/tokens" in str(err)

    def test_token_required(self):
        """TokenRequiredError has repo_id."""
        from hfl.exceptions import TokenRequiredError

        err = TokenRequiredError("private/model")
        assert "private/model" in str(err)
        assert "hfl login" in str(err)
        assert err.repo_id == "private/model"


class TestConfigErrors:
    """Tests for configuration errors."""

    def test_invalid_config_basic(self):
        """InvalidConfigError basic message."""
        from hfl.exceptions import ConfigurationError, InvalidConfigError

        err = InvalidConfigError("quantization", "INVALID")
        assert isinstance(err, ConfigurationError)
        assert "quantization" in str(err)
        assert "INVALID" in str(err)

    def test_invalid_config_with_valid_values(self):
        """InvalidConfigError with valid values."""
        from hfl.exceptions import InvalidConfigError

        err = InvalidConfigError("quantization", "INVALID", ["Q4_K_M", "Q5_K_M", "Q8_0"])
        assert "Q4_K_M" in str(err)
        assert "Q5_K_M" in str(err)
        assert err.key == "quantization"
        assert err.value == "INVALID"


class TestExceptionHierarchy:
    """Tests to verify exception hierarchy."""

    def test_all_inherit_from_hfl_error(self):
        """All exceptions inherit from HFLError."""
        from hfl.exceptions import (
            AuthenticationError,
            ConfigurationError,
            ConversionError,
            DownloadError,
            EngineError,
            HFLError,
            LicenseError,
            ModelNotFoundError,
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
        """Can catch by base class."""
        from hfl.exceptions import HFLError, ModelNotFoundError

        try:
            raise ModelNotFoundError("test")
        except HFLError as e:
            assert e.model_name == "test"

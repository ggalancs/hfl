# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Custom exception hierarchy for hfl.

Using specific exceptions allows:
- Better error handling in client code
- Clearer error messages for the user
- Easier testing and debugging
"""


class HFLError(Exception):
    """Base exception for all hfl errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\n{self.details}"
        return self.message


# --- Model Errors ---


class ModelNotFoundError(HFLError):
    """The requested model was not found in the local registry."""

    status_code = 404

    def __init__(self, model_name: str):
        super().__init__(
            f"Model not found: {model_name}",
            "Use 'hfl list' to see available models or 'hfl pull' to download.",
        )
        self.model_name = model_name


class ModelAlreadyExistsError(HFLError):
    """The model already exists in the local registry."""

    # 400 Bad Request — the caller asked us to create a duplicate.
    # Ollama returns 400 on /api/copy conflict; matching that lets
    # clients distinguish "target exists" from "source not found".
    status_code = 400

    def __init__(self, model_name: str):
        super().__init__(f"Model already exists: {model_name}")
        self.model_name = model_name


# --- Download Errors ---


class DownloadError(HFLError):
    """Error during model download."""

    def __init__(self, repo_id: str, reason: str):
        super().__init__(
            f"Error downloading {repo_id}",
            reason,
        )
        self.repo_id = repo_id


class NetworkError(DownloadError):
    """Network error during download."""

    def __init__(self, repo_id: str, reason: str):
        super().__init__(repo_id, f"Network error: {reason}")


# --- Conversion Errors ---


class ConversionError(HFLError):
    """Error during model format conversion."""

    def __init__(self, source_format: str, target_format: str, reason: str):
        super().__init__(
            f"Error converting from {source_format} to {target_format}",
            reason,
        )
        self.source_format = source_format
        self.target_format = target_format


class ToolNotFoundError(ConversionError):
    """Conversion tool not found."""

    def __init__(self, tool_name: str):
        super().__init__(
            "N/A",
            "N/A",
            f"Tool not found: {tool_name}. It will be installed automatically on the next attempt.",
        )
        self.tool_name = tool_name


# --- License Errors ---


class LicenseError(HFLError):
    """Error related to model licenses."""

    pass


class LicenseNotAcceptedError(LicenseError):
    """The user did not accept the model license."""

    def __init__(self, repo_id: str, license_type: str):
        super().__init__(
            f"License not accepted for {repo_id}",
            f"This model requires accepting the '{license_type}' license.",
        )
        self.repo_id = repo_id
        self.license_type = license_type


class GatedModelError(LicenseError):
    """The model requires prior acceptance on HuggingFace."""

    def __init__(self, repo_id: str):
        super().__init__(
            f"Gated model: {repo_id}",
            f"You must accept the terms at https://huggingface.co/{repo_id} "
            "before you can download it.",
        )
        self.repo_id = repo_id


# --- Inference Engine Errors ---


class EngineError(HFLError):
    """Inference engine error."""

    pass


class ModelNotLoadedError(EngineError):
    """Attempted to use the engine without loading a model."""

    def __init__(self):
        super().__init__(
            "Model not loaded",
            "You must load a model before generating text.",
        )


class MissingDependencyError(EngineError):
    """Engine dependency not installed."""

    def __init__(self, engine_name: str, package: str, install_cmd: str):
        super().__init__(
            f"Missing dependency for {engine_name}",
            f"Install with: {install_cmd}",
        )
        self.engine_name = engine_name
        self.package = package
        self.install_cmd = install_cmd


class OutOfMemoryError(EngineError):
    """Not enough memory for the model."""

    def __init__(self, required_gb: float, available_gb: float):
        super().__init__(
            "Insufficient memory",
            f"The model requires ~{required_gb:.1f}GB but only {available_gb:.1f}GB are available. "
            "Try a smaller model or more aggressive quantization (Q4_K_S, Q3_K_M).",
        )
        self.required_gb = required_gb
        self.available_gb = available_gb


# --- Authentication Errors ---


class AuthenticationError(HFLError):
    """Authentication error with HuggingFace."""

    pass


class InvalidTokenError(AuthenticationError):
    """Invalid HuggingFace token."""

    def __init__(self):
        super().__init__(
            "Invalid token",
            "Verify your token at https://huggingface.co/settings/tokens "
            "or use 'hfl login' to configure it.",
        )


class TokenRequiredError(AuthenticationError):
    """Token required to access the resource."""

    def __init__(self, repo_id: str):
        super().__init__(
            f"Token required for {repo_id}",
            "Use 'hfl login' to configure your HuggingFace token.",
        )
        self.repo_id = repo_id


# --- Configuration Errors ---


class ConfigurationError(HFLError):
    """Configuration error."""

    pass


class InvalidConfigError(ConfigurationError):
    """Invalid configuration value."""

    def __init__(self, key: str, value: str, valid_values: list[str] | None = None):
        details = f"Invalid value for '{key}': {value}"
        if valid_values:
            details += f"\nValid values: {', '.join(valid_values)}"
        super().__init__("Configuration error", details)
        self.key = key
        self.value = value


# --- API Errors ---


class APIError(HFLError):
    """Base for API-related errors with HTTP status code."""

    status_code: int = 500

    def __init__(self, message: str, details: str | None = None, status_code: int | None = None):
        super().__init__(message, details)
        if status_code is not None:
            self.status_code = status_code


class ValidationError(APIError):
    """Request validation failed."""

    status_code = 400


class RateLimitError(APIError):
    """Rate limit exceeded."""

    status_code = 429

    def __init__(self, retry_after: int = 60):
        super().__init__(
            "Rate limit exceeded",
            f"Please try again in {retry_after} seconds.",
        )
        self.retry_after = retry_after


class ModelNotReadyError(APIError):
    """Model is not loaded and ready."""

    status_code = 503

    def __init__(self, model_name: str | None = None):
        msg = "Model not loaded"
        if model_name:
            msg = f"Model '{model_name}' not loaded"
        super().__init__(msg, "Load a model first using the API or CLI.")


class GenerationTimeoutError(APIError):
    """Generation request timed out."""

    status_code = 504

    def __init__(self, timeout_seconds: float, operation: str = "generation"):
        super().__init__(
            f"{operation} timed out",
            f"Request exceeded {timeout_seconds}s timeout."
            " Try a shorter prompt or lower max_tokens.",
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class ModelTypeMismatchError(APIError):
    """Requested model is the wrong type for the endpoint.

    E.g. a TTS endpoint was asked to load an LLM, or vice-versa.
    Maps to HTTP 400 through the global exception handler.
    """

    status_code = 400

    def __init__(self, model_name: str, expected: str, got: str):
        super().__init__(
            f"Model '{model_name}' is not a {expected} model",
            f"Detected model type: {got}. Use a {expected}-compatible model.",
        )
        self.model_name = model_name
        self.expected = expected
        self.got = got


# --- Engine-specific Errors ---


class ModelLoadError(EngineError):
    """Failed to load model."""

    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"Failed to load model: {model_path}",
            reason,
        )
        self.model_path = model_path


class GenerationError(EngineError):
    """Error during text/audio generation."""

    def __init__(self, reason: str):
        super().__init__("Generation failed", reason)

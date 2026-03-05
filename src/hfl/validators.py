# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Input validation utilities for HFL.

Security-critical module that provides:
- Path traversal prevention
- Input bounds validation
- Model name sanitization
- Quantization validation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from hfl.exceptions import HFLError


class ValidationError(HFLError):
    """Input validation error."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


# Regex for valid model names: alphanumeric start, then alphanumeric, dash, dot, underscore, slash
MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.\/]*$")
MAX_MODEL_NAME_LENGTH = 256


@dataclass(frozen=True)
class InputBounds:
    """Immutable bounds for input validation."""

    # Generation parameters
    MAX_TOKENS_LIMIT: int = 131072  # 128K
    MIN_TOKENS: int = 1
    MAX_CONTEXT_SIZE: int = 262144  # 256K
    MIN_CONTEXT_SIZE: int = 128
    TEMPERATURE_MIN: float = 0.0
    TEMPERATURE_MAX: float = 2.0
    TOP_P_MIN: float = 0.0
    TOP_P_MAX: float = 1.0
    TOP_K_MIN: int = 1
    TOP_K_MAX: int = 1000
    REPEAT_PENALTY_MIN: float = 0.0
    REPEAT_PENALTY_MAX: float = 10.0

    # Content limits
    MAX_PROMPT_LENGTH: int = 2_000_000  # 2M chars
    MAX_MESSAGES: int = 10000
    MAX_MESSAGE_LENGTH: int = 500_000  # 500K chars per message

    # Server
    MIN_PORT: int = 1
    MAX_PORT: int = 65535

    # TTS
    MIN_SPEED: float = 0.25
    MAX_SPEED: float = 4.0
    MIN_SAMPLE_RATE: int = 8000
    MAX_SAMPLE_RATE: int = 48000


BOUNDS = InputBounds()

# Valid quantization levels
VALID_QUANTIZATIONS = frozenset(
    {
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_0",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_0",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
        "F16",
        "F32",
    }
)


def validate_model_name(name: str) -> str:
    """Validate model name format and prevent path traversal.

    Args:
        name: Model name to validate

    Returns:
        Validated model name

    Raises:
        ValidationError: If name is invalid or contains path traversal

    Examples:
        >>> validate_model_name("meta-llama/Llama-3-8B")
        'meta-llama/Llama-3-8B'
        >>> validate_model_name("../../../etc/passwd")  # Raises ValidationError
    """
    if not name:
        raise ValidationError("Model name cannot be empty", field="model")

    if len(name) > MAX_MODEL_NAME_LENGTH:
        raise ValidationError(
            f"Model name too long: {len(name)} > {MAX_MODEL_NAME_LENGTH}",
            field="model",
        )

    # Check for path traversal patterns FIRST
    if ".." in name:
        raise ValidationError(
            f"Path traversal detected in model name: {name}",
            field="model",
        )

    if name.startswith("/") or name.startswith("~"):
        raise ValidationError(
            f"Absolute paths not allowed in model name: {name}",
            field="model",
        )

    if name.startswith("\\") or "\\" in name:
        raise ValidationError(
            f"Backslashes not allowed in model name: {name}",
            field="model",
        )

    # Validate format
    if not MODEL_NAME_PATTERN.match(name):
        raise ValidationError(
            f"Invalid model name format: {name}. "
            "Must start with alphanumeric, containing only alphanumeric/dash/dot/underscore/slash.",
            field="model",
        )

    return name


def validate_quantization(quant: str) -> str:
    """Validate quantization level.

    Args:
        quant: Quantization level (e.g., "Q4_K_M", "Q8_0")

    Returns:
        Normalized quantization level (uppercase)

    Raises:
        ValidationError: If quantization level is invalid
    """
    normalized = quant.upper().strip()

    if normalized not in VALID_QUANTIZATIONS:
        valid_list = ", ".join(sorted(VALID_QUANTIZATIONS))
        raise ValidationError(
            f"Invalid quantization level: {quant}. Valid options: {valid_list}",
            field="quantization",
        )

    return normalized


def validate_port(port: int) -> int:
    """Validate port number.

    Args:
        port: Port number

    Returns:
        Validated port number

    Raises:
        ValidationError: If port is out of range
    """
    if not isinstance(port, int):
        raise ValidationError(f"Port must be an integer, got {type(port).__name__}", field="port")

    if not BOUNDS.MIN_PORT <= port <= BOUNDS.MAX_PORT:
        raise ValidationError(
            f"Port must be between {BOUNDS.MIN_PORT} and {BOUNDS.MAX_PORT}, got {port}",
            field="port",
        )

    return port


def validate_generation_params(
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    ctx_size: int | None = None,
    repeat_penalty: float | None = None,
) -> None:
    """Validate generation parameters are within safe bounds.

    Args:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        ctx_size: Context size
        repeat_penalty: Repetition penalty

    Raises:
        ValidationError: If any parameter is out of bounds
    """
    if max_tokens is not None:
        if not isinstance(max_tokens, int):
            raise ValidationError(
                f"max_tokens must be an integer, got {type(max_tokens).__name__}",
                field="max_tokens",
            )
        if max_tokens < BOUNDS.MIN_TOKENS or max_tokens > BOUNDS.MAX_TOKENS_LIMIT:
            raise ValidationError(
                f"max_tokens must be between {BOUNDS.MIN_TOKENS} and {BOUNDS.MAX_TOKENS_LIMIT}, "
                f"got {max_tokens}",
                field="max_tokens",
            )

    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise ValidationError(
                f"temperature must be a number, got {type(temperature).__name__}",
                field="temperature",
            )
        if not BOUNDS.TEMPERATURE_MIN <= temperature <= BOUNDS.TEMPERATURE_MAX:
            raise ValidationError(
                f"temperature must be between {BOUNDS.TEMPERATURE_MIN} and "
                f"{BOUNDS.TEMPERATURE_MAX}, got {temperature}",
                field="temperature",
            )

    if top_p is not None:
        if not isinstance(top_p, (int, float)):
            raise ValidationError(
                f"top_p must be a number, got {type(top_p).__name__}",
                field="top_p",
            )
        if not BOUNDS.TOP_P_MIN <= top_p <= BOUNDS.TOP_P_MAX:
            raise ValidationError(
                f"top_p must be between {BOUNDS.TOP_P_MIN} and {BOUNDS.TOP_P_MAX}, got {top_p}",
                field="top_p",
            )

    if top_k is not None:
        if not isinstance(top_k, int):
            raise ValidationError(
                f"top_k must be an integer, got {type(top_k).__name__}",
                field="top_k",
            )
        if not BOUNDS.TOP_K_MIN <= top_k <= BOUNDS.TOP_K_MAX:
            raise ValidationError(
                f"top_k must be between {BOUNDS.TOP_K_MIN} and {BOUNDS.TOP_K_MAX}, got {top_k}",
                field="top_k",
            )

    if ctx_size is not None:
        if not isinstance(ctx_size, int):
            raise ValidationError(
                f"context_size must be an integer, got {type(ctx_size).__name__}",
                field="context_size",
            )
        if not BOUNDS.MIN_CONTEXT_SIZE <= ctx_size <= BOUNDS.MAX_CONTEXT_SIZE:
            raise ValidationError(
                f"context_size must be between {BOUNDS.MIN_CONTEXT_SIZE} and "
                f"{BOUNDS.MAX_CONTEXT_SIZE}, got {ctx_size}",
                field="context_size",
            )

    if repeat_penalty is not None:
        if not isinstance(repeat_penalty, (int, float)):
            raise ValidationError(
                f"repeat_penalty must be a number, got {type(repeat_penalty).__name__}",
                field="repeat_penalty",
            )
        if not BOUNDS.REPEAT_PENALTY_MIN <= repeat_penalty <= BOUNDS.REPEAT_PENALTY_MAX:
            raise ValidationError(
                f"repeat_penalty must be between {BOUNDS.REPEAT_PENALTY_MIN} and "
                f"{BOUNDS.REPEAT_PENALTY_MAX}, got {repeat_penalty}",
                field="repeat_penalty",
            )


def validate_messages(messages: list[dict[str, Any]]) -> None:
    """Validate chat messages.

    Args:
        messages: List of chat messages

    Raises:
        ValidationError: If messages are invalid
    """
    if not isinstance(messages, list):
        raise ValidationError(
            f"messages must be a list, got {type(messages).__name__}",
            field="messages",
        )

    if len(messages) == 0:
        raise ValidationError("messages cannot be empty", field="messages")

    if len(messages) > BOUNDS.MAX_MESSAGES:
        raise ValidationError(
            f"Too many messages: {len(messages)} > {BOUNDS.MAX_MESSAGES}",
            field="messages",
        )

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValidationError(
                f"Message {i} must be a dict, got {type(msg).__name__}",
                field=f"messages[{i}]",
            )

        if "role" not in msg:
            raise ValidationError(
                f"Message {i} missing 'role' field",
                field=f"messages[{i}].role",
            )

        if "content" not in msg:
            raise ValidationError(
                f"Message {i} missing 'content' field",
                field=f"messages[{i}].content",
            )

        content = msg["content"]
        if isinstance(content, str) and len(content) > BOUNDS.MAX_MESSAGE_LENGTH:
            raise ValidationError(
                f"Message {i} content too long: {len(content)} > {BOUNDS.MAX_MESSAGE_LENGTH}",
                field=f"messages[{i}].content",
            )


def validate_prompt(prompt: str | list[str]) -> None:
    """Validate prompt content.

    Args:
        prompt: Single prompt string or list of prompts

    Raises:
        ValidationError: If prompt is invalid
    """
    if isinstance(prompt, str):
        if len(prompt) > BOUNDS.MAX_PROMPT_LENGTH:
            raise ValidationError(
                f"Prompt too long: {len(prompt)} > {BOUNDS.MAX_PROMPT_LENGTH}",
                field="prompt",
            )
    elif isinstance(prompt, list):
        for i, p in enumerate(prompt):
            if not isinstance(p, str):
                raise ValidationError(
                    f"Prompt {i} must be a string, got {type(p).__name__}",
                    field=f"prompt[{i}]",
                )
            if len(p) > BOUNDS.MAX_PROMPT_LENGTH:
                raise ValidationError(
                    f"Prompt {i} too long: {len(p)} > {BOUNDS.MAX_PROMPT_LENGTH}",
                    field=f"prompt[{i}]",
                )
    else:
        raise ValidationError(
            f"prompt must be a string or list of strings, got {type(prompt).__name__}",
            field="prompt",
        )


def validate_tts_params(
    *,
    speed: float | None = None,
    sample_rate: int | None = None,
) -> None:
    """Validate TTS parameters.

    Args:
        speed: Speech speed multiplier
        sample_rate: Audio sample rate

    Raises:
        ValidationError: If any parameter is out of bounds
    """
    if speed is not None:
        if not isinstance(speed, (int, float)):
            raise ValidationError(
                f"speed must be a number, got {type(speed).__name__}",
                field="speed",
            )
        if not BOUNDS.MIN_SPEED <= speed <= BOUNDS.MAX_SPEED:
            raise ValidationError(
                f"speed must be between {BOUNDS.MIN_SPEED} and {BOUNDS.MAX_SPEED}, got {speed}",
                field="speed",
            )

    if sample_rate is not None:
        if not isinstance(sample_rate, int):
            raise ValidationError(
                f"sample_rate must be an integer, got {type(sample_rate).__name__}",
                field="sample_rate",
            )
        if not BOUNDS.MIN_SAMPLE_RATE <= sample_rate <= BOUNDS.MAX_SAMPLE_RATE:
            raise ValidationError(
                f"sample_rate must be between {BOUNDS.MIN_SAMPLE_RATE} and "
                f"{BOUNDS.MAX_SAMPLE_RATE}, got {sample_rate}",
                field="sample_rate",
            )


def validate_alias(alias: str) -> str:
    """Validate model alias.

    Args:
        alias: Alias string

    Returns:
        Validated alias

    Raises:
        ValidationError: If alias is invalid
    """
    if not alias:
        raise ValidationError("Alias cannot be empty", field="alias")

    if len(alias) > 64:
        raise ValidationError(
            f"Alias too long: {len(alias)} > 64",
            field="alias",
        )

    # Alias should be simpler than model name - no slashes
    alias_pattern = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$")
    if not alias_pattern.match(alias):
        raise ValidationError(
            f"Invalid alias format: {alias}. "
            "Must start with alphanumeric and contain only alphanumeric, dash, dot, or underscore.",
            field="alias",
        )

    return alias

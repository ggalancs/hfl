# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Request/response converters for API compatibility.

Consolidates conversion logic between different API formats:
- OpenAI API format
- Ollama API format
- Internal GenerationConfig
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hfl.engine.base import GenerationConfig

if TYPE_CHECKING:
    from hfl.api.schemas import ChatCompletionRequest, CompletionRequest


def clamp(value: float | int, min_val: float | int, max_val: float | int) -> float | int:
    """Clamp value to range [min_val, max_val].

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def openai_to_generation_config(
    req: ChatCompletionRequest | CompletionRequest,
) -> GenerationConfig:
    """Convert OpenAI API request to GenerationConfig.

    Args:
        req: OpenAI-style request (ChatCompletionRequest or CompletionRequest)

    Returns:
        GenerationConfig for inference
    """
    # Handle stop sequences - can be string or list
    stop = req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else None)

    return GenerationConfig(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens or 2048,
        stop=stop,
        seed=req.seed or -1,
    )


def ollama_to_generation_config(options: dict[str, Any] | None) -> GenerationConfig:
    """Convert Ollama options dict to GenerationConfig.

    Validates and clamps all values to safe ranges per Ollama API spec.

    Args:
        options: Ollama-style options dict (may be None)

    Returns:
        GenerationConfig for inference
    """
    opts = options or {}

    # Validate and clamp all values to safe ranges
    temperature = clamp(float(opts.get("temperature", 0.7)), 0.0, 2.0)
    top_p = clamp(float(opts.get("top_p", 0.9)), 0.0, 1.0)
    top_k = clamp(int(opts.get("top_k", 40)), 1, 1000)
    max_tokens = clamp(int(opts.get("num_predict", 2048)), 1, 128000)
    repeat_penalty = clamp(float(opts.get("repeat_penalty", 1.1)), 0.0, 2.0)
    seed = int(opts.get("seed", -1))

    return GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=int(top_k),
        max_tokens=int(max_tokens),
        repeat_penalty=repeat_penalty,
        seed=seed,
        stop=opts.get("stop"),
    )


def generation_config_to_openai(config: GenerationConfig) -> dict[str, Any]:
    """Convert GenerationConfig to OpenAI-style options dict.

    Useful for logging or returning config in API responses.

    Args:
        config: Internal generation config

    Returns:
        OpenAI-style options dict
    """
    result: dict[str, Any] = {}

    if config.temperature is not None:
        result["temperature"] = config.temperature
    if config.top_p is not None:
        result["top_p"] = config.top_p
    if config.max_tokens is not None:
        result["max_tokens"] = config.max_tokens
    if config.stop:
        result["stop"] = config.stop
    if config.seed is not None and config.seed != -1:
        result["seed"] = config.seed

    return result


def generation_config_to_ollama(config: GenerationConfig) -> dict[str, Any]:
    """Convert GenerationConfig to Ollama-style options dict.

    Useful for logging or returning config in API responses.

    Args:
        config: Internal generation config

    Returns:
        Ollama-style options dict
    """
    result: dict[str, Any] = {}

    if config.temperature is not None:
        result["temperature"] = config.temperature
    if config.top_p is not None:
        result["top_p"] = config.top_p
    if config.top_k is not None:
        result["top_k"] = config.top_k
    if config.max_tokens is not None:
        result["num_predict"] = config.max_tokens
    if config.repeat_penalty is not None:
        result["repeat_penalty"] = config.repeat_penalty
    if config.stop:
        result["stop"] = config.stop
    if config.seed is not None and config.seed != -1:
        result["seed"] = config.seed

    return result

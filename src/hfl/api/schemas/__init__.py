# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Consolidated API schemas.

This module provides all request/response schemas for the HFL API,
organized by API compatibility layer:
- OpenAI-compatible schemas
- Ollama-compatible schemas
- TTS schemas
"""

from hfl.api.schemas.anthropic import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicTextBlock,
)
from hfl.api.schemas.ollama import (
    ChatRequest,
    GenerateRequest,
)
from hfl.api.schemas.openai import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    CompletionRequest,
)
from hfl.api.schemas.tts import (
    AudioFormat,
    NativeTTSRequest,
    OpenAITTSRequest,
    TTSModelInfo,
)

__all__ = [
    # Anthropic
    "AnthropicMessage",
    "AnthropicMessagesRequest",
    "AnthropicTextBlock",
    # OpenAI
    "ChatCompletionMessage",
    "ChatCompletionRequest",
    "CompletionRequest",
    # Ollama
    "GenerateRequest",
    "ChatRequest",
    # TTS
    "AudioFormat",
    "NativeTTSRequest",
    "OpenAITTSRequest",
    "TTSModelInfo",
]

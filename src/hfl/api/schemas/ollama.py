# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible API schemas.

These schemas match the Ollama API specification for drop-in compatibility.
"""

from typing import Literal

from pydantic import BaseModel, Field


class OllamaChatMessage(BaseModel):
    """Validated chat message for Ollama API.

    Ensures messages have proper structure with role and content fields.
    """

    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Role of the message sender",
    )
    content: str = Field(
        ...,
        min_length=0,
        max_length=2_000_000,
        description="Message content",
    )


class GenerateRequest(BaseModel):
    """Request for text generation (Ollama-compatible).

    Compatible with Ollama's /api/generate endpoint.

    Attributes:
        model: Model name or alias to use for generation
        prompt: The prompt text to generate from
        stream: Whether to stream the response (default: True)
        options: Optional generation options (temperature, top_p, etc.)
    """

    model: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Model name or alias",
        examples=["llama3.3-70b-q4", "mistral-7b"],
    )
    prompt: str = Field(
        ...,
        max_length=2_000_000,
        description="Prompt text for generation",
    )
    stream: bool = Field(
        True,
        description="Whether to stream the response",
    )
    options: dict | None = Field(
        None,
        description="Optional generation parameters (temperature, top_p, etc.)",
    )


class ChatRequest(BaseModel):
    """Request for chat completion (Ollama-compatible).

    Compatible with Ollama's /api/chat endpoint.

    Attributes:
        model: Model name or alias to use for chat
        messages: List of message dicts with 'role' and 'content' keys
        stream: Whether to stream the response (default: True)
        options: Optional generation options (temperature, top_p, etc.)
    """

    model: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Model name or alias",
        examples=["llama3.3-70b-q4", "mistral-7b"],
    )
    messages: list[OllamaChatMessage] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Chat messages with role and content",
    )
    stream: bool = Field(
        True,
        description="Whether to stream the response",
    )
    options: dict | None = Field(
        None,
        description="Optional generation parameters (temperature, top_p, etc.)",
    )

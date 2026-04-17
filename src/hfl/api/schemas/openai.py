# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""OpenAI-compatible API schemas.

These schemas match the OpenAI API specification for drop-in compatibility.
"""

from pydantic import BaseModel, Field, field_validator


class ChatCompletionMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message author (system, user, assistant)
        content: The content of the message
    """

    role: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Role of the message author",
        examples=["user", "assistant", "system"],
    )
    content: str = Field(
        ...,
        max_length=2_000_000,
        description="Content of the message",
    )


class ChatCompletionRequest(BaseModel):
    """Request for chat completion.

    Compatible with OpenAI's /v1/chat/completions endpoint.
    """

    model: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Model name or alias",
        examples=["llama3.3-70b-q4", "gpt-4"],
    )
    messages: list[ChatCompletionMessage] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Messages in the conversation",
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    max_tokens: int | None = Field(
        None,
        ge=1,
        le=128000,
        description="Maximum tokens to generate",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response",
    )
    stop: list[str] | str | None = Field(
        None,
        description="Stop sequences",
    )
    frequency_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for token repetition",
    )
    presence_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for new topics",
    )
    seed: int | None = Field(
        None,
        ge=0,
        le=2**32 - 1,
        description="Random seed for reproducibility (unsigned 32-bit)",
    )
    response_format: dict | None = Field(
        None,
        description=(
            "OpenAI JSON-mode envelope. Accepts "
            "{'type':'text'} (default), {'type':'json_object'} for "
            "free-form JSON, or {'type':'json_schema', 'json_schema': "
            "{'name':'...', 'schema':{...}}} for strict schema "
            "conformance. See OLLAMA_PARITY_PLAN P0-5."
        ),
    )

    @field_validator("messages")
    @classmethod
    def validate_total_content(cls, v: list[ChatCompletionMessage]) -> list[ChatCompletionMessage]:
        """Validate total content length across all messages."""
        total = sum(len(m.content) for m in v)
        if total > 2_000_000:
            raise ValueError("Total message content exceeds 2M characters")
        return v

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, v: list[str] | str | None) -> list[str] | str | None:
        """Bound stop sequences: at most 10 entries, each at most 256 chars.

        Prevents a client from pushing the tokenizer into quadratic work
        on every generated token by supplying thousands of long stops.
        """
        if v is None:
            return v
        if isinstance(v, str):
            if len(v) > 256:
                raise ValueError("stop string must be <= 256 characters")
            return v
        if len(v) > 10:
            raise ValueError("stop must contain at most 10 sequences")
        for s in v:
            if not isinstance(s, str):
                raise ValueError("every stop entry must be a string")
            if len(s) > 256:
                raise ValueError("each stop sequence must be <= 256 characters")
        return v


class CompletionRequest(BaseModel):
    """Request for text completion.

    Compatible with OpenAI's /v1/completions endpoint.
    """

    model: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Model name or alias",
    )
    prompt: str | list[str] = Field(
        ...,
        max_length=2_000_000,
        description="Prompt text or list of prompts",
    )
    max_tokens: int = Field(
        256,
        ge=1,
        le=128000,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response",
    )
    stop: list[str] | str | None = Field(
        None,
        description="Stop sequences",
    )
    seed: int | None = Field(
        None,
        ge=0,
        le=2**32 - 1,
        description="Random seed for reproducibility (unsigned 32-bit)",
    )

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, v: list[str] | str | None) -> list[str] | str | None:
        """Same bounds as chat completions (see ChatCompletionRequest.validate_stop)."""
        if v is None:
            return v
        if isinstance(v, str):
            if len(v) > 256:
                raise ValueError("stop string must be <= 256 characters")
            return v
        if len(v) > 10:
            raise ValueError("stop must contain at most 10 sequences")
        for s in v:
            if not isinstance(s, str):
                raise ValueError("every stop entry must be a string")
            if len(s) > 256:
                raise ValueError("each stop sequence must be <= 256 characters")
        return v

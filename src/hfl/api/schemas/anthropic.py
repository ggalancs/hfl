# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Anthropic Messages API-compatible schemas.

These schemas match the Anthropic Messages API specification for compatibility
with tools that use the Anthropic SDK (e.g., Claude Code with ANTHROPIC_BASE_URL).
"""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, Field, field_validator


class AnthropicTextBlock(BaseModel):
    """A text content block in an Anthropic message."""

    type: str = "text"
    text: str


class AnthropicMessage(BaseModel):
    """A single message in an Anthropic conversation.

    Content can be a string or a list of content blocks.
    """

    role: str = Field(
        ...,
        description="Role of the message author (user or assistant)",
    )
    content: Union[str, list[AnthropicTextBlock]] = Field(
        ...,
        description="Message content as string or list of content blocks",
    )

    def get_text(self) -> str:
        """Extract plain text from content regardless of format."""
        if isinstance(self.content, str):
            return self.content
        return "".join(block.text for block in self.content if block.type == "text")


class AnthropicMessagesRequest(BaseModel):
    """Request for Anthropic Messages API.

    Compatible with POST /v1/messages endpoint.
    """

    model: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Model name (may include hfl/ prefix)",
    )
    messages: list[AnthropicMessage] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Messages in the conversation",
    )
    max_tokens: int = Field(
        4096,
        ge=1,
        le=128000,
        description="Maximum tokens to generate",
    )
    system: Union[str, list[AnthropicTextBlock], None] = Field(
        None,
        description="System prompt as string or list of text blocks",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature",
    )
    top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    top_k: int | None = Field(
        None,
        ge=0,
        description="Top-k sampling",
    )
    stop_sequences: list[str] | None = Field(
        None,
        max_length=10,
        description="Stop sequences (at most 10 entries)",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response",
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Request metadata",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[AnthropicMessage]) -> list[AnthropicMessage]:
        """Validate message content length."""
        total = sum(len(m.get_text()) for m in v)
        if total > 2_000_000:
            raise ValueError("Total message content exceeds 2M characters")
        return v

    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, v: list[str] | None) -> list[str] | None:
        """Each stop sequence must be a bounded string (<=256 chars)."""
        if v is None:
            return v
        for s in v:
            if not isinstance(s, str):
                raise ValueError("every stop_sequences entry must be a string")
            if len(s) > 256:
                raise ValueError("each stop sequence must be <= 256 characters")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Bound metadata: at most 64 keys, no value > 1024 chars.

        The Anthropic API itself only recognises a small subset (user_id
        etc.), but accepts arbitrary keys. We cap both breadth and depth
        so a client can't DoS the JSON path with a multi-megabyte dict.
        """
        if v is None:
            return v
        if len(v) > 64:
            raise ValueError("metadata must contain at most 64 keys")
        for key, val in v.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("metadata keys must be strings <= 128 characters")
            if isinstance(val, str) and len(val) > 1024:
                raise ValueError(f"metadata[{key!r}] string value exceeds 1024 characters")
        return v

    def get_system_text(self) -> str | None:
        """Extract system prompt as plain text."""
        if self.system is None:
            return None
        if isinstance(self.system, str):
            return self.system
        return "".join(block.text for block in self.system if block.type == "text")

    def resolve_model_name(self) -> str:
        """Strip provider prefix from model name (e.g., hfl/qwen-coder -> qwen-coder)."""
        if "/" in self.model:
            return self.model.split("/", 1)[1]
        return self.model

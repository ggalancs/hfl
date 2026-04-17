# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""OpenAI-compatible API schemas.

These schemas match the OpenAI API specification for drop-in compatibility.
"""

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------
# Vision / multimodal content parts (Phase 4, P0-6)
# ----------------------------------------------------------------------


class ImageUrl(BaseModel):
    """Inner object of an OpenAI ``image_url`` content part.

    ``url`` accepts either an http(s):// URL (NOT supported — HFL is
    local-first and won't fetch remote assets) or a base64 data URI
    (``data:image/...;base64,...``). Only the data-URI form is
    honoured; http URLs are rejected at the route boundary.

    ``detail`` mirrors OpenAI's hint and is accepted for
    compatibility, though HFL currently ignores it (the vision
    backends run their own resampling).
    """

    url: str = Field(..., min_length=1, max_length=30_000_000)
    detail: Literal["low", "high", "auto"] = Field(default="auto")


class TextContentPart(BaseModel):
    """OpenAI text content part — the plain-text variant."""

    type: Literal["text"] = "text"
    text: str = Field(..., max_length=2_000_000)


class ImageContentPart(BaseModel):
    """OpenAI image content part."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Union[TextContentPart, ImageContentPart]


class ChatCompletionMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message author (system, user, assistant).
        content: Either a plain string (text-only, classical chat) or
            a list of content parts (vision requests, OpenAI-style).
            The bounded-length invariant is the same in both cases:
            total textual length ≤ 2_000_000 characters.
    """

    role: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Role of the message author",
        examples=["user", "assistant", "system"],
    )
    content: str | list[ContentPart] = Field(
        ...,
        description="Text-only content OR a list of text/image parts (vision).",
    )

    @field_validator("content")
    @classmethod
    def _bound_content(cls, v: str | list[ContentPart]) -> str | list[ContentPart]:
        """Enforce size caps on both the string and list shapes."""
        if isinstance(v, str):
            if len(v) > 2_000_000:
                raise ValueError("content string exceeds 2_000_000 characters")
            return v
        if not isinstance(v, list) or not v:
            raise ValueError("content list must be non-empty")
        # Cap the number of parts to prevent boundless message bloat.
        if len(v) > 64:
            raise ValueError("content list must have at most 64 parts")
        # Sum only text lengths toward the 2M cap; images are bounded
        # separately by the image validator.
        total_text = 0
        for part in v:
            if isinstance(part, TextContentPart):
                total_text += len(part.text)
        if total_text > 2_000_000:
            raise ValueError("total text across content parts exceeds 2_000_000 characters")
        return v


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
        """Validate total textual content length across all messages.

        With Phase 4's multimodal content ``list[ContentPart]`` shape,
        we sum only the text parts — image parts are bounded by the
        image validator, not by this character cap.
        """
        total = 0
        for m in v:
            if isinstance(m.content, str):
                total += len(m.content)
            else:
                for part in m.content:
                    # isinstance check would need TextContentPart
                    # import; model_dump lookup avoids the cycle.
                    if isinstance(part, TextContentPart):
                        total += len(part.text)
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

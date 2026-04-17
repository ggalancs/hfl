# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama-compatible API schemas.

These schemas match the Ollama API specification for drop-in compatibility,
including structured tool calling support (tools / tool_calls / role=tool).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class OllamaToolFunctionDef(BaseModel):
    """Definition of a callable tool function (client -> server)."""

    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(None, max_length=4096)
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for the function parameters (must be object)",
    )


class OllamaTool(BaseModel):
    """A tool available to the model."""

    type: Literal["function"] = Field("function")
    function: OllamaToolFunctionDef


class OllamaToolCallFunction(BaseModel):
    """Function portion of a tool call (server -> client or multi-turn input)."""

    name: str = Field(..., min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed JSON arguments as an object (never a string)",
    )


class OllamaToolCall(BaseModel):
    """A single tool call emitted by the assistant."""

    function: OllamaToolCallFunction

    model_config = {"extra": "ignore"}


class OllamaChatMessage(BaseModel):
    """Validated chat message for Ollama API.

    Supports four roles: ``system``, ``user``, ``assistant``, and ``tool``.
    Assistant messages may carry ``tool_calls`` instead of (or alongside)
    ``content``. Tool messages must carry ``name`` bound to the prior call.
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ...,
        description="Role of the message sender",
    )
    content: str | None = Field(
        None,
        max_length=2_000_000,
        description="Message content (may be empty when tool_calls is set)",
    )
    name: str | None = Field(
        None,
        max_length=128,
        description="Tool name (required when role=tool)",
    )
    tool_calls: list[OllamaToolCall] | None = Field(
        None,
        description="Assistant tool calls (canonical wire shape)",
    )
    tool_call_id: str | None = Field(
        None,
        max_length=256,
        description="Optional id linking a tool result to its call",
    )
    # Vision / multimodal — Phase 4, P0-6. Each entry is either a
    # bare base64 string OR a ``data:image/...;base64,...`` URI.
    # The router decodes + validates via hfl.api.image_validator
    # before forwarding to the engine.
    images: list[str] | None = Field(
        None,
        max_length=32,
        description=(
            "Base64-encoded images attached to this message. Accepted "
            "formats: PNG, JPEG, WEBP, GIF. Max 32 per message, each "
            "≤20 MiB, ≤4096x4096 pixels."
        ),
    )

    @model_validator(mode="after")
    def _check_role_fields(self) -> "OllamaChatMessage":
        if self.role == "tool":
            if not self.name:
                raise ValueError("tool messages must include a 'name' field")
            if self.content is None:
                raise ValueError("tool messages must include 'content'")
        elif self.role in ("system", "user"):
            if self.content is None:
                raise ValueError(f"{self.role} messages must include 'content'")
        else:  # assistant
            if self.content is None and not self.tool_calls:
                raise ValueError("assistant messages must include 'content' or 'tool_calls'")
            if self.content is None:
                # Canonical wire: empty string when only tool_calls
                self.content = ""
        return self


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
    keep_alive: str | int | float | None = Field(
        None,
        description=(
            "How long to keep the model loaded after this request. "
            "Accepts Ollama-style values: a Go duration ('5m', '30s', "
            "'1h30m'), raw seconds (10), 0 to unload immediately, -1 "
            "to keep loaded indefinitely, or null for the server default."
        ),
    )
    format: str | dict[str, Any] | None = Field(
        None,
        description=(
            "Structured-output constraint. Accepts 'json' (free-form "
            "JSON) or a JSON Schema object for strict conformance. "
            "See OLLAMA_PARITY_PLAN P0-5."
        ),
    )
    system: str | None = Field(
        None,
        max_length=2_000_000,
        description=(
            "System prompt override for this request. Inserted as the "
            "first message with role=system regardless of what the "
            "Modelfile defined. Ollama-parity (P1-1)."
        ),
    )
    think: bool | None = Field(
        None,
        description=(
            "Expose the model's reasoning / chain-of-thought channel "
            "in the response (Ollama 2026 behaviour). Default False "
            "for thinking-capable models (content stays clean); set "
            "True to receive the full channel output for inspection."
        ),
    )
    template: str | None = Field(
        None,
        max_length=2_000_000,
        description=(
            "Per-request chat-template override (Jinja). Substitutes the "
            "model's default template for this request only. Phase 6 "
            "P2-3 — Ollama-parity."
        ),
    )
    raw: bool | None = Field(
        None,
        description=(
            "Raw prompt mode. When true, the prompt is forwarded to the "
            "model verbatim (no chat template, no BOS). Useful for "
            "evaluation harnesses or manual prompt engineering. Phase 6 "
            "P2-3 — Ollama-parity."
        ),
    )


class ChatRequest(BaseModel):
    """Request for chat completion (Ollama-compatible).

    Compatible with Ollama's /api/chat endpoint, including the structured
    ``tools`` field and the ability to carry ``role=tool`` messages back.

    Attributes:
        model: Model name or alias to use for chat
        messages: List of chat messages
        stream: Whether to stream the response (default: True)
        options: Optional generation options (temperature, top_p, etc.)
        tools: Optional list of tools the model may call
        tool_choice: Optional tool-selection policy ("auto", "none", or
            a specific function name)
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
        description="Chat messages",
    )
    stream: bool = Field(
        True,
        description="Whether to stream the response",
    )
    options: dict | None = Field(
        None,
        description="Optional generation parameters (temperature, top_p, etc.)",
    )
    tools: list[OllamaTool] | None = Field(
        None,
        max_length=128,
        description="Tools the model may call",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        None,
        description="Tool-selection policy (auto, none, or a specific tool)",
    )
    keep_alive: str | int | float | None = Field(
        None,
        description=(
            "Model residency override — see GenerateRequest.keep_alive. "
            "Supports '5m' / 30 / 0 (unload) / -1 (keep forever) / null."
        ),
    )
    format: str | dict[str, Any] | None = Field(
        None,
        description=(
            "Structured-output constraint. Accepts 'json' (free-form "
            "JSON) or a JSON Schema object. See GenerateRequest.format."
        ),
    )
    think: bool | None = Field(
        None,
        description=(
            "Expose the model's reasoning channel in the response "
            "(Ollama 2026). Default False for thinking-capable "
            "models (content stays clean); set True to receive the "
            "raw channel output for inspection / debugging."
        ),
    )
    system: str | None = Field(
        None,
        max_length=2_000_000,
        description=(
            "System prompt override for this chat. Inserted as the "
            "FIRST message with role=system, regardless of what the "
            "caller already included or what the Modelfile defined "
            "(Ollama-parity P1-1)."
        ),
    )
    agent_loop: bool | None = Field(
        None,
        description=(
            "Run the server-side agent loop (Phase 10 P1). When true, "
            "HFL dispatches tool calls itself and re-submits results "
            "to the model, returning only the final assistant turn "
            "alongside a ``tool_trace`` for replay. Only meaningful on "
            "non-streaming requests."
        ),
    )
    max_iterations: int | None = Field(
        None,
        ge=1,
        le=20,
        description=(
            "Hard cap on agent-loop iterations (default 6). Ignored unless ``agent_loop`` is true."
        ),
    )

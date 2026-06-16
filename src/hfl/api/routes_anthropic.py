# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatible with the Anthropic Messages API.

Allows tools using the Anthropic SDK (e.g., Claude Code with
ANTHROPIC_BASE_URL) to use HFL as a backend.

Implemented endpoints:
  POST /v1/messages  - Create a message (streaming and non-streaming)
"""

import contextlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

from hfl.api.chat_core import resolve_chat_output
from hfl.api.converters import anthropic_to_generation_config
from hfl.api.errors import service_unavailable
from hfl.api.helpers import (
    prepare_stream_response,
    queue_response_from_error,
    run_dispatched,
)
from hfl.api.schemas.anthropic import AnthropicMessagesRequest
from hfl.engine.base import ChatMessage
from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

if TYPE_CHECKING:
    from hfl.api.state import ServerState
    from hfl.engine.base import GenerationConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Anthropic API"])


# --- Helpers ---


def _get_state() -> "ServerState":
    """Get the singleton server state."""
    from hfl.api.state import get_state

    return get_state()


async def _ensure_model_loaded(model_name: str) -> None:
    """Load the model if it is not already in memory (thread-safe)."""
    from hfl.api.model_loader import load_llm

    await load_llm(model_name)


def _anthropic_tools_to_payload(req: AnthropicMessagesRequest) -> list[dict] | None:
    """Translate Anthropic ``tools`` into the engine's OpenAI-function
    payload, honouring ``tool_choice`` narrowing.

    Anthropic tool defs (``{name, description, input_schema}``) map to the
    ``{"type": "function", "function": {name, description, parameters}}``
    shape the engines' tool-aware chat templates and the per-family parser
    already expect — the same payload the OpenAI route forwards.
    """
    if not req.tools:
        return None
    choice = req.tool_choice or {}
    if isinstance(choice, dict) and choice.get("type") == "none":
        # Hard opt-out (Anthropic ``tool_choice: {"type": "none"}``): don't
        # advertise tools to the template. The per-family marker parser is
        # also suppressed via ``tools_disabled`` in the handler.
        return None
    payload: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in req.tools
    ]
    if isinstance(choice, dict) and choice.get("type") == "tool":
        name = choice.get("name")
        if name:
            narrowed = [t for t in payload if t["function"]["name"] == name]
            if narrowed:
                return narrowed
    return payload


def _tool_result_text(content: Any) -> str:
    """Flatten an Anthropic ``tool_result`` block's content to plain text.

    The content may be a bare string or a list of nested blocks
    (``[{"type": "text", "text": ...}]``).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _request_to_messages(req: AnthropicMessagesRequest) -> list[ChatMessage]:
    """Convert an Anthropic request into the internal ChatMessage list.

    Handles system-prompt injection, plain text, and the tool content
    blocks Claude Code relies on:

    - assistant ``tool_use`` blocks -> a ``tool_calls`` turn;
    - user ``tool_result`` blocks   -> ``role=tool`` messages keyed by the
      originating ``tool_use_id`` (function name recovered from the matching
      ``tool_use`` so the engine's tool template can label the result).
    """
    messages: list[ChatMessage] = []

    system_text = req.get_system_text()
    if system_text:
        messages.append(ChatMessage(role="system", content=system_text))

    id_to_name: dict[str, str] = {}

    for msg in req.messages:
        if isinstance(msg.content, str):
            messages.append(ChatMessage(role=msg.role, content=msg.content))
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[Any] = []
        for block in msg.content:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "tool_use":
                if block.id and block.name:
                    id_to_name[block.id] = block.name
                tool_calls.append(
                    {"function": {"name": block.name or "", "arguments": block.input or {}}}
                )
            elif block.type == "tool_result":
                tool_results.append(block)

        text = "".join(text_parts)
        if msg.role == "assistant":
            messages.append(
                ChatMessage(role="assistant", content=text, tool_calls=tool_calls or None)
            )
        elif text or not tool_results:
            # A user turn that is purely tool_result blocks adds no text
            # message; otherwise carry the user's text.
            messages.append(ChatMessage(role=msg.role, content=text))

        for tr in tool_results:
            messages.append(
                ChatMessage(
                    role="tool",
                    content=_tool_result_text(tr.content),
                    tool_call_id=tr.tool_use_id,
                    name=id_to_name.get(tr.tool_use_id or ""),
                )
            )

    return messages


def _to_anthropic_tool_use(canonical: list[dict]) -> list[dict]:
    """Map canonical ``{"function": {"name", "arguments": dict}}`` calls to
    Anthropic ``tool_use`` content blocks (``arguments`` -> ``input`` dict)."""
    blocks: list[dict] = []
    for call in canonical:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        args = fn.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name", ""),
                "input": args,
            }
        )
    return blocks


def _sse(event: str, data: dict) -> str:
    """Serialise one Anthropic SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stop_reason_to_anthropic(stop_reason: str) -> str:
    """Map internal stop reasons to Anthropic format."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
    }
    return mapping.get(stop_reason, "end_turn")


# --- Endpoints ---


@router.post(
    "/v1/messages",
    response_model=None,
    tags=["Anthropic"],
    summary="Create a message",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def create_message(
    req: AnthropicMessagesRequest,
) -> dict[str, Any] | StreamingResponse | Response:
    """Anthropic-compatible ``POST /v1/messages``.

    Accepts a Messages-API request (optionally with a provider-prefixed
    model like ``hfl/qwen-coder``), strips the prefix, loads the model,
    then streams SSE events (``req.stream=True``) or returns the full
    assistant message. Returns a structured 400 when the input exceeds
    the model's context window.
    """
    model_name = req.resolve_model_name()
    await _ensure_model_loaded(model_name)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{model_name}' failed to load", path="/v1/messages")

    messages = _request_to_messages(req)
    tools = _anthropic_tools_to_payload(req)
    # ``tool_choice: {"type": "none"}`` is a hard opt-out: tools are not
    # advertised and the marker parser is suppressed, so the reply can never
    # contain a tool_use block (mirrors the OpenAI route).
    tools_disabled = isinstance(req.tool_choice, dict) and req.tool_choice.get("type") == "none"
    gen_config = anthropic_to_generation_config(req)

    if req.stream:
        return await prepare_stream_response(
            lambda slot: _stream_messages(
                model_name, messages, gen_config, tools, slot, echo_model=req.model
            ),
            media_type="text/event-stream",
            path="/v1/messages",
        )

    # Non-streaming, serialized by the inference dispatcher (spec §5.3).
    try:
        result = await run_dispatched(
            state.engine.chat,
            messages,
            gen_config,
            tools=tools,
            operation="anthropic_messages",
        )
    except (QueueFullError, QueueTimeoutError) as exc:
        return queue_response_from_error(exc, path="/v1/messages")
    except ValueError as e:
        # llama_cpp raises ValueError when tokens exceed context
        # window. Branch on the exception text but never forward it
        # to the client (CodeQL ``py/stack-trace-exposure`` — the raw
        # repr may include paths / class names / line numbers).
        error_msg = str(e)
        if "exceed context window" in error_msg or "context" in error_msg.lower():
            logger.info("/v1/messages rejected: context window exceeded")
            return Response(
                content=json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Prompt exceeds the model's context window.",
                        },
                    }
                ),
                status_code=400,
                media_type="application/json",
            )
        raise

    # Shared decision (see chat_core): prefer the engine's structured tool
    # calls, else parse markers out of the text. A tool call flips the turn
    # to a ``tool_use`` content block + ``stop_reason: tool_use`` so the
    # Anthropic SDK / Claude Code agent loop dispatches the tool.
    resolved = resolve_chat_output(
        result.text,
        model_name,
        tools,
        getattr(result, "tool_calls", None),
        tools_disabled=tools_disabled,
    )
    if resolved.has_tool_calls:
        content_blocks: list[dict] = []
        if resolved.content:
            content_blocks.append({"type": "text", "text": resolved.content})
        content_blocks.extend(_to_anthropic_tool_use(resolved.tool_calls))
        stop_reason = "tool_use"
    else:
        content_blocks = [{"type": "text", "text": resolved.content}]
        stop_reason = _stop_reason_to_anthropic(result.stop_reason)

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": req.model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": result.tokens_prompt,
            "output_tokens": result.tokens_generated,
        },
    }


async def _stream_messages(
    model: str,
    messages: list[ChatMessage],
    config: "GenerationConfig",
    tools: list[dict] | None = None,
    slot_cm: Any | None = None,
    *,
    echo_model: str | None = None,
) -> AsyncIterator[str]:
    """Generate Anthropic-compatible SSE streaming responses.

    Plain turns stream text token-by-token. Tool-aware turns (``tools``
    declared) buffer the whole reply — like the OpenAI route — so a
    tool-call marker is parsed into structured ``tool_use`` content blocks
    (``stop_reason: tool_use``) instead of leaking as text.

    ``slot_cm`` is the dispatcher slot held for the entire stream
    (spec §5.3); it is released in ``finally``.
    """
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        err = json.dumps(
            {"type": "error", "error": {"type": "server_error", "message": "Model not loaded"}}
        )
        yield f"event: error\ndata: {err}\n\n"
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)
        return

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    tool_aware = bool(tools)

    # event: message_start
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            # API-14: echo the client's model string verbatim (incl. any
            # provider prefix), matching the non-streaming response.
            "model": echo_model or model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield _sse("message_start", message_start)
    yield _sse("ping", {"type": "ping"})

    # Plain turns open a text block up front and stream into it; tool-aware
    # turns emit no blocks until the buffered reply is parsed in format_done.
    if not tool_aware:
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )

    output_tokens = 0
    accumulated: list[str] = []

    def format_delta(token: str) -> str:
        nonlocal output_tokens
        # Count every token (incl. buffered tool-aware ones) so the
        # usage.output_tokens and the max_tokens stop-reason detection
        # (_plain_stop_reason) are live on the tool-aware path too (API-10).
        output_tokens += 1
        if tool_aware:
            accumulated.append(token)
            return ""
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": token},
        }
        return _sse("content_block_delta", delta)

    def _text_block(index: int, text: str) -> str:
        return (
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            + _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
            + _sse("content_block_stop", {"type": "content_block_stop", "index": index})
        )

    def _plain_stop_reason() -> str:
        # API-10: report max_tokens when the cap was hit, else end_turn (the
        # streaming path only exposes the emitted-token count).
        return (
            "max_tokens"
            if (config.max_tokens and output_tokens >= config.max_tokens)
            else "end_turn"
        )

    def format_done() -> str:
        if not tool_aware:
            return (
                _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                + _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": _plain_stop_reason(), "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens},
                    },
                )
                + _sse("message_stop", {"type": "message_stop"})
            )

        # Tool-aware: parse the buffered reply into structured blocks.
        raw = "".join(accumulated)
        resolved = resolve_chat_output(raw, model, tools, None)
        events: list[str] = []
        index = 0
        if resolved.has_tool_calls:
            if resolved.content:
                events.append(_text_block(index, resolved.content))
                index += 1
            for block in _to_anthropic_tool_use(resolved.tool_calls):
                events.append(
                    _sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": block["id"],
                                "name": block["name"],
                                "input": {},
                            },
                        },
                    )
                    + _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(block["input"], ensure_ascii=False),
                            },
                        },
                    )
                    + _sse("content_block_stop", {"type": "content_block_stop", "index": index})
                )
                index += 1
            stop_reason = "tool_use"
        else:
            events.append(_text_block(0, resolved.content))
            stop_reason = _plain_stop_reason()

        events.append(
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": len(raw.split())},
                },
            )
        )
        events.append(_sse("message_stop", {"type": "message_stop"}))
        return "".join(events)

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=state.engine.chat_stream(messages, config, tools),
            format_item=format_delta,
            format_done=format_done,
        ):
            yield chunk
    except Exception:
        # Never emit ``str(exc)`` on a stream — the exception repr
        # can leak internal paths / library names (CodeQL
        # ``py/stack-trace-exposure``). Full traceback lands in the
        # server log via ``logger.exception``.
        logger.exception("/v1/messages stream failed")
        error_payload = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "message": "Internal server error during streaming.",
                },
            }
        )
        yield f"event: error\ndata: {error_payload}\n\n"
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)

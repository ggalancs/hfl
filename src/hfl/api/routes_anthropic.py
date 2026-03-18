# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatible with the Anthropic Messages API.

Allows tools using the Anthropic SDK (e.g., Claude Code with
ANTHROPIC_BASE_URL) to use HFL as a backend.

Implemented endpoints:
  POST /v1/messages  - Create a message (streaming and non-streaming)
"""

import json
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

from hfl.api.converters import anthropic_to_generation_config
from hfl.api.errors import service_unavailable
from hfl.api.helpers import run_with_timeout
from hfl.api.schemas.anthropic import AnthropicMessagesRequest
from hfl.engine.base import ChatMessage

if TYPE_CHECKING:
    from hfl.api.state import ServerState
    from hfl.engine.base import GenerationConfig

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


def _request_to_messages(req: AnthropicMessagesRequest) -> list[ChatMessage]:
    """Convert Anthropic request to internal ChatMessage list.

    Handles system prompt injection and content block extraction.
    """
    messages: list[ChatMessage] = []

    # Inject system prompt as first message if present
    system_text = req.get_system_text()
    if system_text:
        messages.append(ChatMessage(role="system", content=system_text))

    for msg in req.messages:
        messages.append(ChatMessage(role=msg.role, content=msg.get_text()))

    return messages


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
    model_name = req.resolve_model_name()
    await _ensure_model_loaded(model_name)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{model_name}' failed to load")

    messages = _request_to_messages(req)
    gen_config = anthropic_to_generation_config(req)

    if req.stream:
        return StreamingResponse(
            _stream_messages(model_name, messages, gen_config),
            media_type="text/event-stream",
        )

    # Non-streaming
    try:
        result = await run_with_timeout(
            state.engine.chat, messages, gen_config, operation="anthropic_messages"
        )
    except ValueError as e:
        # llama_cpp raises ValueError when tokens exceed context window
        error_msg = str(e)
        if "exceed context window" in error_msg or "context" in error_msg.lower():
            return Response(
                content=json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": error_msg,
                        },
                    }
                ),
                status_code=400,
                media_type="application/json",
            )
        raise

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": result.text}],
        "model": req.model,
        "stop_reason": _stop_reason_to_anthropic(result.stop_reason),
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
) -> AsyncIterator[str]:
    """Generate Anthropic-compatible SSE streaming responses."""
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        err = json.dumps(
            {"type": "error", "error": {"type": "server_error", "message": "Model not loaded"}}
        )
        yield f"event: error\ndata: {err}\n\n"
        return

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # event: message_start
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # event: content_block_start
    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

    # event: ping
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    # Stream content deltas
    output_tokens = 0

    def format_delta(token: str) -> str:
        nonlocal output_tokens
        output_tokens += 1
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": token},
        }
        return f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"

    def format_done() -> str:
        # content_block_stop + message_delta + message_stop
        block_stop = {"type": "content_block_stop", "index": 0}
        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
        msg_stop = {"type": "message_stop"}
        return (
            f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"
            f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
            f"event: message_stop\ndata: {json.dumps(msg_stop)}\n\n"
        )

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=state.engine.chat_stream(messages, config),
            format_item=format_delta,
            format_done=format_done,
        ):
            yield chunk
    except Exception as e:
        err = {
            "type": "error",
            "error": {"type": "server_error", "message": str(e)},
        }
        yield f"event: error\ndata: {json.dumps(err)}\n\n"

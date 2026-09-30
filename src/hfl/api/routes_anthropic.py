# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatible with the Anthropic Messages API.

Allows tools using the Anthropic SDK (e.g., Claude Code with
ANTHROPIC_BASE_URL) to use HFL as a backend.

Implemented endpoints:
  POST /v1/messages  - Create a message (streaming and non-streaming)
"""

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
from hfl.api.modelfile_defaults import apply_to_chat, explicit_fields
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


# Anthropic request fields -> the Modelfile parameters they set.
ANTHROPIC_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "num_predict",
    "stop_sequences": "stop",
}


def _prepared(
    req: AnthropicMessagesRequest, manifest: Any
) -> tuple[list[ChatMessage], "GenerationConfig"]:
    """The engine messages and config for ``req``, with the created model's
    Modelfile applied — shared by /v1/messages and its count_tokens, which
    must count what the model would be fed."""
    config = anthropic_to_generation_config(req)
    messages = apply_to_chat(
        manifest, _request_to_messages(req), config, explicit_fields(req, ANTHROPIC_OPTIONS)
    )
    return messages, config


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
        images: list[bytes] = []
        for index, block in enumerate(msg.content):
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "image":
                from hfl.api.vision import decode_anthropic_image

                source = (block.model_extra or {}).get("source")
                images.append(decode_anthropic_image(source, f"content[{index}]"))
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
        elif text or images or not tool_results:
            # A user turn that is purely tool_result blocks adds no text
            # message; otherwise carry the user's text, and its images.
            messages.append(ChatMessage(role=msg.role, content=text, images=images or None))

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


def _thinking_block(text: str) -> dict:
    """A ``thinking`` content block. Anthropic signs its own so they can be
    sent back; a local model's reasoning has nothing to prove, and what a
    client sends back is ignored (``_request_to_messages``)."""
    return {"type": "thinking", "thinking": text, "signature": ""}


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
    "/v1/messages/count_tokens",
    response_model=None,
    tags=["Anthropic"],
    summary="Count a message's input tokens",
    responses={
        404: {"description": "Model not found"},
        501: {"description": "The model's backend cannot count exactly"},
    },
)
async def count_message_tokens(req: AnthropicMessagesRequest) -> dict[str, Any] | Response:
    """Anthropic-compatible ``POST /v1/messages/count_tokens``: the tokens
    ``/v1/messages`` would feed the model for this request — its template,
    tools and thinking switch applied — with nothing generated.
    """
    model_name = req.resolve_model_name()
    await _ensure_model_loaded(model_name)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(
            f"Model '{model_name}' failed to load", path="/v1/messages/count_tokens"
        )
    tools = _anthropic_tools_to_payload(req)  # none for tool_choice "none", as there
    messages, gen_config = _prepared(req, state.current_model)
    try:
        count = await run_dispatched(
            state.engine.count_prompt_tokens,
            messages,
            gen_config,
            tools,
            operation="count_tokens",
        )
    except (QueueFullError, QueueTimeoutError) as exc:
        return queue_response_from_error(exc, path="/v1/messages/count_tokens")
    except NotImplementedError:
        error = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "This model's backend cannot count a prompt's tokens exactly.",
            },
        }
        return Response(content=json.dumps(error), status_code=501, media_type="application/json")
    return {"input_tokens": int(count)}


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

    messages, gen_config = _prepared(req, state.current_model)
    tools = _anthropic_tools_to_payload(req)
    # ``tool_choice: {"type": "none"}`` is a hard opt-out: tools are not
    # advertised and the marker parser is suppressed, so the reply can never
    # contain a tool_use block (mirrors the OpenAI route).
    tools_disabled = isinstance(req.tool_choice, dict) and req.tool_choice.get("type") == "none"

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
    content_blocks: list[dict] = []
    if resolved.reasoning and gen_config.expose_reasoning:
        content_blocks.append(_thinking_block(resolved.reasoning))
    if resolved.has_tool_calls:
        if resolved.content:
            content_blocks.append({"type": "text", "text": resolved.content})
        content_blocks.extend(_to_anthropic_tool_use(resolved.tool_calls))
        stop_reason = "tool_use"
    else:
        content_blocks.append({"type": "text", "text": resolved.content})
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


class _AnthropicStream:
    """Anthropic's SSE events for a streamed turn (``hfl.api.chat_stream``).

    Plain turns stream into blocks opened as their text arrives — the
    reasoning's ``thinking`` block (only when the client enabled thinking;
    otherwise dropped, never sent as text), then the answer's ``text``
    block. Tool-aware turns buffer the reply and parse it into structured
    ``tool_use`` blocks (``stop_reason: tool_use``) instead of leaking text.
    """

    log_label = "/v1/messages"

    def __init__(self, model: str, echo_model: str | None) -> None:
        self.model, self.echo_model = model, echo_model
        self.index, self.kind, self.had_text = -1, "", False  # the open block

    def not_loaded(self) -> str:
        err = {"type": "error", "error": {"type": "server_error", "message": "Model not loaded"}}
        return f"event: error\ndata: {json.dumps(err)}\n\n"

    def failed(self) -> str:
        err = {
            "type": "error",
            "error": {"type": "server_error", "message": "Internal server error during streaming."},
        }
        return f"event: error\ndata: {json.dumps(err)}\n\n"

    def start(self, turn: Any) -> str:
        message: dict[str, Any] = {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [],
            # API-14: the client's model string verbatim (any provider prefix).
            "model": self.echo_model or self.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return _sse("message_start", {"type": "message_start", "message": message}) + _sse(
            "ping", {"type": "ping"}
        )

    def _open(self, kind: str) -> str:
        out = self._close()
        self.index += 1
        self.kind = kind
        self.had_text = self.had_text or kind == "text"
        empty = _thinking_block("") if kind == "thinking" else {"type": "text", "text": ""}
        start = {"type": "content_block_start", "index": self.index, "content_block": empty}
        return out + _sse("content_block_start", start)

    def _close(self) -> str:
        if not self.kind:
            return ""
        self.kind = ""
        return _sse("content_block_stop", {"type": "content_block_stop", "index": self.index})

    def _write(self, kind: str, text: str) -> str:
        if not text:
            return ""
        out = "" if self.kind == kind else self._open(kind)
        if kind == "thinking":
            delta = {"type": "thinking_delta", "thinking": text}
        else:
            delta = {"type": "text_delta", "text": text}
        body = {"type": "content_block_delta", "index": self.index, "delta": delta}
        return out + _sse("content_block_delta", body)

    def _split(self, turn: Any, answer: str, reasoning: str) -> str:
        shown = reasoning if turn.config.expose_reasoning else ""
        return self._write("thinking", shown) + self._write("text", answer)

    def token(self, turn: Any, token: str) -> str:
        # Every token counts (buffered ones too): usage.output_tokens and the
        # max_tokens stop reason are live on the tool-aware path (API-10).
        return "" if turn.tool_aware else self._split(turn, *turn.splitter.feed(token))

    def _usage(self, turn: Any) -> dict[str, int]:
        # The engine's own counts when it keeps them, else the chunk counter
        # — never a whitespace word count (BPE sub-words diverge from it).
        prompt_n, generated = turn.counts()
        usage = {"output_tokens": generated if generated is not None else turn.emitted}
        if prompt_n is not None:
            usage["input_tokens"] = prompt_n
        return usage

    def _end(self, turn: Any, stop_reason: str) -> str:
        delta = {"stop_reason": stop_reason, "stop_sequence": None}
        body = {"type": "message_delta", "delta": delta, "usage": self._usage(turn)}
        return _sse("message_delta", body) + _sse("message_stop", {"type": "message_stop"})

    def _plain_stop(self, turn: Any) -> str:
        # API-10: max_tokens when the cap was hit, else end_turn.
        cap = turn.config.max_tokens
        return "max_tokens" if (cap and self._usage(turn)["output_tokens"] >= cap) else "end_turn"

    def done(self, turn: Any) -> str:
        if not turn.tool_aware:
            rest = self._split(turn, *turn.splitter.flush())
            if not self.had_text:
                rest += self._open("text")  # every reply has a text block
            return rest + self._close() + self._end(turn, self._plain_stop(turn))
        resolved = resolve_chat_output(turn.text(), self.model, turn.tools, None)
        events: list[str] = []
        index = 0
        if resolved.reasoning and turn.config.expose_reasoning:
            events.append(_block(index, _thinking_block(""), "thinking_delta", resolved.reasoning))
            index += 1
        text = {"type": "text", "text": ""}
        if not resolved.has_tool_calls:
            events.append(_block(index, text, "text_delta", resolved.content))
            return "".join(events) + self._end(turn, self._plain_stop(turn))
        if resolved.content:
            events.append(_block(index, text, "text_delta", resolved.content))
            index += 1
        for use in _to_anthropic_tool_use(resolved.tool_calls):
            head = {"type": "tool_use", "id": use["id"], "name": use["name"], "input": {}}
            payload = json.dumps(use["input"], ensure_ascii=False)
            events.append(_block(index, head, "input_json_delta", payload))
            index += 1
        return "".join(events) + self._end(turn, "tool_use")


def _block(index: int, head: dict, delta_type: str, value: str) -> str:
    """One whole content block: start, its one delta, stop."""
    key = {"thinking_delta": "thinking", "text_delta": "text"}.get(delta_type, "partial_json")
    start = {"type": "content_block_start", "index": index, "content_block": head}
    delta = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": delta_type, key: value},
    }
    return (
        _sse("content_block_start", start)
        + _sse("content_block_delta", delta)
        + _sse("content_block_stop", {"type": "content_block_stop", "index": index})
    )


def _stream_messages(
    model: str,
    messages: list[ChatMessage],
    config: "GenerationConfig",
    tools: list[dict] | None = None,
    slot_cm: Any | None = None,
    *,
    echo_model: str | None = None,
) -> AsyncIterator[str]:
    """Anthropic-compatible SSE; ``slot_cm`` (the dispatcher slot, spec
    §5.3) is released whatever happens."""
    from hfl.api.chat_stream import run_chat_stream

    return run_chat_stream(
        _AnthropicStream(model, echo_model),
        engine=_get_state().engine,
        model=model,
        messages=messages,
        config=config,
        tools=tools,
        slot_cm=slot_cm,
    )

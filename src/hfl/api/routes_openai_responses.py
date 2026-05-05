# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""OpenAI Responses API (``POST /v1/responses``).

OpenAI introduced this endpoint in 2025 as a higher-level wrapper
around chat completions: it bundles tools, structured output, and
reasoning summaries into a single shape, and is the path the new
``client.responses.create(...)`` SDK call hits.

HFL implements it on top of the existing chat-completion machinery —
no new engine path. The mapping is:

  Responses request                    Chat-completion equivalent
  -------------------------------      ----------------------------
  ``input``  (str | list)              ``messages`` (with role guess)
  ``instructions``                     leading ``system`` message
  ``tools``                            ``tools`` (passed through)
  ``reasoning.effort``                 ``think`` (off/low/medium/high)
  ``response_format``                  ``response_format``
  ``stream`` = false                   non-stream → render output[]
  ``stream`` = true                    SSE with ``response.*`` events

The endpoint is **not stateful** (matches Ollama's own /v1/responses
limitation) — every request is self-contained, the server does not
persist a ``response_id`` chain.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hfl.api.helpers import run_dispatched
from hfl.engine.base import ChatMessage, GenerationConfig

if TYPE_CHECKING:
    from hfl.api.state import ServerState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OpenAI"])


# --- Schemas ----------------------------------------------------------------


class ResponsesRequest(BaseModel):
    """Minimal Responses-API request envelope.

    Permissive on purpose: OpenAI keeps adding optional fields, and we
    forward the ones we recognise while ignoring the rest. Strict
    validation lives at the engine layer.
    """

    model: str = Field(..., min_length=1, max_length=256)
    # ``input`` accepts either a single string ("hello") or an array of
    # message-shaped dicts ({"role": "user", "content": "hello"} —
    # which can also have ``content`` as a list of typed parts).
    input: str | list[dict[str, Any]] = Field(...)
    instructions: str | None = Field(default=None, max_length=200_000)
    tools: list[dict[str, Any]] | None = Field(default=None)
    reasoning: dict[str, Any] | None = Field(default=None)
    response_format: dict[str, Any] | None = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=128_000)
    stream: bool = Field(default=False)
    metadata: dict[str, Any] | None = Field(default=None)


# --- Helpers ----------------------------------------------------------------


def _get_state() -> "ServerState":
    from hfl.api.state import get_state

    return get_state()


def _input_to_messages(
    input_value: str | list[dict[str, Any]],
    instructions: str | None,
) -> list[ChatMessage]:
    """Translate the Responses ``input`` field into ``ChatMessage[]``.

    Rules:
    - A bare string becomes a single ``user`` message.
    - A list of dicts is forwarded with ``role`` defaulting to ``user``
      and ``content`` flattened to text. List-of-parts content (the
      OpenAI multimodal shape) is reduced by concatenating the
      ``input_text`` / ``text`` fields; image parts are dropped here
      because /v1/responses' image support is not in scope yet.
    - ``instructions`` is prepended as a ``system`` message when
      present.
    """
    messages: list[ChatMessage] = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))

    if isinstance(input_value, str):
        messages.append(ChatMessage(role="user", content=input_value))
        return messages

    for raw in input_value:
        role = str(raw.get("role") or "user")
        content = raw.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                # OpenAI uses ``input_text``/``output_text``; we accept
                # both plus the legacy ``text`` shape.
                ptype = part.get("type")
                if ptype in {"input_text", "output_text", "text"}:
                    txt = part.get("text") or ""
                    if isinstance(txt, str):
                        chunks.append(txt)
            text = "".join(chunks)
        else:
            text = ""
        messages.append(ChatMessage(role=role, content=text))
    return messages


def _resolve_thinking(reasoning: dict[str, Any] | None) -> str | None:
    """Map ``reasoning.effort`` to the engine's ``thinking_level``.

    OpenAI accepts ``"low"``/``"medium"``/``"high"``. We keep the
    same vocabulary so the existing ``GenerationConfig.thinking_level``
    grammar applies unchanged.
    """
    if not reasoning:
        return None
    effort = reasoning.get("effort")
    if isinstance(effort, str) and effort.lower() in {"low", "medium", "high"}:
        return effort.lower()
    return None


def _build_gen_config(req: ResponsesRequest) -> GenerationConfig:
    cfg = GenerationConfig(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_output_tokens or 0,
    )
    level = _resolve_thinking(req.reasoning)
    if level is not None:
        cfg.thinking_level = level
        cfg.expose_reasoning = level != "off"
    if req.response_format is not None:
        from hfl.api.structured_outputs import normalize_openai_response_format

        cfg.response_format = normalize_openai_response_format(req.response_format)
    return cfg


def _render_response(
    *,
    response_id: str,
    model: str,
    text: str,
    tokens_input: int,
    tokens_output: int,
    tool_calls: list[dict[str, Any]] | None,
    reasoning_text: str | None,
) -> dict[str, Any]:
    """Build the canonical OpenAI Responses output envelope.

    The shape mirrors what ``client.responses.create(...)`` returns —
    ``output`` is a heterogeneous list whose order matters: reasoning
    summaries first when present, then the assistant message, then any
    ``function_call`` items.
    """
    output: list[dict[str, Any]] = []

    if reasoning_text:
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
            }
        )

    if text:
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": text, "annotations": []},
                ],
            }
        )

    if tool_calls:
        for call in tool_calls:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "name": fn.get("name") or "",
                    "arguments": args,
                    "status": "completed",
                }
            )

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": tokens_input,
            "output_tokens": tokens_output,
            "total_tokens": tokens_input + tokens_output,
        },
        "metadata": None,
    }


async def _stream_response(
    response_id: str,
    model: str,
    messages: list[ChatMessage],
    cfg: GenerationConfig,
    tools: list[dict[str, Any]] | None,
) -> AsyncIterator[str]:
    """SSE stream that re-emits chat tokens as Responses events.

    Event grammar (subset of OpenAI's spec — what the SDK actually
    keys on for non-tool replies):

      response.created
      response.output_text.delta  (one per token chunk)
      response.completed
    """
    state = _get_state()
    if state.engine is None:
        err = {
            "type": "response.failed",
            "error": {"code": "engine_unavailable", "message": "Model not loaded"},
        }
        yield f"data: {json.dumps(err)}\n\n"
        return

    created = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model,
        },
    }
    yield f"data: {json.dumps(created)}\n\n"

    accumulated: list[str] = []

    if tools is not None:
        try:
            sync_iter = state.engine.chat_stream(messages, cfg, tools=tools)
        except TypeError:
            sync_iter = state.engine.chat_stream(messages, cfg)
    else:
        sync_iter = state.engine.chat_stream(messages, cfg)

    import asyncio

    def _next(it: Any) -> tuple[bool, str]:
        try:
            return True, next(it)
        except StopIteration:
            return False, ""

    while True:
        ok, token = await asyncio.to_thread(_next, sync_iter)
        if not ok:
            break
        accumulated.append(token)
        evt = {"type": "response.output_text.delta", "delta": token}
        yield f"data: {json.dumps(evt)}\n\n"

    text = "".join(accumulated)
    completed = {
        "type": "response.completed",
        "response": _render_response(
            response_id=response_id,
            model=model,
            text=text,
            tokens_input=0,
            tokens_output=len(accumulated),
            tool_calls=None,
            reasoning_text=None,
        ),
    }
    yield f"data: {json.dumps(completed)}\n\n"
    yield "data: [DONE]\n\n"


# --- Endpoint ---------------------------------------------------------------


@router.post(
    "/v1/responses",
    response_model=None,
    tags=["OpenAI"],
    summary="Create response (OpenAI Responses API)",
    responses={
        400: {"description": "Invalid request"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def responses(req: ResponsesRequest) -> dict[str, Any] | StreamingResponse:
    """OpenAI-compatible ``POST /v1/responses``.

    Bundles input, instructions, tools, reasoning effort and structured
    output into a single endpoint. Internally maps to chat-completion
    semantics; not stateful.
    """
    from hfl.api.model_loader import load_llm

    await load_llm(req.model)

    state = _get_state()
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = _input_to_messages(req.input, req.instructions)
    cfg = _build_gen_config(req)

    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    if req.stream:
        return StreamingResponse(
            _stream_response(response_id, req.model, messages, cfg, req.tools),
            media_type="text/event-stream",
        )

    if req.tools is not None:
        try:
            result = await run_dispatched(
                state.engine.chat,
                messages,
                cfg,
                tools=req.tools,
            )
        except TypeError:
            result = await run_dispatched(state.engine.chat, messages, cfg)
    else:
        result = await run_dispatched(state.engine.chat, messages, cfg)

    raw_text = getattr(result, "text", "") or ""
    tokens_input = int(getattr(result, "tokens_prompt", 0) or 0)
    tokens_output = int(getattr(result, "tokens_generated", 0) or 0)
    reasoning_text = getattr(result, "reasoning_text", None)
    engine_tool_calls = getattr(result, "tool_calls", None)

    # Run the tool-call parser when the engine did not already surface
    # structured tool_calls. Mirrors the logic in routes_native:_finalize.
    tool_calls: list[dict[str, Any]] | None = None
    cleaned_text = raw_text
    if isinstance(engine_tool_calls, list) and engine_tool_calls:
        tool_calls = list(engine_tool_calls)
        cleaned_text = ""
    else:
        try:
            from hfl.api.tool_parsers import dispatch as parse_tool_calls

            cleaned, parsed = parse_tool_calls(raw_text, req.model, req.tools)
            if parsed:
                tool_calls = parsed
                cleaned_text = ""
            else:
                cleaned_text = cleaned
        except Exception:  # pragma: no cover — parser bugs must not 500
            logger.exception("tool-call parser failed for /v1/responses")

    return _render_response(
        response_id=response_id,
        model=req.model,
        text=cleaned_text,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tool_calls=tool_calls,
        reasoning_text=reasoning_text,
    )

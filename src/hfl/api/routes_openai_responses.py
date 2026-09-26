# SPDX-License-Identifier: Apache-2.0
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

import contextlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from hfl.api.chat_core import resolve_chat_output
from hfl.api.helpers import prepare_stream_response, run_dispatched
from hfl.api.thinking import ThinkingSplitter
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
    # Continue a stored response's conversation (``response_store``);
    # ``store: false`` keeps this one from being stored.
    previous_response_id: str | None = Field(default=None, max_length=128)
    store: bool = Field(default=True)


# --- Helpers ----------------------------------------------------------------


def _get_state() -> "ServerState":
    from hfl.api.state import get_state

    return get_state()


def _input_to_messages(
    input_value: str | list[dict[str, Any]],
    instructions: str | None,
    known_calls: dict[str, str] | None = None,
) -> list[ChatMessage]:
    """Translate the Responses ``input`` field into ``ChatMessage[]``.

    Rules:
    - A bare string becomes a single ``user`` message.
    - A list of dicts is forwarded with ``role`` defaulting to ``user``
      and ``content`` flattened to text. List-of-parts content (the
      OpenAI multimodal shape) is reduced by concatenating the
      ``input_text`` / ``text`` fields; image parts are dropped here
      because /v1/responses' image support is not in scope yet.
    - ``function_call`` / ``function_call_output`` items become an
      assistant turn with ``tool_calls`` and ``tool`` results; the
      ``developer`` role becomes ``system``; ``reasoning`` items are skipped.
    - ``instructions`` is prepended as a ``system`` message when
      present.
    - ``known_calls`` names the calls of earlier turns (``call_id`` →
      function), for results sent after ``previous_response_id``.
    """
    messages: list[ChatMessage] = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))

    if isinstance(input_value, str):
        messages.append(ChatMessage(role="user", content=input_value))
        return messages

    # Agents send the whole turn history as typed items (``store`` false):
    # earlier tool calls as ``function_call`` and their results as
    # ``function_call_output``. Consecutive calls belong to one assistant
    # turn; ``reasoning`` items carry nothing a local model can use.
    call_names: dict[str, str] = dict(known_calls or {})
    for raw in input_value:
        kind = raw.get("type")
        if kind == "reasoning":
            continue
        if kind == "function_call":
            call_id = str(raw.get("call_id") or "")
            name = str(raw.get("name") or "")
            call_names[call_id] = name
            call = {"id": call_id, "function": {"name": name, "arguments": _arguments(raw)}}
            last = messages[-1] if messages else None
            if last is not None and last.role == "assistant" and last.tool_calls:
                last.tool_calls.append(call)
            else:
                messages.append(ChatMessage(role="assistant", content="", tool_calls=[call]))
            continue
        if kind == "function_call_output":
            call_id = str(raw.get("call_id") or "")
            messages.append(
                ChatMessage(
                    role="tool",
                    content=_content_text(raw.get("output")),
                    tool_call_id=call_id,
                    name=call_names.get(call_id),
                )
            )
            continue
        role = str(raw.get("role") or "user")
        if role == "developer":
            role = "system"  # chat templates know no "developer" role
        messages.append(ChatMessage(role=role, content=_content_text(raw.get("content"))))
    return messages


def _assistant_turn(output: list[dict[str, Any]]) -> list[ChatMessage]:
    """A response's ``output`` items as the assistant turn a later request
    continues from: its text, and its calls with the ``call_id`` the client
    will answer with."""
    turn: list[ChatMessage] = []
    calls: list[dict[str, Any]] = []
    for item in output:
        if item.get("type") == "message":
            text = _content_text(item.get("content"))
            turn.append(ChatMessage(role="assistant", content=text))
        elif item.get("type") == "function_call":
            function = {"name": str(item.get("name") or ""), "arguments": _arguments(item)}
            calls.append({"id": item.get("call_id"), "function": function})
    if calls:
        turn.append(ChatMessage(role="assistant", content="", tool_calls=calls))
    return turn


def _call_names(history: list[ChatMessage]) -> dict[str, str]:
    return {
        str(call.get("id") or ""): str((call.get("function") or {}).get("name") or "")
        for message in history
        for call in (message.tool_calls or [])
    }


def _not_found(previous: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Previous response with id '{previous}' not found.",
                "type": "invalid_request_error",
                "param": "previous_response_id",
                "code": "previous_response_not_found",
            }
        },
    )


def _content_text(content: Any) -> str:
    """Text of a Responses ``content``: a string, or a list of typed parts
    (``input_text`` / ``output_text`` / legacy ``text``; images dropped)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in {"input_text", "output_text", "text"}:
            txt = part.get("text") or ""
            if isinstance(txt, str):
                chunks.append(txt)
    return "".join(chunks)


def _arguments(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _chat_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Responses tools in the chat-completions shape the engines expect.

    The Responses API defines function tools flat (``{"type": "function",
    "name", "parameters"}``); chat templates read ``tool["function"]``.
    Hosted tools (``web_search``, ``namespace``, ...) run on OpenAI's side
    and have no local meaning, so they are dropped.
    """
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        if isinstance(tool.get("function"), dict):
            out.append(tool)
            continue
        if not tool.get("name"):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return out or None


def _resolve_thinking(reasoning: dict[str, Any] | None) -> str | None:
    """Map ``reasoning.effort`` to the engine's ``thinking_level``.

    ``"low"``/``"medium"``/``"high"`` as they are; ``"none"`` turns
    reasoning off and ``"minimal"`` is the lowest level, as for chat
    completions' ``reasoning_effort`` (``none`` used to be ignored, so a
    thinking model could not be told to stop thinking).
    """
    if not reasoning:
        return None
    effort = reasoning.get("effort")
    if not isinstance(effort, str):
        return None
    level = {"none": "off", "minimal": "low"}.get(effort.lower(), effort.lower())
    return level if level in {"off", "low", "medium", "high"} else None


def _build_gen_config(req: ResponsesRequest) -> GenerationConfig:
    cfg = GenerationConfig(
        temperature=req.temperature,
        top_p=req.top_p,
        # API-11: default to a sane cap (matching the chat route's 2048) rather
        # than 0/unbounded when the client omits max_output_tokens.
        max_tokens=req.max_output_tokens or 2048,
    )
    level = _resolve_thinking(req.reasoning)
    if level is not None:
        cfg.thinking_level = level
        cfg.expose_reasoning = level != "off"
        cfg.reasoning = level
    if req.response_format is not None:
        from hfl.api.structured_outputs import normalize_openai_response_format

        cfg.response_format = normalize_openai_response_format(req.response_format)
    return cfg


def _message_item(item_id: str, text: str, status: str = "completed") -> dict[str, Any]:
    content = [{"type": "output_text", "text": text, "annotations": []}] if text else []
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": status,
        "content": content,
    }


def _function_call_items(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for call in tool_calls:
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name") or "",
                "arguments": args,
                "status": "completed",
            }
        )
    return items


def _envelope(
    response_id: str,
    model: str,
    output: list[dict[str, Any]],
    tokens_input: int,
    tokens_output: int,
) -> dict[str, Any]:
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
        output.append(_message_item(f"msg_{uuid.uuid4().hex[:24]}", text))
    if tool_calls:
        output.extend(_function_call_items(tool_calls))
    return _envelope(response_id, model, output, tokens_input, tokens_output)


async def _stream_response(
    response_id: str,
    model: str,
    messages: list[ChatMessage],
    cfg: GenerationConfig,
    tools: list[dict[str, Any]] | None,
    slot_cm: Any | None = None,
    *,
    extra: dict[str, Any] | None = None,
    on_done: Callable[[list[dict[str, Any]]], None] | None = None,
) -> AsyncIterator[str]:
    """SSE stream that re-emits chat tokens as Responses events.

    Event grammar (subset of OpenAI's spec — what the SDK actually
    keys on):

      response.created
      response.output_text.delta  (one per token chunk; suppressed when
                                   ``tools`` are declared so a raw
                                   ``<tool_call>`` marker never leaks as text)
      response.completed          (final envelope, incl. structured
                                   ``function_call`` items when tools fired)

    The engine is driven through ``stream_with_backpressure`` so the work
    runs on the dispatcher slot held in ``slot_cm`` for the whole stream —
    serialising against every other inference request, since the shared
    llama.cpp / transformers model is non-reentrant and concurrent use
    corrupts its KV cache — and the sync iterator is closed on teardown
    (client disconnect) instead of leaking its worker thread until GC
    (CON-3). ``slot_cm`` is released in ``finally`` regardless of outcome.
    """
    from hfl.api.streaming import stream_with_backpressure

    try:
        state = _get_state()
        if state.engine is None:
            err = {
                "type": "response.failed",
                "error": {"code": "engine_unavailable", "message": "Model not loaded"},
            }
            yield f"data: {json.dumps(err)}\n\n"
            return

        seq = iter(range(1_000_000))

        def event(kind: str, **fields: Any) -> str:
            payload = {"type": kind, "sequence_number": next(seq), **fields}
            return f"data: {json.dumps(payload)}\n\n"

        in_progress = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model,
            "output": [],
        }
        yield event("response.created", response=in_progress)
        yield event("response.in_progress", response=in_progress)

        tool_aware = bool(tools)
        accumulated: list[str] = []
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        rs_id = f"rs_{uuid.uuid4().hex[:24]}"
        # Reasoning goes in its own ``reasoning`` item, before the message,
        # never as answer text; items open as their text arrives, and each
        # takes the next output_index.
        splitter = ThinkingSplitter()
        show_reasoning = cfg.reasoning != "off"
        items: dict[str, Any] = {"next": 0, "open": "", "reasoning": [], "answer": []}
        index_of: dict[str, int] = {}
        output: list[dict[str, Any]] = []

        def summary(text: str) -> dict[str, Any]:
            return {"type": "summary_text", "text": text}

        def reasoning_item(text: str | None) -> dict[str, Any]:
            return {"type": "reasoning", "id": rs_id, "summary": [summary(text)] if text else []}

        def opened(kind: str) -> str:
            out = closed()
            index_of[kind] = items["next"]
            items["next"] += 1
            items["open"] = kind
            at = index_of[kind]
            if kind == "reasoning":
                return (
                    out
                    + event(
                        "response.output_item.added", output_index=at, item=reasoning_item(None)
                    )
                    + event(
                        "response.reasoning_summary_part.added",
                        item_id=rs_id,
                        output_index=at,
                        summary_index=0,
                        part=summary(""),
                    )
                )
            return (
                out
                + event(
                    "response.output_item.added",
                    output_index=at,
                    item=_message_item(msg_id, "", status="in_progress"),
                )
                + event(
                    "response.content_part.added",
                    item_id=msg_id,
                    output_index=at,
                    content_index=0,
                    part={"type": "output_text", "text": "", "annotations": []},
                )
            )

        def closed() -> str:
            kind, items["open"] = items["open"], ""
            if not kind:
                return ""
            at = index_of[kind]
            if kind == "reasoning":
                text = "".join(items["reasoning"])
                output.append(reasoning_item(text))
                return (
                    event(
                        "response.reasoning_summary_text.done",
                        item_id=rs_id,
                        output_index=at,
                        summary_index=0,
                        text=text,
                    )
                    + event(
                        "response.reasoning_summary_part.done",
                        item_id=rs_id,
                        output_index=at,
                        summary_index=0,
                        part=summary(text),
                    )
                    + event("response.output_item.done", output_index=at, item=reasoning_item(text))
                )
            text = "".join(items["answer"])
            output.append(_message_item(msg_id, text))
            part = {"type": "output_text", "text": text, "annotations": []}
            return (
                event(
                    "response.output_text.done",
                    item_id=msg_id,
                    output_index=at,
                    content_index=0,
                    text=text,
                )
                + event(
                    "response.content_part.done",
                    item_id=msg_id,
                    output_index=at,
                    content_index=0,
                    part=part,
                )
                + event(
                    "response.output_item.done", output_index=at, item=_message_item(msg_id, text)
                )
            )

        def written(kind: str, text: str) -> str:
            if not text:
                return ""
            out = "" if items["open"] == kind else opened(kind)
            items[kind].append(text)
            if kind == "reasoning":
                return out + event(
                    "response.reasoning_summary_text.delta",
                    item_id=rs_id,
                    output_index=index_of[kind],
                    summary_index=0,
                    delta=text,
                )
            return out + event(
                "response.output_text.delta",
                item_id=msg_id,
                output_index=index_of[kind],
                content_index=0,
                delta=text,
            )

        def split(answer: str, reasoning: str) -> str:
            shown = reasoning if show_reasoning else ""
            return written("reasoning", shown) + written("answer", answer)

        def format_item(token: str) -> str:
            accumulated.append(token)
            # Tool-aware turns buffer everything and emit nothing until done,
            # so a raw tool-call marker is never streamed verbatim as text.
            if tool_aware:
                return ""
            return split(*splitter.feed(token))

        def format_done() -> str:
            if tool_aware:
                resolved = resolve_chat_output("".join(accumulated), model, tools)
                answer, reasoning = resolved.content, resolved.reasoning or ""
                tool_calls = resolved.tool_calls
            else:
                answer, reasoning = splitter.flush()
                tool_calls = []
            out = split("" if tool_calls else answer, reasoning)
            if not tool_calls and "answer" not in index_of:
                # Every turn without a call has a message, if an empty one.
                out += opened("answer")
            out += closed()
            for item in _function_call_items(tool_calls):
                index = items["next"]
                items["next"] += 1
                out += event(
                    "response.output_item.added",
                    output_index=index,
                    item={**item, "arguments": "", "status": "in_progress"},
                )
                out += event(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=index,
                    delta=item["arguments"],
                )
                out += event(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=index,
                    arguments=item["arguments"],
                )
                out += event("response.output_item.done", output_index=index, item=item)
                output.append(item)
            from hfl.engine.base import stream_counts

            prompt_n, generated = stream_counts(sync_iter)
            completed = _envelope(
                response_id,
                model,
                output,
                prompt_n or 0,
                generated if generated is not None else len(accumulated),
            )
            completed.update(extra or {})
            if on_done is not None:
                on_done(output)
            return out + event("response.completed", response=completed) + "data: [DONE]\n\n"

        if tool_aware:
            try:
                sync_iter = state.engine.chat_stream(messages, cfg, tools=tools)
            except TypeError:
                sync_iter = state.engine.chat_stream(messages, cfg)
        else:
            sync_iter = state.engine.chat_stream(messages, cfg)

        try:
            async for chunk in stream_with_backpressure(
                sync_iterator=sync_iter,
                format_item=format_item,
                format_done=format_done,
            ):
                yield chunk
        except Exception:
            logger.exception("responses stream failed")
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "response.failed",
                        "error": {
                            "code": "stream_error",
                            "message": "Internal server error during streaming.",
                        },
                    }
                )
                + "\n\n"
            )
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)


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
async def responses(req: ResponsesRequest) -> dict[str, Any] | StreamingResponse | JSONResponse:
    """OpenAI-compatible ``POST /v1/responses``.

    Bundles input, instructions, tools, reasoning effort and structured
    output into a single endpoint. Internally maps to chat-completion
    semantics; ``previous_response_id`` continues a response stored in this
    process (``response_store``).
    """
    from hfl.api.model_loader import load_llm

    await load_llm(req.model)

    state = _get_state()
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from hfl.api.response_store import get_response_store

    store = get_response_store()
    history: list[ChatMessage] = []
    if req.previous_response_id:
        found = store.history(req.previous_response_id)
        if found is None:
            return _not_found(req.previous_response_id)
        history = found
    # The previous response's ``instructions`` are not carried over (as
    # OpenAI does): only this request's, before the conversation.
    system = _input_to_messages([], req.instructions)
    new = _input_to_messages(req.input, None, _call_names(history))
    messages = [*system, *history, *new]
    cfg = _build_gen_config(req)
    tools = _chat_tools(req.tools)

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    extra = {"previous_response_id": req.previous_response_id, "store": req.store}

    def remember(output: list[dict[str, Any]]) -> None:
        if req.store:
            store.put(response_id, req.previous_response_id, [*new, *_assistant_turn(output)])

    if req.stream:
        # API/CON: hold a dispatcher slot for the whole stream (serialise
        # against other inference requests on the shared non-reentrant model)
        # and drive the engine through stream_with_backpressure so the
        # iterator is closed on disconnect — matching every other dialect.
        return await prepare_stream_response(
            lambda slot: _stream_response(
                response_id, req.model, messages, cfg, tools, slot, extra=extra, on_done=remember
            ),
            media_type="text/event-stream",
            path="/v1/responses",
        )

    if tools is not None:
        try:
            result = await run_dispatched(
                state.engine.chat,
                messages,
                cfg,
                tools=tools,
            )
        except TypeError:
            result = await run_dispatched(state.engine.chat, messages, cfg)
    else:
        result = await run_dispatched(state.engine.chat, messages, cfg)

    raw_text = getattr(result, "text", "") or ""
    tokens_input = int(getattr(result, "tokens_prompt", 0) or 0)
    tokens_output = int(getattr(result, "tokens_generated", 0) or 0)
    engine_tool_calls = getattr(result, "tool_calls", None)

    # The shared decision (chat_core): the engine's structured tool_calls,
    # else markers parsed out of the text; the reasoning taken out of the
    # answer either way.
    resolved = resolve_chat_output(raw_text, req.model, tools, engine_tool_calls)
    tool_calls = list(resolved.tool_calls) or None
    cleaned_text = resolved.content
    reasoning_text = resolved.reasoning if cfg.reasoning != "off" else None

    rendered = _render_response(
        response_id=response_id,
        model=req.model,
        text=cleaned_text,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tool_calls=tool_calls,
        reasoning_text=reasoning_text,
    )
    rendered.update(extra)
    remember(rendered["output"])
    return rendered

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatible with the Ollama native API.
Allows using hfl as a drop-in replacement for Ollama.
"""

import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import Response, StreamingResponse

from hfl.api.converters import ollama_to_generation_config
from hfl.api.errors import service_unavailable
from hfl.api.helpers import (
    apply_keep_alive,
    prepare_stream_response,
    queue_response_from_error,
    run_dispatched,
    unload_after_response,
)
from hfl.api.schemas import ChatRequest, GenerateRequest
from hfl.api.tool_parsers import dispatch as parse_tool_calls
from hfl.core.container import get_registry
from hfl.engine.base import ChatMessage, GenerationConfig
from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

if TYPE_CHECKING:
    from hfl.api.state import ServerState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ollama API"])


# --- Helpers ---


def _get_state() -> "ServerState":
    """Get the singleton server state."""
    from hfl.api.state import get_state

    return get_state()


async def _ensure_model_loaded(model_name: str) -> None:
    """Load the model if it is not already in memory (thread-safe)."""
    from hfl.api.model_loader import load_llm

    await load_llm(model_name)


def _options_to_config(options: dict | None) -> GenerationConfig:
    """Convert Ollama options dict to GenerationConfig."""
    return ollama_to_generation_config(options)


def _to_chat_messages(req: ChatRequest) -> list[ChatMessage]:
    """Convert an Ollama request's messages into engine-level ChatMessages.

    Preserves ``tool_calls``, ``name`` and ``tool_call_id`` so multi-turn
    agent loops (spec rule C6) propagate back to the model. Also
    decodes + validates ``images`` (Phase 4, P0-6) so the engine
    sees raw bytes.
    """
    from hfl.api.vision import decode_ollama_images

    out: list[ChatMessage] = []
    for m in req.messages:
        tool_calls = None
        if m.tool_calls:
            tool_calls = [tc.model_dump() for tc in m.tool_calls]
        out.append(
            ChatMessage(
                role=m.role,
                content=m.content or "",
                tool_calls=tool_calls,
                name=m.name,
                tool_call_id=m.tool_call_id,
                images=decode_ollama_images(m.images),
            )
        )
    return out


def _tools_payload(req: ChatRequest) -> list[dict] | None:
    """Materialize the ``tools`` field as a list of plain dicts, or None."""
    if req.tools is None:
        return None
    return [t.model_dump() for t in req.tools]


def _merge_mcp_tools(existing: list[dict] | None) -> list[dict] | None:
    """Merge tools exposed by connected MCP servers into ``existing``.

    Returns ``existing`` untouched when no MCP tools are active, so
    requests that opt out of tool calling (``tools=None``) continue
    to suppress tool calls even if MCP servers are connected — the
    caller's intent wins.
    """
    try:
        from hfl.mcp.client import get_client
    except Exception:
        return existing
    try:
        mcp_tools = get_client().list_tools()
    except Exception:
        logger.exception("failed to read MCP tool list")
        return existing
    if not mcp_tools:
        return existing
    mcp_payload = [t.to_ollama_tool() for t in mcp_tools]
    if existing is None:
        # Caller didn't declare tools and hasn't opted out (opt-out
        # is ``tools=[]``). We treat ``None`` as "no preference" and
        # surface MCP tools.
        return mcp_payload
    return existing + mcp_payload


def _baked_messages(manifest: "Any | None") -> list[ChatMessage]:
    """Return the manifest's Modelfile ``MESSAGE`` entries as ChatMessages.

    Empty list when the manifest has none (or when ``manifest`` is
    ``None``, which happens if the engine somehow loaded without a
    current_model set — should never occur on the happy path but the
    route stays defensive).
    """
    if manifest is None:
        return []
    raw = getattr(manifest, "messages", None) or []
    out: list[ChatMessage] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        out.append(ChatMessage(role=role, content=content))
    return out


def _splice_baked_messages(
    messages: list[ChatMessage],
    baked: list[ChatMessage],
) -> list[ChatMessage]:
    """Insert ``baked`` messages after any leading system block.

    The contract matches Ollama's Modelfile semantics: MESSAGE entries
    sit between the system prompt(s) and the live turn. We find the
    first non-system message and splice the bake-in there; if the
    caller sent only system messages, they land at the end.
    """
    if not baked:
        return messages
    split = 0
    for i, m in enumerate(messages):
        if m.role != "system":
            split = i
            break
    else:
        split = len(messages)
    return messages[:split] + baked + messages[split:]


def _build_chat_message(
    raw_text: str,
    model_name: str,
    tools: list[dict] | None,
    engine_tool_calls: list[dict] | None,
) -> dict:
    """Build the canonical assistant message for an Ollama chat response.

    - If the engine returned structured ``tool_calls``, use them as-is.
    - Otherwise, run the text through the per-family parser and move any
      recovered calls into ``tool_calls``, stripping them from ``content``.
    - When ``tool_calls`` is non-empty, ``content`` is set to "" so the
      client never has to parse narration out of the tool payload
      (spec rule C4).
    - ``tool_calls`` is always a list (spec rule C7): empty when none.
    """
    # Only trust structured tool_calls from the engine if they are an
    # actual list — MagicMock auto-attrs in unit tests would otherwise
    # masquerade as populated data.
    if isinstance(engine_tool_calls, list) and engine_tool_calls:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": engine_tool_calls,
        }

    cleaned, parsed_calls = parse_tool_calls(raw_text, model_name, tools)
    if parsed_calls:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": parsed_calls,
        }
    return {
        "role": "assistant",
        "content": cleaned,
        "tool_calls": [],
    }


# --- Endpoints ---


@router.post(
    "/api/generate",
    response_model=None,
    tags=["Ollama"],
    summary="Generate text",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def api_generate(
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any] | StreamingResponse | Response:
    """Ollama-compatible ``POST /api/generate`` (raw prompt completion).

    Either streams NDJSON chunks (``req.stream=True``) or returns the
    complete response envelope. Options map from Ollama's ``options``
    dict to HFL's ``GenerationConfig`` at the route boundary. The
    ``keep_alive`` field, if present, records a deadline on the server
    state (surfaced by ``/api/ps``'s ``expires_at``) or queues an
    immediate unload when set to 0.
    """
    unload_after = apply_keep_alive(req.model, req.keep_alive)
    await _ensure_model_loaded(req.model)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{req.model}' failed to load")

    # Schedule post-response unload when the client asked for it.
    if unload_after:
        background_tasks.add_task(unload_after_response, req.model)

    gen_config = _options_to_config(req.options)

    # OLLAMA_PARITY_PLAN P0-5: honour ``format`` for structured output.
    if req.format is not None:
        from hfl.api.structured_outputs import normalize_ollama_format

        gen_config.response_format = normalize_ollama_format(req.format)

    # OLLAMA_PARITY_PLAN P1-1: expose reasoning channel on request.
    if req.think:
        gen_config.expose_reasoning = True

    # OLLAMA_PARITY_PLAN P2-3: per-request template override and raw
    # prompt mode. The engine decides how to honour each — llama-cpp
    # rewires its chat handler for ``template_override``, and falls
    # through to ``Llama.__call__`` (no chat formatting) when
    # ``raw=True``. Backends without plug-in templates (vLLM) ignore
    # both flags silently.
    if req.template is not None:
        gen_config.template_override = req.template
    if req.raw:
        gen_config.raw = True

    # OLLAMA_PARITY_PLAN P1-1: system-prompt override. When present,
    # we flow it through the engine as a system-role message so the
    # same /api/generate route can now do single-shot prompting with
    # a custom system prompt, mirroring Ollama's behaviour. Skipped
    # when ``raw=True`` because raw mode intentionally bypasses any
    # prompt shaping.
    if req.system and not req.raw:
        system_preamble = req.system + "\n\n"
        final_prompt = system_preamble + req.prompt
    else:
        final_prompt = req.prompt

    if req.stream:
        return await prepare_stream_response(
            lambda slot: _stream_generate(req.model, final_prompt, gen_config, slot),
            media_type="application/x-ndjson",
        )

    # Run sync engine call in thread pool with timeout, serialized by
    # the inference dispatcher (spec §5.3).
    try:
        result = await run_dispatched(
            state.engine.generate,
            final_prompt,
            gen_config,
            operation="generate",
        )
    except (QueueFullError, QueueTimeoutError) as exc:
        return queue_response_from_error(exc)
    # Phase 5 P1-3: real nanosecond timings (was hard-coded to 0
    # pre-0.5.1). Clients keying off these fields now see the
    # engine's actual measurements.
    envelope: dict[str, Any] = {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "response": result.text,
        "done": True,
        "total_duration": result.total_duration,
        "load_duration": result.load_duration,
        "prompt_eval_count": result.tokens_prompt,
        "prompt_eval_duration": result.prompt_eval_duration,
        "eval_count": result.tokens_generated,
        "eval_duration": result.eval_duration,
    }
    # Phase 7 P2-4: opt-in legacy ``context`` token array for
    # multi-turn continuation. Only surfaces when the client sent
    # ``options.keep_context=true``.
    if gen_config.keep_context and result.context_tokens is not None:
        envelope["context"] = result.context_tokens
    return envelope


async def _stream_generate(
    model_name: str,
    prompt: str,
    config: GenerationConfig,
    slot_cm: Any | None = None,
) -> AsyncIterator[str]:
    """Stream text generation in Ollama NDJSON format with backpressure.

    ``slot_cm`` is an already-entered dispatcher slot context manager
    (from :func:`_acquire_stream_slot`). It is released at the end of
    the generator regardless of success or failure.
    """
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        error = {"model": model_name, "error": "Model not loaded", "done": True}
        yield json.dumps(error) + "\n"
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)
        return

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def format_chunk(token: str) -> str:
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "response": token,
            "done": False,
        }
        return json.dumps(chunk) + "\n"

    def format_done() -> str:
        chunk = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": "",
            "done": True,
        }
        return json.dumps(chunk) + "\n"

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=state.engine.generate_stream(prompt, config),
            format_item=format_chunk,
            format_done=format_done,
        ):
            yield chunk
    except Exception:
        # Never forward ``str(exc)`` to the client — it may reveal
        # paths, library class names, or line numbers (CodeQL
        # ``py/stack-trace-exposure``). Full traceback goes to the
        # server log via ``logger.exception``.
        logger.exception("ollama stream failed for model %s", model_name)
        error = {
            "model": model_name,
            "error": "Internal server error during streaming.",
            "done": True,
        }
        yield json.dumps(error) + "\n"
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)


@router.post(
    "/api/chat",
    response_model=None,
    tags=["Ollama"],
    summary="Chat with model",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def api_chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any] | StreamingResponse | Response:
    """Ollama-compatible ``POST /api/chat`` (multi-turn chat).

    Supports tool calls: ``req.tools`` flows into the engine; the
    response's ``message.tool_calls`` is populated either from the
    engine's native structure or by running the per-family tool parser
    over the generated text (spec rule C4). Honours Ollama's
    ``keep_alive`` field — see ``api_generate`` for semantics.
    """
    unload_after = apply_keep_alive(req.model, req.keep_alive)
    await _ensure_model_loaded(req.model)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(f"Model '{req.model}' failed to load")

    if unload_after:
        background_tasks.add_task(unload_after_response, req.model)

    messages = _to_chat_messages(req)
    gen_config = _options_to_config(req.options)
    tools = _tools_payload(req)
    # Phase 9 P0: fold MCP tools into the tools list so every
    # connected external server is reachable from the model without
    # the client having to declare them. Best-effort — if the MCP
    # SDK isn't installed or no servers are connected, this is a
    # no-op.
    tools = _merge_mcp_tools(tools)

    # OLLAMA_PARITY_PLAN P0-5: structured-output constraint.
    if req.format is not None:
        from hfl.api.structured_outputs import normalize_ollama_format

        gen_config.response_format = normalize_ollama_format(req.format)

    # P1-1: expose reasoning channel if the client asked for it.
    if req.think:
        gen_config.expose_reasoning = True

    # P1-1: system-prompt override. Inserted as the FIRST message so
    # the model's chat template reads it before any user / tool
    # content. Any system message the caller already included stays
    # in place (we don't collapse — Ollama allows multiple).
    if req.system:
        from hfl.engine.base import ChatMessage as _CM

        messages = [_CM(role="system", content=req.system)] + messages

    # Phase 8 P3-2: Modelfile ``MESSAGE`` baked-in few-shot. If the
    # resolved manifest carries MESSAGE instructions, prepend them
    # to the conversation after any system messages so the model
    # sees the canonical ``user → assistant`` exemplars before the
    # live turn. No-op when the manifest has no baked messages
    # (the vast majority of models).
    baked = _baked_messages(state.current_model)
    if baked:
        messages = _splice_baked_messages(messages, baked)

    if req.stream:
        return await prepare_stream_response(
            lambda slot: _stream_chat(req.model, messages, gen_config, tools, slot),
            media_type="application/x-ndjson",
        )

    # Run sync engine call in thread pool with timeout, serialized by
    # the inference dispatcher (spec §5.3). ``tools`` is forwarded as a
    # kwarg so engines without tool support raise TypeError and fall
    # back via the per-engine shim.
    trace_payload: list[dict] | None = None
    try:
        if req.agent_loop:
            # Phase 10 P1: native agent loop. Dispatches tool calls
            # itself (MCP only for now) and re-submits results to the
            # model until we get a tool-call-free reply or reach
            # ``max_iterations``.
            from hfl.api.agent_loop import run_agent_loop

            async def _chat_fn(msgs, cfg, tools_arg):
                return await run_dispatched(
                    state.engine.chat,
                    msgs,
                    cfg,
                    tools=tools_arg,
                    operation="chat",
                )

            mcp_caller = None
            try:
                from hfl.mcp.client import get_client as _get_mcp

                mcp_client = _get_mcp()
                mcp_caller = mcp_client.call_tool
            except Exception:
                mcp_caller = None

            loop_result = await run_agent_loop(
                messages=messages,
                config=gen_config,
                tools=tools,
                chat_fn=_chat_fn,
                mcp_caller=mcp_caller,
                max_iterations=req.max_iterations or 6,
            )
            result = loop_result.final
            trace_payload = [
                {
                    "iteration": entry.iteration,
                    "name": entry.name,
                    "arguments": entry.arguments,
                    "result": entry.result,
                    "error": entry.error,
                    "call_id": entry.call_id,
                }
                for entry in loop_result.tool_trace
            ]
        else:
            result = await run_dispatched(
                state.engine.chat,
                messages,
                gen_config,
                tools=tools,
                operation="chat",
            )
    except (QueueFullError, QueueTimeoutError) as exc:
        return queue_response_from_error(exc)

    # P1-1: when think=True, capture the reasoning from the RAW
    # engine text BEFORE the tool parser runs — the per-family
    # parsers in tool_parsers already strip ``<think>`` / channel
    # blocks as part of their cleanup, so we'd lose the reasoning if
    # we extracted it from the post-build message.
    extracted_thinking: str | None = None
    raw_for_build = result.text
    if req.think:
        from hfl.api.thinking import extract_thinking

        _, extracted_thinking = extract_thinking(result.text)

    message = _build_chat_message(
        raw_text=raw_for_build,
        model_name=req.model,
        tools=tools,
        engine_tool_calls=getattr(result, "tool_calls", None),
    )

    if extracted_thinking:
        message["thinking"] = extracted_thinking

    envelope: dict[str, Any] = {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": message,
        "done": True,
        # Phase 5 P1-3: real timings. Ollama-shape fields.
        "total_duration": result.total_duration,
        "load_duration": result.load_duration,
        "prompt_eval_count": result.tokens_prompt,
        "prompt_eval_duration": result.prompt_eval_duration,
        "eval_count": result.tokens_generated,
        "eval_duration": result.eval_duration,
    }
    # Phase 10 P1: replay-ready agent-loop trace for consumers that
    # want to inspect each tool invocation the server dispatched on
    # their behalf.
    if trace_payload is not None:
        envelope["tool_trace"] = trace_payload
    return envelope


async def _stream_chat(
    model_name: str,
    messages: list[ChatMessage],
    config: GenerationConfig,
    tools: list[dict] | None = None,
    slot_cm: Any | None = None,
) -> AsyncIterator[str]:
    """Stream chat in Ollama NDJSON format with backpressure.

    The full generated text is accumulated so that the final ``done: true``
    chunk can emit structured ``tool_calls`` extracted from the assembled
    output (spec rule C5). Intermediate chunks keep ``tool_calls: null``.

    ``slot_cm`` is an already-entered dispatcher slot context manager
    from :func:`_acquire_stream_slot`; it is released in ``finally``
    regardless of outcome (spec §5.3 — a hung stream must not leak
    capacity).
    """
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        error = {"model": model_name, "error": "Model not loaded", "done": True}
        yield json.dumps(error) + "\n"
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)
        return

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    accumulated: list[str] = []

    def format_chunk(token: str) -> str:
        accumulated.append(token)
        chunk = {
            "model": model_name,
            "created_at": created_at,
            "message": {
                "role": "assistant",
                "content": token,
                "tool_calls": None,
            },
            "done": False,
        }
        return json.dumps(chunk) + "\n"

    def format_done() -> str:
        full_text = "".join(accumulated)
        cleaned, calls = parse_tool_calls(full_text, model_name, tools)
        final_message: dict = {"role": "assistant", "content": "", "tool_calls": []}
        if calls:
            final_message["tool_calls"] = calls
        chunk = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": final_message,
            "done": True,
        }
        return json.dumps(chunk) + "\n"

    # Obtain the sync token iterator, tolerating engines without ``tools``
    # support by falling back to the 2-arg signature.
    if tools is not None:
        try:
            sync_iterator = state.engine.chat_stream(messages, config, tools=tools)
        except TypeError:
            sync_iterator = state.engine.chat_stream(messages, config)
    else:
        sync_iterator = state.engine.chat_stream(messages, config)

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=sync_iterator,
            format_item=format_chunk,
            format_done=format_done,
        ):
            yield chunk
    except Exception:
        # Never forward ``str(exc)`` to the client — it may reveal
        # paths, library class names, or line numbers (CodeQL
        # ``py/stack-trace-exposure``). Full traceback goes to the
        # server log via ``logger.exception``.
        logger.exception("ollama stream failed for model %s", model_name)
        error = {
            "model": model_name,
            "error": "Internal server error during streaming.",
            "done": True,
        }
        yield json.dumps(error) + "\n"
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)


@router.get("/api/tags", tags=["Ollama"], summary="List local models")
async def api_tags() -> dict[str, Any]:
    """List models (Ollama compatible).

    Returns model list matching Ollama's /api/tags format.
    Uses None instead of empty strings for missing optional fields.
    """
    registry = get_registry()
    return {
        "models": [
            {
                "name": m.name,
                "modified_at": m.created_at,
                "size": m.size_bytes,
                "digest": m.file_hash or "",  # Use actual hash if available
                "details": {
                    "parent_model": m.repo_id,
                    "format": m.format,
                    "family": m.architecture,  # None instead of ""
                    "parameter_size": m.parameters,  # None instead of ""
                    "quantization_level": m.quantization,  # None instead of ""
                },
            }
            for m in registry.list_all()
        ]
    }


@router.get("/api/version", tags=["Ollama"], summary="Get server version")
async def api_version() -> dict[str, str]:
    """Server version (Ollama compatible)."""
    return {"version": "0.1.0"}


@router.head("/")
async def head_root() -> dict[str, Any]:
    """Health check for Ollama compatibility."""
    return {}

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Endpoints compatible with the OpenAI API.
Drop-in replacement for applications using the OpenAI SDK.
"""

import contextlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Union

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response, StreamingResponse

from hfl.api.chat_core import resolve_chat_output
from hfl.api.converters import openai_to_generation_config
from hfl.api.errors import service_unavailable
from hfl.api.helpers import (
    prepare_stream_response,
    queue_response_from_error,
    run_dispatched,
)
from hfl.api.schemas import ChatCompletionRequest, CompletionRequest
from hfl.api.thinking import ThinkingSplitter
from hfl.core.container import get_registry
from hfl.engine.base import ChatMessage, GenerationConfig, stream_counts
from hfl.engine.dispatcher import QueueFullError, QueueTimeoutError

if TYPE_CHECKING:
    from hfl.api.state import ServerState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OpenAI API"])


# --- Helpers ---


def _get_state() -> "ServerState":
    """Get the singleton server state."""
    from hfl.api.state import get_state

    return get_state()


async def _ensure_model_loaded(model_name: str) -> None:
    """Load the model if it is not already in memory (thread-safe)."""
    from hfl.api.model_loader import load_llm

    await load_llm(model_name)


def _to_gen_config(req: Union[ChatCompletionRequest, CompletionRequest]) -> GenerationConfig:
    """Convert OpenAI request to GenerationConfig."""
    return openai_to_generation_config(req)


def _tools_payload(req: ChatCompletionRequest) -> list[dict] | None:
    """OpenAI ``tools[]`` (honouring ``tool_choice``) as plain dicts.

    ``None`` (no preference) and ``[]`` (explicit opt-out) both collapse to
    ``None`` so plain chat is never accidentally routed through the
    tool-aware template.

    ``tool_choice`` is honoured to the extent a parse-based backend can:

    - ``"none"`` -> tools are dropped entirely, so the chat template never
      advertises them and the output is never scanned for tool-call markers.
      This is a hard guarantee: the response cannot contain ``tool_calls``.
    - ``{"type": "function", "function": {"name": "X"}}`` -> only tool ``X``
      is forwarded, narrowing the model's choice to the requested function.
    - ``"auto"`` / ``"required"`` / unset -> all declared tools are forwarded.
      ("required" cannot be hard-enforced without constrained decoding, so it
      behaves like "auto" here.)
    """
    if not req.tools:
        return None
    choice = req.tool_choice
    if isinstance(choice, str) and choice.strip().lower() == "none":
        return None
    tools = [t.model_dump() for t in req.tools]
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        if name:
            narrowed = [t for t in tools if t.get("function", {}).get("name") == name]
            if narrowed:
                return narrowed
    return tools


def _openai_messages_to_chat(req: ChatCompletionRequest) -> list[ChatMessage]:
    """Build engine ``ChatMessage`` list from an OpenAI request.

    Carries the tool-calling fields across so multi-turn agent loops
    converge: a prior assistant ``tool_calls`` turn and the matching
    ``role=tool`` results are forwarded to the model. OpenAI sends tool-call
    ``arguments`` as a JSON **string**; the engine's canonical shape wants a
    dict, so we parse here (rule C3).
    """
    from hfl.api.vision import split_openai_content

    out: list[ChatMessage] = []
    for m in req.messages:
        text, images = split_openai_content(m.content or "")
        tool_calls: list[dict] | None = None
        if m.tool_calls:
            tool_calls = []
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (ValueError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append({"function": {"name": tc.function.name, "arguments": args}})
        out.append(
            ChatMessage(
                role=m.role,
                content=text,
                images=images,
                tool_calls=tool_calls,
                name=m.name,
                tool_call_id=m.tool_call_id,
            )
        )
    return out


def _to_openai_tool_calls(canonical: list[dict]) -> list[dict]:
    """Map canonical ``{"function":{"name","arguments":dict}}`` calls to the
    OpenAI wire shape ``{"id","type":"function","function":{"name","arguments":str}}``.

    ``arguments`` is serialised back to a JSON string — OpenAI clients
    (and the Vercel AI SDK) expect a string there, not an object.
    """
    out: list[dict] = []
    for call in canonical:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return out


# --- Endpoints ---


@router.post(
    "/v1/chat/completions",
    response_model=None,
    tags=["OpenAI"],
    summary="Create chat completion",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def chat_completions(
    req: ChatCompletionRequest,
) -> dict[str, Any] | StreamingResponse | Response:
    """OpenAI-compatible ``POST /v1/chat/completions``.

    Loads the requested model (if needed), then either streams SSE
    chunks (``req.stream=True``) or returns the full chat-completion
    response. Concurrent requests are serialised through the inference
    dispatcher (spec §5.3); dispatcher rejections map to 429/503.
    """
    await _ensure_model_loaded(req.model)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(
            f"Model '{req.model}' failed to load", path="/v1/chat/completions"
        )

    # Phase 4 P0-6: multimodal content + tool-calling fields are folded
    # into the engine ChatMessage list at the route boundary so engines
    # stay simple.
    messages = _openai_messages_to_chat(req)
    tools = _tools_payload(req)
    # "none" is a hard opt-out: never scan the output for tool-call markers,
    # not even the per-family parser (which fires on markers regardless of
    # whether tools were forwarded).
    tools_disabled = isinstance(req.tool_choice, str) and req.tool_choice.strip().lower() == "none"
    gen_config = _to_gen_config(req)

    # OLLAMA_PARITY_PLAN P0-5: honour OpenAI ``response_format``.
    if req.response_format is not None:
        from hfl.api.structured_outputs import normalize_openai_response_format

        gen_config.response_format = normalize_openai_response_format(req.response_format)

    if req.stream and (req.logprobs or req.n > 1):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "logprobs and n > 1 are not supported with stream yet",
                    "type": "invalid_request_error",
                    "param": "logprobs" if req.logprobs else "n",
                }
            },
        )
    if req.logprobs:
        gen_config.logprobs = req.top_logprobs or 0

    if req.stream:
        return await prepare_stream_response(
            lambda slot: _stream_chat(
                req.model, messages, gen_config, tools, slot, _include_usage(req)
            ),
            media_type="text/event-stream",
            path="/v1/chat/completions",
        )

    # Run sync engine call in thread pool with timeout, serialized by
    # the inference dispatcher (spec §5.3). ``tools`` is forwarded as a
    # kwarg so the model's tool-aware chat template is applied.
    # ``n`` answers, one after another (the model is shared); a seed, when
    # given, is moved on for each so they can differ.
    results = []
    for index in range(req.n):
        if index and gen_config.seed >= 0:
            gen_config.seed += 1
        try:
            results.append(
                await run_dispatched(
                    state.engine.chat,
                    messages,
                    gen_config,
                    tools=tools,
                    operation="chat_completion",
                )
            )
        except (QueueFullError, QueueTimeoutError) as exc:
            return queue_response_from_error(exc, path="/v1/chat/completions")
        except NotImplementedError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
    choices = [
        _choice(index, result, req, tools, tools_disabled, gen_config)
        for index, result in enumerate(results)
    ]
    first = results[0]
    generated = sum(result.tokens_generated for result in results)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": first.tokens_prompt,
            "completion_tokens": generated,
            "total_tokens": first.tokens_prompt + generated,
        },
    }


def _choice(
    index: int,
    result: Any,
    req: ChatCompletionRequest,
    tools: list[dict] | None,
    tools_disabled: bool,
    gen_config: GenerationConfig,
) -> dict[str, Any]:
    """One answer as an OpenAI ``choice``."""
    # Shared decision (see chat_core): prefer engine tool_calls, else parse
    # markers; "none" disables both. When a tool call is present, content is
    # null and finish_reason flips to ``tool_calls`` so OpenAI-SDK agent loops
    # dispatch instead of stopping.
    resolved = resolve_chat_output(
        result.text,
        req.model,
        tools,
        getattr(result, "tool_calls", None),
        tools_disabled=tools_disabled,
    )
    if resolved.has_tool_calls:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": _to_openai_tool_calls(resolved.tool_calls),
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": resolved.content}
        finish_reason = result.stop_reason
    if resolved.reasoning and gen_config.reasoning != "off":
        # Never inside the answer; beside it, as DeepSeek, vLLM and
        # llama-server send it.
        message["reasoning_content"] = resolved.reasoning
    choice: dict[str, Any] = {"index": index, "message": message, "finish_reason": finish_reason}
    if req.logprobs:
        choice["logprobs"] = {"content": getattr(result, "logprobs", None) or []}
    return choice


def _include_usage(req: Any) -> bool:
    options = getattr(req, "stream_options", None) or {}
    return bool(isinstance(options, dict) and options.get("include_usage"))


async def _stream_chat(
    model: str,
    messages: list[ChatMessage],
    config: GenerationConfig,
    tools: list[dict] | None = None,
    slot_cm: Any | None = None,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """Generate OpenAI-compatible SSE responses with backpressure.

    Plain chat (``tools`` is falsy) streams token-by-token as content
    deltas. When ``tools`` are declared the full generation is buffered so
    the per-family parser can lift tool-call markers out of the text into a
    single structured ``tool_calls`` delta (finish_reason ``tool_calls``) —
    this keeps the raw ``<tool_call>`` JSON from ever leaking to the client
    as content (spec rules C4/C5).

    ``slot_cm`` is the dispatcher slot held for the entire stream
    (spec §5.3). It is released in ``finally`` regardless of outcome.
    """
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        err = json.dumps(
            {
                "error": "Model not loaded",
                "code": "SERVICE_UNAVAILABLE",
            }
        )
        yield f"data: {err}\n\n"
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)
        return

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())  # Consistent timestamp across all chunks
    first_chunk = True
    tool_aware = bool(tools)
    accumulated: list[str] = []
    emitted = [0]  # API-13: token count → report finish_reason "length" on cap
    # Reasoning streams apart from the answer, in ``reasoning_content``.
    splitter = ThinkingSplitter()
    show_reasoning = config.reasoning != "off"

    def _chunk_json(delta: dict, finish_reason: str | None) -> str:
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": None,  # OpenAI compatibility
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_chunk(token: str) -> str:
        nonlocal first_chunk
        emitted[0] += 1
        # Tool-aware turns buffer everything; emit nothing until done so a
        # tool-call marker is never streamed verbatim as content.
        if tool_aware:
            accumulated.append(token)
            return ""
        return _deltas(*splitter.feed(token))

    def _deltas(answer: str, reasoning: str) -> str:
        nonlocal first_chunk
        out = ""
        for key, text in (("reasoning_content", reasoning if show_reasoning else ""),
                          ("content", answer)):  # fmt: skip
            if not text:
                continue
            delta: dict[str, str] = {key: text}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            out += _chunk_json(delta, None)
        return out

    stream = state.engine.chat_stream(messages, config, tools)

    def _usage_chunk() -> str:
        # ``stream_options.include_usage``: one last chunk, empty choices,
        # with the tokens the engine counted — none when it cannot count.
        prompt_n, generated = stream_counts(stream)
        if not include_usage or prompt_n is None or generated is None:
            return ""
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": None,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_n,
                "completion_tokens": generated,
                "total_tokens": prompt_n + generated,
            },
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_done() -> str:
        # API-13: report "length" when generation hit the token cap, else
        # "stop" (real OpenAI distinguishes them; the stream only exposes the
        # emitted-token count, so this is best-effort but accurate at the cap).
        generated = stream_counts(stream)[1]
        count = generated if generated is not None else emitted[0]
        stop_finish = "length" if (config.max_tokens and count >= config.max_tokens) else "stop"
        if not tool_aware:
            return (
                _deltas(*splitter.flush())
                + _chunk_json({}, stop_finish)
                + _usage_chunk()
                + "data: [DONE]\n\n"
            )

        resolved = resolve_chat_output("".join(accumulated), model, tools)
        thought = _deltas("", resolved.reasoning or "")
        if resolved.has_tool_calls:
            calls = _to_openai_tool_calls(resolved.tool_calls)
            delta = {
                "role": "assistant",
                "tool_calls": [{"index": i, **tc} for i, tc in enumerate(calls)],
            }
            return (
                thought
                + _chunk_json(delta, None)
                + _chunk_json({}, "tool_calls")
                + _usage_chunk()
                + "data: [DONE]\n\n"
            )
        # No tool call after all — surface the cleaned text as one delta.
        return (
            thought
            + _chunk_json({"role": "assistant", "content": resolved.content}, None)
            + _chunk_json({}, stop_finish)
            + _usage_chunk()
            + "data: [DONE]\n\n"
        )

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=stream,
            format_item=format_chunk,
            format_done=format_done,
        ):
            yield chunk
    except Exception:
        # Don't emit ``str(exc)`` on the stream — it may leak paths
        # or class names (CodeQL ``py/stack-trace-exposure``). Full
        # traceback is in the server log.
        logger.exception("openai stream failed")
        yield (
            "data: "
            + json.dumps(
                {
                    "error": "Internal server error during streaming.",
                    "code": "STREAM_ERROR",
                }
            )
            + "\n\n"
        )
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)


@router.post(
    "/v1/completions",
    response_model=None,
    tags=["OpenAI"],
    summary="Create text completion",
    responses={
        400: {"description": "Invalid request parameters"},
        404: {"description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        504: {"description": "Generation timeout"},
    },
)
async def completions(req: CompletionRequest) -> dict[str, Any] | StreamingResponse | Response:
    """OpenAI-compatible ``POST /v1/completions`` (legacy text completion).

    Same dispatcher / streaming / error semantics as
    ``chat_completions`` but takes a plain prompt instead of a
    message list.
    """
    await _ensure_model_loaded(req.model)
    state = _get_state()
    if state.engine is None:
        return service_unavailable(
            f"Model '{req.model}' failed to load", path="/v1/chat/completions"
        )

    # An empty ``prompt`` array would IndexError on ``[0]`` below and surface
    # as a 500; real OpenAI returns 400. Raise the domain ValidationError so the
    # HFLError handler renders it in the OpenAI/Anthropic dialect (API-3).
    if isinstance(req.prompt, list) and not req.prompt:
        from hfl.exceptions import ValidationError as APIValidationError

        raise APIValidationError("'prompt' must not be empty")
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    gen_config = _to_gen_config(req)

    if req.stream:
        return await prepare_stream_response(
            lambda slot: _stream_completion(
                req.model, prompt, gen_config, slot, _include_usage(req)
            ),
            media_type="text/event-stream",
            path="/v1/completions",
        )

    # Run sync engine call in thread pool with timeout, serialized by
    # the inference dispatcher (spec §5.3).
    try:
        result = await run_dispatched(
            state.engine.generate,
            prompt,
            gen_config,
            operation="text_completion",
        )
    except (QueueFullError, QueueTimeoutError) as exc:
        return queue_response_from_error(exc, path="/v1/chat/completions")

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "text": result.text,
                "index": 0,
                "finish_reason": result.stop_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.tokens_prompt,
            "completion_tokens": result.tokens_generated,
            "total_tokens": result.tokens_prompt + result.tokens_generated,
        },
    }


async def _stream_completion(
    model: str,
    prompt: str,
    config: GenerationConfig,
    slot_cm: Any | None = None,
    include_usage: bool = False,
) -> AsyncIterator[str]:
    """Generate text completion SSE responses with backpressure.

    ``slot_cm`` is the dispatcher slot held for the entire stream
    (spec §5.3); it is released in ``finally``.
    """
    from hfl.api.streaming import stream_with_backpressure

    state = _get_state()
    if state.engine is None:
        err = json.dumps(
            {
                "error": "Model not loaded",
                "code": "SERVICE_UNAVAILABLE",
            }
        )
        yield f"data: {err}\n\n"
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)
        return

    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    stream = state.engine.generate_stream(prompt, config)
    created = int(time.time())  # Consistent timestamp across all chunks
    emitted = [0]  # API-13: token count → report finish_reason "length" on cap

    def format_chunk(token: str) -> str:
        emitted[0] += 1
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "system_fingerprint": None,  # OpenAI compatibility
            "choices": [{"text": token, "index": 0, "finish_reason": None}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def format_done() -> str:
        # API-13: emit a final chunk carrying finish_reason — the legacy
        # completions stream previously sent only [DONE], so clients never
        # saw a stop reason at all.
        prompt_n, generated = stream_counts(stream)
        count = generated if generated is not None else emitted[0]
        finish = "length" if (config.max_tokens and count >= config.max_tokens) else "stop"
        final: dict[str, Any] = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "system_fingerprint": None,
            "choices": [{"text": "", "index": 0, "finish_reason": finish}],
        }
        usage = ""
        if include_usage and prompt_n is not None and generated is not None:
            usage_chunk = {
                **final,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_n,
                    "completion_tokens": generated,
                    "total_tokens": prompt_n + generated,
                },
            }
            usage = f"data: {json.dumps(usage_chunk)}\n\n"
        return f"data: {json.dumps(final)}\n\n" + usage + "data: [DONE]\n\n"

    try:
        async for chunk in stream_with_backpressure(
            sync_iterator=stream,
            format_item=format_chunk,
            format_done=format_done,
        ):
            yield chunk
    except Exception:
        # Don't emit ``str(exc)`` on the stream — it may leak paths
        # or class names (CodeQL ``py/stack-trace-exposure``). Full
        # traceback is in the server log.
        logger.exception("openai stream failed")
        yield (
            "data: "
            + json.dumps(
                {
                    "error": "Internal server error during streaming.",
                    "code": "STREAM_ERROR",
                }
            )
            + "\n\n"
        )
    finally:
        if slot_cm is not None:
            with contextlib.suppress(Exception):
                await slot_cm.__aexit__(None, None, None)


@router.get("/v1/models", tags=["OpenAI"], summary="List available models")
async def list_models() -> dict[str, Any]:
    """OpenAI-compatible ``GET /v1/models`` — list all registry entries.

    Returns the canonical envelope ``{"object": "list", "data": [...]}``
    where each entry carries id, created timestamp, and owned_by.
    """
    registry = get_registry()
    models = registry.list_all()
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.repo_id.split("/")[0] if "/" in m.repo_id else "local",
            }
            for m in models
        ],
    }

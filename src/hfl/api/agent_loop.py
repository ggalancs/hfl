# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Native agent loop for /api/chat (Phase 10 P1).

When a client opts in via ``agent_loop=true``, HFL drives the
tool-use cycle server-side:

  1. Call the model with the user's messages + tools.
  2. If the model emits ``tool_calls``, dispatch each (MCP tools
     via the client, everything else rejected — the model shouldn't
     hallucinate external tool names).
  3. Append ``role=tool`` messages with the results.
  4. Re-call the model.
  5. Stop when either the model returns a tool-call-free reply or
     ``max_iterations`` is reached.

Parallelism: tool calls inside the same turn fire concurrently via
``asyncio.gather`` — a single turn that asks for "weather in
London and New York" resolves both in parallel rather than
round-tripping through the model twice.

Returns the final assistant message plus a ``tool_trace`` suitable
for replay / debugging.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)

__all__ = [
    "AgentTraceEntry",
    "AgentLoopResult",
    "run_agent_loop",
    "DEFAULT_MAX_ITERATIONS",
]

DEFAULT_MAX_ITERATIONS = 6


@dataclass
class AgentTraceEntry:
    """One tool invocation recorded for the trace."""

    iteration: int
    call_id: str | None
    name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None = None


@dataclass
class AgentLoopResult:
    """Return value of ``run_agent_loop``."""

    final: GenerationResult
    messages: list[ChatMessage]
    tool_trace: list[AgentTraceEntry] = field(default_factory=list)
    iterations: int = 0
    terminated_by: str = "final_answer"  # "final_answer" | "max_iterations"


# ----------------------------------------------------------------------
# Tool dispatch
# ----------------------------------------------------------------------


async def _dispatch_tool_call(
    call: dict[str, Any],
    iteration: int,
    mcp_caller: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
) -> AgentTraceEntry:
    """Invoke one tool. Returns the trace entry — never raises."""
    function = call.get("function") or {}
    name = function.get("name") or ""
    raw_args = function.get("arguments")
    call_id = call.get("id")

    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return AgentTraceEntry(
                iteration=iteration,
                call_id=call_id,
                name=name,
                arguments={},
                result=None,
                error="malformed tool-call JSON",
            )
    elif isinstance(raw_args, dict):
        arguments = raw_args
    else:
        arguments = {}

    if "__" in name and mcp_caller is not None:
        try:
            result = await mcp_caller(name, arguments)
        except Exception as exc:
            logger.exception("agent-loop tool dispatch failed: %s", name)
            return AgentTraceEntry(
                iteration=iteration,
                call_id=call_id,
                name=name,
                arguments=arguments,
                result=None,
                error=f"{type(exc).__name__}",
            )
        return AgentTraceEntry(
            iteration=iteration,
            call_id=call_id,
            name=name,
            arguments=arguments,
            result=result,
        )

    return AgentTraceEntry(
        iteration=iteration,
        call_id=call_id,
        name=name,
        arguments=arguments,
        result=None,
        error="no handler registered for this tool",
    )


def _serialize_tool_result(result: Any) -> str:
    """Collapse an arbitrary tool result into a string for the chat turn."""
    if result is None:
        return "null"
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


async def run_agent_loop(
    *,
    messages: list[ChatMessage],
    config: GenerationConfig,
    tools: list[dict] | None,
    chat_fn: Callable[
        [list[ChatMessage], GenerationConfig, list[dict] | None],
        Awaitable[GenerationResult],
    ],
    mcp_caller: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentLoopResult:
    """Drive a multi-turn tool-use cycle.

    Parameters
    ----------
    messages:
        Initial conversation — mutated **in place** is avoided; a
        fresh list is passed to ``chat_fn`` every iteration.
    config:
        ``GenerationConfig`` forwarded to ``chat_fn`` on every call.
    tools:
        Tool schema list passed to the model. May be ``None``.
    chat_fn:
        Awaitable that actually runs the model. The route injects a
        closure that dispatches through the engine dispatcher + runs
        in a thread.
    mcp_caller:
        Optional ``async def (name, args) -> result`` — called when
        the model emits a tool whose name contains ``__`` (the MCP
        ``server_id__tool_name`` convention). ``None`` disables MCP
        dispatch entirely.
    max_iterations:
        Hard cap on the number of model turns inside the loop.
        Default 6. Setting this to 1 means "one call, never loop".
    """
    working: list[ChatMessage] = list(messages)
    trace: list[AgentTraceEntry] = []
    final: GenerationResult | None = None

    for iteration in range(1, max_iterations + 1):
        final = await chat_fn(working, config, tools)

        if not final.tool_calls:
            return AgentLoopResult(
                final=final,
                messages=working,
                tool_trace=trace,
                iterations=iteration,
                terminated_by="final_answer",
            )

        # Record the assistant turn that requested the tool calls —
        # the model needs to see it on the next iteration to keep a
        # coherent conversation.
        working.append(
            ChatMessage(
                role="assistant",
                content=final.text,
                tool_calls=list(final.tool_calls),
            )
        )

        dispatch_tasks = [
            _dispatch_tool_call(call, iteration, mcp_caller) for call in final.tool_calls
        ]
        entries = await asyncio.gather(*dispatch_tasks)
        trace.extend(entries)

        for entry in entries:
            payload = (
                f"ERROR: {entry.error}" if entry.error else _serialize_tool_result(entry.result)
            )
            working.append(
                ChatMessage(
                    role="tool",
                    content=payload,
                    name=entry.name,
                    tool_call_id=entry.call_id,
                )
            )

    return AgentLoopResult(
        final=final if final is not None else GenerationResult(text=""),
        messages=working,
        tool_trace=trace,
        iterations=max_iterations,
        terminated_by="max_iterations",
    )

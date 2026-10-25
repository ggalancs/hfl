# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Anthropic's web search server tool, run by HFL.

A Messages-API client asks for ``{"type": "web_search_20250305", "name":
"web_search", "max_uses": 8}`` (Claude Code's WebSearch does, for every
search): a tool Anthropic's own servers run. HFL ignored it, and the model
answered from memory with a source it made up (measured: Python "3.13.0"
while python.org said 3.15.1). Here HFL runs it with its own web search
(``hfl websearch``): the model calls ``web_search(query)`` like any tool,
HFL searches, hands the results back, and the reply carries the
``server_tool_use`` and ``web_search_tool_result`` blocks Anthropic's does.
With no search available the reply is the ``unavailable`` error and no
answer: one from memory, with made-up sources, is worse than none.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from hfl.engine.base import ChatMessage

logger = logging.getLogger(__name__)

# Anthropic dates the versions of the tool (web_search_20250305, ...).
_SEARCH_TYPE = "web_search_"
DEFAULT_MAX_USES = 5
MAX_USES_CAP = 20
RESULTS_PER_SEARCH = 5


@dataclass
class WebSearchTool:
    """The web search server tool a request asked for."""

    name: str
    max_uses: int = DEFAULT_MAX_USES
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)

    def as_function(self) -> dict[str, Any]:
        """The tool as the model sees it: a function it calls."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the web. Returns the title, URL and a snippet of each page "
                    "found. Use it when the answer needs current or specific facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"}
                    },
                    "required": ["query"],
                },
            },
        }


def _extra(tool: Any) -> dict[str, Any]:
    return dict(getattr(tool, "model_extra", None) or {})


def is_web_search(tool: Any) -> bool:
    """Whether a request tool is Anthropic's web search server tool."""
    return str(_extra(tool).get("type") or "").startswith(_SEARCH_TYPE)


def _domains(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(d).strip().lower().lstrip(".") for d in value if str(d).strip()]


def web_search_tool(tools: list[Any] | None) -> WebSearchTool | None:
    """The web search server tool among a request's ``tools``, if any."""
    for tool in tools or []:
        if not is_web_search(tool):
            continue
        extra = _extra(tool)
        try:
            uses = int(extra.get("max_uses") or DEFAULT_MAX_USES)
        except (TypeError, ValueError):
            uses = DEFAULT_MAX_USES
        return WebSearchTool(
            name=str(getattr(tool, "name", None) or "web_search"),
            max_uses=max(1, min(uses, MAX_USES_CAP)),
            allowed_domains=_domains(extra.get("allowed_domains")),
            blocked_domains=_domains(extra.get("blocked_domains")),
        )
    return None


def _on(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _allowed(url: str, tool: WebSearchTool) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if tool.allowed_domains and not any(_on(host, d) for d in tool.allowed_domains):
        return False
    return not any(_on(host, d) for d in tool.blocked_domains)


def _opaque(snippet: str) -> str:
    """``encrypted_content``: opaque to the client, as Anthropic's is, and
    readable by HFL when the block comes back in a later turn."""
    return base64.b64encode(json.dumps({"snippet": snippet}).encode()).decode()


def _snippet(encrypted: Any) -> str:
    try:
        return str(json.loads(base64.b64decode(str(encrypted)))["snippet"])
    except Exception:
        return ""


def history_text(block: Any) -> str | None:
    """A ``server_tool_use`` or ``web_search_tool_result`` block from an
    earlier turn, as text for the model; None for any other block."""
    kind = getattr(block, "type", None)
    if kind == "server_tool_use":
        query = (getattr(block, "input", None) or {}).get("query", "")
        return f"(Searched the web for: {query})"
    if kind != "web_search_tool_result":
        return None
    content = getattr(block, "content", None)
    if isinstance(content, dict):
        return f"(The web search failed: {content.get('error_code', 'error')})"
    lines = [
        f"- {r.get('title', '')} ({r.get('url', '')}): {_snippet(r.get('encrypted_content'))}"
        for r in content or []
        if isinstance(r, dict)
    ]
    return "Web search results:\n" + "\n".join(lines) if lines else "(The web search found nothing)"


async def _web_search(query: str, max_results: int) -> dict[str, Any]:
    """HFL's web search (``hfl websearch``), imported when first used."""
    from hfl.tools.web_search import search

    return await search(query, max_results)


def _results_text(results: list[dict[str, str]]) -> str:
    if not results:
        return "No results."
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\nURL: {r.get('url', '')}\n{r.get('content', '')}"
        for i, r in enumerate(results, 1)
    )


def _call_name(call: dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("name") or "")


def _call_query(call: dict[str, Any]) -> str:
    args = (call.get("function") or {}).get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return args.strip()
    return str((args or {}).get("query") or "").strip() if isinstance(args, dict) else ""


@dataclass
class SearchTurn:
    """The reply to a turn with web search: its content blocks and usage."""

    blocks: list[dict[str, Any]]
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0

    @property
    def usage(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "server_tool_use": {"web_search_requests": self.searches},
        }


async def run(
    tool: WebSearchTool,
    messages: list[ChatMessage],
    tools: list[dict[str, Any]] | None,
    *,
    chat: Callable[[list[ChatMessage], list[dict[str, Any]]], Awaitable[Any]],
    resolve: Callable[[Any, list[dict[str, Any]]], Any],
    to_tool_use: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    stop_reason: Callable[[str], str],
    search: Callable[[str, int], Awaitable[dict[str, Any]]] | None = None,
) -> SearchTurn:
    """One turn in which the model may search the web, up to ``max_uses``
    times. ``chat`` runs the model; ``resolve`` reads its reply (text and
    tool calls); calls to the client's own tools end the turn as
    ``tool_use`` blocks, for the client to run."""
    searcher = search if search is not None else _web_search
    offered = [*(tools or []), tool.as_function()]
    history = list(messages)
    turn = SearchTurn(blocks=[], stop_reason="end_turn")
    # Up to max_uses rounds of searches, then one to answer with them.
    for _ in range(tool.max_uses + 2):
        # Searched enough: the search is no longer offered, so it answers.
        now = offered if turn.searches < tool.max_uses else list(tools or [])
        result = await chat(history, now)
        turn.input_tokens += int(getattr(result, "tokens_prompt", 0) or 0)
        turn.output_tokens += int(getattr(result, "tokens_generated", 0) or 0)
        resolved = resolve(result, now)
        calls: list[dict[str, Any]] = list(resolved.tool_calls) if resolved.has_tool_calls else []
        if resolved.content and resolved.content.strip():
            turn.blocks.append({"type": "text", "text": resolved.content})
        if not calls:
            turn.stop_reason = stop_reason(getattr(result, "stop_reason", "stop"))
            return turn
        searches = [c for c in calls if _call_name(c) == tool.name]
        others = [c for c in calls if _call_name(c) != tool.name]
        asked: list[dict[str, Any]] = []
        answers: list[ChatMessage] = []
        for call in searches:
            query = _call_query(call)
            tool_id = f"srvtoolu_{uuid.uuid4().hex[:24]}"
            turn.blocks.append(
                {
                    "type": "server_tool_use",
                    "id": tool_id,
                    "name": tool.name,
                    "input": {"query": query},
                }
            )
            error = None
            found: list[dict[str, str]] = []
            if turn.searches >= tool.max_uses:
                error = "max_uses_exceeded"
            elif not query:
                error = "invalid_input"
            else:
                try:
                    reply = await searcher(query, RESULTS_PER_SEARCH)
                except Exception as exc:  # WebSearchError, or the network
                    logger.warning("web search for a server tool failed: %s", exc)
                    error = "unavailable"
                else:
                    turn.searches += 1
                    found = [
                        r for r in reply.get("results") or [] if _allowed(r.get("url", ""), tool)
                    ]
            if error is not None:
                turn.blocks.append(
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_id,
                        "content": {"type": "web_search_tool_result_error", "error_code": error},
                    }
                )
                if error == "unavailable":
                    return turn
                text = f"The web search did not run ({error}). Answer with what you found already."
            else:
                turn.blocks.append(
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": r.get("url", ""),
                                "title": r.get("title", ""),
                                "encrypted_content": _opaque(r.get("content", "")),
                                "page_age": None,
                            }
                            for r in found
                        ],
                    }
                )
                text = _results_text(found)
            asked.append(
                {"id": tool_id, "function": {"name": tool.name, "arguments": {"query": query}}}
            )
            answers.append(
                ChatMessage(role="tool", content=text, tool_call_id=tool_id, name=tool.name)
            )
        if others:
            # The client's own tools: it runs them, and asks again.
            turn.blocks.extend(to_tool_use(others))
            turn.stop_reason = "tool_use"
            return turn
        history.append(
            ChatMessage(role="assistant", content=resolved.content or "", tool_calls=asked)
        )
        history.extend(answers)
    return turn


def sse_events(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """A finished message as the events of Anthropic's stream: each block
    opened, filled and closed, then the stop reason and the usage."""
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **{k: v for k, v in message.items() if k not in ("content", "usage")},
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": message["usage"]["input_tokens"], "output_tokens": 0},
                },
            },
        )
    ]
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        if kind == "text":
            head: dict[str, Any] = {"type": "text", "text": ""}
            delta: dict[str, Any] | None = {"type": "text_delta", "text": block["text"]}
        elif kind in ("server_tool_use", "tool_use"):
            head = {**block, "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block.get("input") or {}),
            }
        else:  # web_search_tool_result: whole in its start event, as Anthropic sends it
            head, delta = block, None
        events.append(
            (
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": head},
            )
        )
        if delta is not None:
            events.append(
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": delta},
                )
            )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {
                    "output_tokens": message["usage"]["output_tokens"],
                    "server_tool_use": message["usage"]["server_tool_use"],
                },
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return events

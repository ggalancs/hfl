# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Dialect-agnostic core of a chat completion.

Every chat surface (OpenAI ``/v1/chat/completions``, Ollama ``/api/chat``,
Anthropic) has to make the same decision once the engine has produced text:
did the model emit a tool call, and if so what is it? That decision is the
spot where the OpenAI route once drifted behind the Ollama route and silently
dropped tool calling. Centralising it here means a route can only differ in
its wire-format translation, never in this logic.

The output is the *canonical* tool-call shape
(``{"function": {"name": str, "arguments": dict}}``); each route maps it to
its own wire format (OpenAI wants ``arguments`` as a JSON string with an
``id``/``type``; Ollama wants the dict as-is).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hfl.api.tool_parsers import dispatch as parse_tool_calls


@dataclass(frozen=True)
class ChatOutput:
    """Resolved assistant turn, independent of API dialect.

    ``content`` is the cleaned narration text, or ``""`` when the turn is a
    tool call (spec rule C4). ``tool_calls`` is the canonical list (empty when
    none — spec rule C7).
    """

    content: str
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def resolve_chat_output(
    raw_text: str,
    model_name: str,
    tools: list[dict] | None,
    engine_tool_calls: list[dict] | None = None,
    *,
    tools_disabled: bool = False,
) -> ChatOutput:
    """Turn raw engine output into a ``ChatOutput``.

    - ``tools_disabled`` (the client sent ``tool_choice: "none"``): never scan
      for markers; the text is returned verbatim. This is required, not just
      cosmetic — the per-family parser fires on a ``<tool_call>`` marker
      regardless of the tools list, so suppression has to be explicit.
    - Otherwise prefer the engine's structured ``tool_calls`` when present,
      then fall back to parsing markers out of the text. ``engine_tool_calls``
      is only trusted when it's a real non-empty list (a test ``MagicMock``
      auto-attr must not masquerade as data).
    """
    if tools_disabled:
        return ChatOutput(content=raw_text)
    if isinstance(engine_tool_calls, list) and engine_tool_calls:
        return ChatOutput(content="", tool_calls=engine_tool_calls)
    cleaned, parsed_calls = parse_tool_calls(raw_text, model_name, tools)
    if parsed_calls:
        return ChatOutput(content="", tool_calls=parsed_calls)
    return ChatOutput(content=cleaned)

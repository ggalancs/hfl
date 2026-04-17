# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Reasoning / thinking channel extraction (Phase 5, P1-1).

When a client sets ``think=True`` on an Ollama request, the engine
produces text that includes the model's native reasoning channel
markers (``<think>...</think>``, Gemma 4's
``<|channel>thought...<channel|>``, DeepSeek-R1's ``<think>...
</think>``, etc.). Ollama's 2026 response envelope exposes two
separate fields:

    {
        "message": {
            "role": "assistant",
            "content": "the final answer",
            "thinking": "the reasoning"
        }
    }

This module extracts the thinking from a raw text blob so the route
layer can populate both fields. The extractor is format-permissive —
it recognises every thinking-channel dialect HFL knows about so
clients get consistent behaviour regardless of which model family
produced the output.
"""

from __future__ import annotations

import re

# Patterns we recognise as "reasoning" blocks. Order matters: the
# more specific patterns first so e.g. Gemma 4's split-pipe form
# doesn't get mis-matched as a plain ``<think>`` block.
_REASONING_PATTERNS: list[re.Pattern[str]] = [
    # Gemma 4: ``<|channel>thought...<channel|>`` (split-pipe markers).
    re.compile(r"<\|channel>thought(.*?)<channel\|>", re.DOTALL),
    # Gemma 4 alt: ``<|think>...<think|>``.
    re.compile(r"<\|think>(.*?)<think\|>", re.DOTALL),
    # DeepSeek-R1 / Qwen3-Thinking: XML-style ``<think>...</think>``.
    re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    # Some OSS models use ``<thinking>...</thinking>``.
    re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE),
    # OpenAI o1-style reasoning tags.
    re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE),
]

# Generic cleanup — Gemma 4 also ships orphan channel/turn markers
# alongside the reasoning content. These strip only the markers
# (not their content), complementing the full-block extractors above.
_GEMMA4_ORPHAN_OPEN = re.compile(r"<\|(?:channel|turn)>[a-z_]*\n?")
_GEMMA4_ORPHAN_CLOSE = re.compile(r"<(?:channel|turn|think)\|>")


def extract_thinking(text: str) -> tuple[str, str | None]:
    """Split raw model output into ``(clean_content, thinking)``.

    Walks every known reasoning-channel pattern and pulls out their
    bodies. Multiple reasoning blocks are concatenated with a blank
    line so the caller receives a single thinking string. When the
    text contains no recognised reasoning markers the function
    returns ``(text, None)`` unchanged.

    Args:
        text: Raw engine output, possibly with reasoning markers.

    Returns:
        Tuple where the first element is the text with all reasoning
        blocks + orphan markers stripped, and the second is either
        the concatenated reasoning text or ``None`` when no blocks
        were found.
    """
    if not text:
        return text, None

    thinking_parts: list[str] = []
    cleaned = text

    for pattern in _REASONING_PATTERNS:
        for match in pattern.finditer(cleaned):
            body = match.group(1).strip()
            if body:
                thinking_parts.append(body)
        cleaned = pattern.sub("", cleaned)

    # Strip any remaining orphan channel/turn markers (content kept).
    cleaned = _GEMMA4_ORPHAN_OPEN.sub("", cleaned)
    cleaned = _GEMMA4_ORPHAN_CLOSE.sub("", cleaned)
    cleaned = cleaned.strip()

    if not thinking_parts:
        return cleaned, None

    thinking = "\n\n".join(thinking_parts)
    return cleaned, thinking

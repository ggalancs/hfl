# SPDX-License-Identifier: Apache-2.0
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
    # gpt-oss (Harmony): the ``analysis`` channel, up to its ``<|end|>``.
    re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.DOTALL),
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
# Harmony's own markers around the answer (tool-call ones are the tool
# parser's business and are not reasoning).
_HARMONY_ORPHANS = re.compile(
    r"<\|start\|>assistant|<\|channel\|>final<\|message\|>|<\|end\|>|<\|return\|>"
)


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
    cleaned = _HARMONY_ORPHANS.sub("", cleaned)
    cleaned = cleaned.strip()

    if not thinking_parts:
        return cleaned, None

    thinking = "\n\n".join(thinking_parts)
    return cleaned, thinking


# Markers the streaming splitter acts on: reasoning opens, reasoning closes,
# and channel markers around the answer that are simply dropped.
_STREAM_OPEN = (
    "<think>",
    "<thinking>",
    "<|channel|>analysis<|message|>",  # gpt-oss (Harmony)
    "<|channel>thought",  # Gemma 4
    "<|think>",
)
_STREAM_CLOSE = ("</think>", "</thinking>", "<|end|>", "<channel|>", "<think|>")
_STREAM_DROP = (
    "<|start|>assistant",
    "<|channel|>final<|message|>",
    "<|return|>",
    "<|channel>",
    "<|turn>",
    "<turn|>",
)
_STREAM_MARKERS = sorted(
    [(m, "open") for m in _STREAM_OPEN]
    + [(m, "close") for m in _STREAM_CLOSE]
    + [(m, "drop") for m in _STREAM_DROP],
    key=lambda pair: -len(pair[0]),
)


class ThinkingSplitter:
    """Split a streamed reply into answer and reasoning as it arrives —
    what ``extract_thinking`` does for a finished one, for ``think=true``
    streams (Ollama sends the reasoning in ``message.thinking``).

    Markers split across chunks are held back until they complete; each
    part's leading and trailing whitespace is dropped, as
    ``extract_thinking`` strips it (trailing whitespace is held until more
    text of the same part shows it was not trailing).
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._thinking = False
        self._started = {False: False, True: False}  # per part: any text yet
        self._spaces = {False: "", True: ""}  # per part: whitespace held back

    def _emit(self, text: str, out: dict[bool, list[str]]) -> None:
        part = self._thinking
        text = self._spaces[part] + text
        if not self._started[part]:
            text = text.lstrip()
        kept = text.rstrip()
        self._spaces[part] = text[len(kept) :]
        if kept:
            self._started[part] = True
            out[part].append(kept)

    def feed(self, chunk: str) -> tuple[str, str]:
        """``(answer, reasoning)`` safe to send for this chunk."""
        self._buffer += chunk
        out: dict[bool, list[str]] = {False: [], True: []}
        while self._buffer:
            cut = self._buffer.find("<")
            if cut != 0:
                plain = self._buffer if cut < 0 else self._buffer[:cut]
                self._emit(plain, out)
                self._buffer = self._buffer[len(plain) :]
                continue
            matched = next(
                (pair for pair in _STREAM_MARKERS if self._buffer.startswith(pair[0])), None
            )
            could_grow = any(
                len(self._buffer) < len(m) and m.startswith(self._buffer)
                for m, _ in _STREAM_MARKERS
            )
            if could_grow and (
                matched is None
                or any(
                    len(m) > len(matched[0]) and m.startswith(self._buffer)
                    for m, _ in _STREAM_MARKERS
                )
            ):
                break  # wait for the rest of a marker
            if matched is None:
                self._emit("<", out)
                self._buffer = self._buffer[1:]
                continue
            marker, kind = matched
            self._buffer = self._buffer[len(marker) :]
            if kind == "open":
                self._thinking = True
            elif kind == "close":
                self._thinking = False
        return "".join(out[False]), "".join(out[True])

    def flush(self) -> tuple[str, str]:
        """What is left at the end of the stream."""
        rest, self._buffer = self._buffer, ""
        out: dict[bool, list[str]] = {False: [], True: []}
        if rest:
            self._emit(rest, out)
        return "".join(out[False]), "".join(out[True])

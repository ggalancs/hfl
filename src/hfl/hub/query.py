# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Reading a search the way a person writes it.

The Hub's ``search=`` matches text inside repo ids. ``hfl search "coding
assistant 7b"`` sent that phrase as is, and the Hub answered with the five
repos whose names contain it — 7 downloads for the best — and none of the
coder models people use. A person means: a coding model, about 7B.

``parse`` splits a query into what it asks for — a task, a size, a format —
and the words that are left; ``hub_queries`` turns that into Hub searches.
Nothing is guessed silently: ``describe`` says how the query was read, and
a query with nothing to interpret is searched literally, as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class _Task:
    """A task: the words people use for it, what such models have in their
    repo ids (the Hub's search matches ids), and the Hub's pipeline tag when
    one names the task."""

    words: tuple[str, ...]
    names: tuple[str, ...]
    pipeline: str | None = None


_TASKS: dict[str, _Task] = {
    "code": _Task(
        ("code", "coding", "coder", "programming", "programmer", "developer"), ("coder", "code")
    ),
    "vision": _Task(
        ("vision", "image", "images", "multimodal", "vl", "vlm", "visual"),
        ("vl", "vision"),
        "image-text-to-text",
    ),
    "embedding": _Task(
        ("embedding", "embeddings", "embed", "retrieval", "rag"), ("embed",), "feature-extraction"
    ),
    "speech": _Task(("tts", "speech", "voice", "text-to-speech"), ("tts",), "text-to-speech"),
    "transcription": _Task(
        ("transcription", "transcribe", "asr", "stt", "whisper"),
        ("whisper",),
        "automatic-speech-recognition",
    ),
    "reasoning": _Task(("reasoning", "thinking", "reasoner"), ("r1", "thinking", "reasoning")),
    "math": _Task(("math", "maths", "mathematics"), ("math",)),
}
# Words that say nothing about which model: dropped, never searched.
_FILLER = frozenset(
    "a an the for to of with and or model models llm llms assistant assistants "
    "chat chatbot instruct best good great small local fast new latest".split()
)
_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*b$", re.IGNORECASE)


@dataclass
class Intent:
    """What a query asks for."""

    task: str | None = None
    size_b: float | None = None
    gguf: bool = False
    words: list[str] = field(default_factory=list)

    @property
    def interpreted(self) -> bool:
        """Whether anything beyond plain words was found."""
        return self.task is not None or self.size_b is not None or self.gguf

    def size_range(self) -> tuple[float, float] | None:
        """The parameter counts that match the size asked for: "7b" is 6–9B,
        the way people round (Qwen 7B, Llama 8B, Mistral 7B)."""
        if self.size_b is None:
            return None
        return (self.size_b * 0.75, self.size_b * 1.3)


def parse(query: str) -> Intent:
    """Split ``query`` into a task, a size, a format and the other words."""
    intent = Intent()
    for raw in re.split(r"[\s,]+", query.strip()):
        word = raw.strip("\"'").lower()
        if not word:
            continue
        size = _SIZE.match(word)
        if size and intent.size_b is None:
            intent.size_b = float(size.group(1))
            continue
        if word == "gguf":
            intent.gguf = True
            continue
        task = next((name for name, spec in _TASKS.items() if word in spec.words), None)
        if task is not None and intent.task is None:
            intent.task = task
            continue
        if word in _FILLER or task is not None:
            continue
        intent.words.append(raw.strip("\"'"))
    return intent


def hub_queries(intent: Intent) -> list[dict[str, str]]:
    """The Hub searches that answer ``intent``: one per name its task goes
    by (with the other words), each with the task's pipeline tag if any."""
    names = _TASKS[intent.task].names if intent.task else ()
    pipeline = _TASKS[intent.task].pipeline if intent.task else None
    extra = " ".join(intent.words)
    searches = [f"{extra} {name}".strip() for name in names] or ([extra] if extra else [])
    out: list[dict[str, str]] = []
    for text in searches or [""]:
        query: dict[str, str] = {}
        if text:
            query["search"] = text
        if pipeline:
            query["pipeline_tag"] = pipeline
        out.append(query)
    return out


def describe(intent: Intent) -> str:
    """How the query was read, for the user."""
    parts = []
    if intent.task:
        parts.append(f"{intent.task} models")
    if intent.size_b is not None:
        low, high = intent.size_range() or (0, 0)
        parts.append(f"{low:g}–{high:g}B parameters")
    if intent.gguf:
        parts.append("GGUF")
    if intent.words:
        parts.append("named like " + " ".join(intent.words))
    return ", ".join(parts)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A created model's Modelfile, applied to every request that uses it.

``hfl create`` / ``POST /api/create`` store SYSTEM, PARAMETER and MESSAGE on
the manifest, and ``show`` displayed them, but inference ignored all but
MESSAGE (and that on ``/api/chat`` only): a model made with ``SYSTEM "Answer
only in French."`` and ``PARAMETER temperature 0`` answered in English at the
default temperature (local audit A8/B8). Every API now applies them here,
with Ollama's precedence:

- SYSTEM is the default system prompt: used when the request has none.
- MESSAGE entries sit between the system prompt and the conversation.
- PARAMETER values are defaults: an option the request sets wins.

Each route says which options its request set explicitly (``explicit``), in
Modelfile names; everything else is filled from the Modelfile.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from hfl.engine.base import ChatMessage, GenerationConfig

# Modelfile PARAMETER name -> GenerationConfig attribute. ``num_ctx`` is not
# here: it sizes the context at load (the manifest's ``context_length``).
PARAMETERS: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "num_predict": "max_tokens",
    "repeat_penalty": "repeat_penalty",
    "seed": "seed",
    "stop": "stop",
}
_INTS = {"top_k", "num_predict", "seed"}


def _field(manifest: Any, name: str, kind: type) -> Any:
    """``manifest.name`` when it is a ``kind`` (a manifest-like stand-in's
    other attributes, e.g. a mock's, never count as Modelfile values)."""
    value = getattr(manifest, name, None)
    return value if isinstance(value, kind) else None


def explicit_options(options: Mapping[str, Any] | None) -> set[str]:
    """The Modelfile parameters an Ollama ``options`` dict sets."""
    return {k for k, v in (options or {}).items() if k in PARAMETERS and v is not None}


def explicit_fields(request: Any, fields: Mapping[str, str]) -> set[str]:
    """The Modelfile parameters a pydantic request set, by ``fields``
    (request field -> Modelfile parameter). A field counts only when the
    client sent it: defaults the schema filled in do not."""
    sent: set[str] = getattr(request, "model_fields_set", set())
    return {
        parameter
        for field, parameter in fields.items()
        if field in sent and getattr(request, field, None) is not None
    }


def apply_parameters(manifest: Any, config: GenerationConfig, explicit: Iterable[str]) -> None:
    """Fill ``config`` from the Modelfile's PARAMETER values the request left unset."""
    chosen = set(explicit)
    defaults = _field(manifest, "default_parameters", dict) or {}
    for name, value in defaults.items():
        attr = PARAMETERS.get(name)
        if attr is None or name in chosen or value is None:
            continue
        try:
            if name == "stop":
                value = [value] if isinstance(value, str) else [str(v) for v in value]
            elif name in _INTS:
                value = int(value)
            else:
                value = float(value)
        except (TypeError, ValueError):
            continue  # a value the parser let through but no engine could use
        setattr(config, attr, value)
        if name == "repeat_penalty":
            config.repeat_penalty_chosen = True


def baked_messages(manifest: Any) -> list[ChatMessage]:
    """The Modelfile's MESSAGE entries as ChatMessages (malformed ones skipped)."""
    out: list[ChatMessage] = []
    for entry in _field(manifest, "messages", list) or []:
        if not isinstance(entry, dict):
            continue
        role, content = entry.get("role"), entry.get("content")
        if isinstance(role, str) and isinstance(content, str):
            out.append(ChatMessage(role=role, content=content))
    return out


def splice_baked(messages: list[ChatMessage], baked: list[ChatMessage]) -> list[ChatMessage]:
    """``baked`` after the leading system messages, before the conversation."""
    if not baked:
        return messages
    split = next((i for i, m in enumerate(messages) if m.role != "system"), len(messages))
    return [*messages[:split], *baked, *messages[split:]]


def apply_to_chat(
    manifest: Any,
    messages: list[ChatMessage],
    config: GenerationConfig,
    explicit: Iterable[str],
) -> list[ChatMessage]:
    """SYSTEM, MESSAGE and PARAMETER applied to a chat request."""
    if manifest is None:
        return messages
    apply_parameters(manifest, config, explicit)
    system = _field(manifest, "system", str)
    if system and not any(m.role == "system" for m in messages):
        messages = [ChatMessage(role="system", content=system), *messages]
    return splice_baked(messages, baked_messages(manifest))


def default_system(manifest: Any, requested: str | None) -> str | None:
    """The system prompt for a completion request: the request's, else SYSTEM."""
    if requested:
        return requested
    system: str | None = _field(manifest, "system", str)
    return system or None


def default_template(manifest: Any, requested: str | None) -> str | None:
    """The completion template: the request's, else the Modelfile's TEMPLATE."""
    if requested is not None:
        return requested
    template: str | None = _field(manifest, "chat_template", str)
    return template

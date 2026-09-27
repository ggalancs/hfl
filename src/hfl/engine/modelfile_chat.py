# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A Modelfile ``TEMPLATE`` applied to a chat, as Ollama applies it.

``hfl create`` stored TEMPLATE and ``/api/generate`` used it, but a chat
ignored it: every chat API formatted the conversation with the model's own
template, so a model created to change its prompt format only changed it for
completions. Here the conversation is rendered with the TEMPLATE instead and
the engine completes that text (``InferenceEngine`` routes every engine's
``chat``/``chat_stream`` through :func:`templated_prompt`).

The data and the two modes follow Ollama's ``template.Execute``
(``template/template.go``): consecutive messages of one role are merged
(tool results are not), all system messages joined make ``.System``; a
template that reads ``.Messages`` is rendered once, with ``.Messages``,
``.Tools`` and the thinking fields; one that does not is rendered once per
turn with ``.System``/``.Prompt``/``.Response``, the last turn cut after
``.Response`` so the prompt ends where the answer goes.

Not applied — the model's own template is used, and the log says why — when
a message carries images (a text prompt cannot), or the template cannot be
rendered here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any

from hfl.converter.go_template import GoStruct, GoTemplateError, render_strict
from hfl.engine.base import ChatMessage, GenerationConfig

logger = logging.getLogger(__name__)

_MESSAGES = re.compile(r"\.Messages\b", re.IGNORECASE)


def is_go_template(template: str | None) -> bool:
    """A Modelfile TEMPLATE in Go syntax (a Jinja one has ``{%``)."""
    return bool(template) and "{{" in template and "{%" not in template  # type: ignore[operator]


def _collate(messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    system: list[str] = []
    collated: list[ChatMessage] = []
    for message in messages:
        if message.role == "system":
            system.append(message.content)
        last = collated[-1] if collated else None
        if last is not None and last.role == message.role and message.role != "tool":
            collated[-1] = replace(last, content=last.content + "\n\n" + message.content)
        else:
            collated.append(message)
    return "\n\n".join(system), collated


def _tool_call(index: int, call: dict) -> GoStruct:
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    return GoStruct(
        {
            "ID": call.get("id", ""),
            "Function": GoStruct(
                {
                    "Index": function.get("index", index),
                    "Name": function.get("name", ""),
                    "Arguments": arguments,
                },
                wire={
                    "index": function.get("index", index),
                    "name": function.get("name", ""),
                    "arguments": arguments,
                },
            ),
        },
        wire=call,
    )


def _message(message: ChatMessage) -> GoStruct:
    fields = {
        "Role": message.role,
        "Content": message.content,
        "Thinking": "",
        "Images": [],
        "ToolCalls": [_tool_call(i, c) for i, c in enumerate(message.tool_calls or [])],
        "ToolName": message.name or "",
        "ToolCallID": message.tool_call_id or "",
    }
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = message.tool_calls
    return GoStruct(fields, wire=wire)


def _property(prop: Any) -> Any:
    """A JSON-schema property as Ollama's ``ToolProperty``: Go field names
    (``.Type``, ``.Description``), a type list printed as Go prints it."""
    if not isinstance(prop, dict):
        return prop
    fields: dict[str, Any] = {}
    for key, value in prop.items():
        name = key[:1].upper() + key[1:]
        if key == "type" and isinstance(value, list):
            value = value[0] if len(value) == 1 else "[" + " ".join(map(str, value)) + "]"
        elif key == "properties" and isinstance(value, dict):
            value = {k: _property(v) for k, v in value.items()}
        elif key == "items":
            value = _property(value)
        fields[name] = value
    return GoStruct(fields, wire=prop)


def _tool(tool: dict) -> GoStruct:
    """Ollama's ``templateTool``: Go field names to read, its JSON tags'
    shape (and order) for ``json``."""
    function = tool.get("function") or {}
    params = function.get("parameters") or {}
    wire_params: dict[str, Any] = {"type": params.get("type", "")}
    for key in ("$defs", "items", "required"):
        if params.get(key):
            wire_params[key] = params[key]
    wire_params["properties"] = params.get("properties") or {}
    wire_function = {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": wire_params,
    }
    wire: dict[str, Any] = {"type": tool.get("type", "function")}
    if tool.get("items"):
        wire["items"] = tool["items"]
    wire["function"] = wire_function
    return GoStruct(
        {
            "Type": wire["type"],
            "Items": tool.get("items"),
            "Function": GoStruct(
                {
                    "Name": wire_function["name"],
                    "Description": wire_function["description"],
                    "Parameters": GoStruct(
                        {
                            "Type": wire_params["type"],
                            "Defs": params.get("$defs"),
                            "Items": params.get("items"),
                            "Required": params.get("required") or [],
                            "Properties": {
                                k: _property(v) for k, v in wire_params["properties"].items()
                            },
                        },
                        wire=wire_params,
                    ),
                },
                wire=wire_function,
            ),
        },
        wire=wire,
    )


def render_chat(
    template: str,
    messages: list[ChatMessage],
    tools: list[dict] | None = None,
    reasoning: str | None = None,
) -> str:
    """The prompt ``template`` makes of this conversation (GoTemplateError
    when it cannot be rendered)."""
    think = reasoning is not None and reasoning != "off"
    thinking = {
        "Think": think,
        "ThinkLevel": reasoning if think else "",
        "IsThinkSet": reasoning is not None,
    }
    system, collated = _collate(messages)
    if _MESSAGES.search(template):
        return render_strict(
            template,
            {
                "System": system,
                "Messages": [_message(m) for m in collated],
                "Tools": [_tool(t) for t in tools] if tools else None,
                "Response": "",
                **thinking,
            },
        )
    out: list[str] = []
    turn = {"System": "", "Prompt": "", "Response": ""}

    def execute() -> None:
        out.append(render_strict(template, {**turn, **thinking}))
        turn.update(System="", Prompt="", Response="")

    for message in collated:
        if message.role == "system":
            if turn["Prompt"] or turn["Response"]:
                execute()
            turn["System"] = message.content
        elif message.role == "user":
            if turn["Response"]:
                execute()
            turn["Prompt"] = message.content
        elif message.role == "assistant":
            turn["Response"] = message.content
    out.append(render_strict(template, {**turn, **thinking}, cut_after_response=True))
    return "".join(out)


def templated_prompt(
    messages: list[ChatMessage], config: GenerationConfig | None, tools: list[dict] | None
) -> tuple[str, GenerationConfig] | None:
    """(prompt, config for a plain completion) when this chat goes through
    the Modelfile's TEMPLATE; None to chat as usual."""
    template = config.modelfile_template if config is not None else None
    if config is None or not template or config.raw:
        return None
    if any(m.images for m in messages):
        logger.info("Modelfile TEMPLATE not applied: a message carries images")
        return None
    try:
        prompt = render_chat(template, messages, tools, config.reasoning)
    except GoTemplateError as exc:
        logger.warning("Modelfile TEMPLATE not applied, the model's own is used: %s", exc)
        return None
    # A completion of that exact text: no second template on top.
    return prompt, replace(config, modelfile_template=None, template_override=None, raw=False)

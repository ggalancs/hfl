# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A Modelfile TEMPLATE formats the chat, as Ollama does (plan 0.22 P1-7).

It used to apply to ``/api/generate`` only. The expected prompts below were
produced by Ollama v0.34.3's own ``template`` package from these templates
and conversations; the same comparison over all 20 templates Ollama ships,
six conversations and four ``think`` values gave 504/504 identical prompts."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from hfl.converter.go_template import GoTemplateError, render_strict
from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult, InferenceEngine
from hfl.engine.modelfile_chat import is_go_template, render_chat

MESSAGES = (
    "{{- if .System }}<sys>{{ .System }}</sys>\n{{ end }}"
    "{{- if .Tools }}<tools>{{ json .Tools }}</tools>\n{{ end }}"
    "{{- range $i, $m := .Messages }}"
    "{{- $last := eq (len (slice $.Messages $i)) 1 }}"
    '{{- if eq .Role "system" }}{{ continue }}{{ end }}'
    "<{{ .Role }}>{{ .Content }}"
    '{{- range .ToolCalls }}<call>{{ printf "%s %v" .Function.Name .Function.Arguments }}</call>'
    "{{ end }}"
    '{{- if and $last (ne .Role "assistant") }}\n<assistant>{{ if $.IsThinkSet }}'
    "{{ if $.Think }}<think>{{ else }}<nothink>{{ end }}{{ end }}{{ end }}\n"
    "{{- end }}"
)
LEGACY = (
    "{{ if .System }}### System\n{{ .System }}\n\n{{ end }}"
    "### User\n{{ .Prompt }}\n\n### Bot\n{{ .Response }}</s>\n"
)
PROPS = (
    "{{ range .Tools }}{{ .Function.Name }}({{ range $n, $p := .Function.Parameters.Properties }}"
    "{{ $n }}: {{ $p.Type }} — {{ $p.Description }}; {{ end }}){{ end }}|"
    "{{ range .Messages }}{{ .Content }}{{ end }}"
)
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather <now>",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string", "description": "City"},
                "days": {"type": ["integer", "null"], "description": "Days"},
            },
        },
    },
}
CALL = {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}


def _m(role: str, content: str, **extra) -> ChatMessage:
    return ChatMessage(role=role, content=content, **extra)


TURNS = [
    _m("system", "Be brief."),
    _m("user", "hi"),
    _m("assistant", "hello"),
    _m("user", "and"),
    _m("user", "again"),
]
TOOLS = [
    _m("user", "weather?"),
    _m("assistant", "", tool_calls=[CALL]),
    _m("tool", "18C", name="get_weather"),
    _m("user", "thanks"),
]
MIDSYS = [_m("user", "u1"), _m("assistant", "a1"), _m("system", "S2"), _m("user", "u2")]


@pytest.mark.parametrize(
    ("template", "messages", "tools", "reasoning", "ollama"),
    [
        (
            MESSAGES,
            TURNS,
            None,
            None,
            "<sys>Be brief.</sys>\n<user>hi<assistant>hello<user>and\n\nagain\n<assistant>",
        ),
        (
            MESSAGES,
            TURNS,
            None,
            "off",
            "<sys>Be brief.</sys>\n<user>hi<assistant>hello<user>and\n\nagain\n"
            "<assistant><nothink>",
        ),
        (
            MESSAGES,
            TOOLS,
            [TOOL],
            "high",
            '<tools>[{"type":"function","function":{"name":"get_weather","description":'
            '"Weather \\u003cnow\\u003e","parameters":{"type":"object","required":["city"],'
            '"properties":{"city":{"type":"string","description":"City"},"days":{"type":'
            '["integer","null"],"description":"Days"}}}}}]</tools>\n<user>weather?<assistant>'
            '<call>get_weather {"city":"Paris"}</call><tool>18C<user>thanks\n<assistant><think>',
        ),
        (
            LEGACY,
            TURNS,
            None,
            None,
            "### System\nBe brief.\n\n### User\nhi\n\n### Bot\nhello</s>\n### User\nand\n\nagain"
            "\n\n### Bot\n",
        ),
        (
            LEGACY,
            MIDSYS,
            None,
            None,
            "### User\nu1\n\n### Bot\na1</s>\n### System\nS2\n\n### User\nu2\n\n### Bot\n",
        ),
        (
            PROPS,
            TOOLS,
            [TOOL],
            None,
            "get_weather(city: string — City; days: [integer null] — Days; )|weather?18Cthanks",
        ),
    ],
    ids=["messages", "think-off", "tools-think", "legacy", "legacy-midsys", "tool-properties"],
)
def test_the_prompt_ollama_makes(template, messages, tools, reasoning, ollama) -> None:
    assert render_chat(template, messages, tools, reasoning) == ollama


def test_the_renderer_handles_go_only_syntax() -> None:
    data = {"A": "", "B": "b", "L": [1, 2, 3]}
    assert render_strict('{{ $x := "a" }}{{ if .B }}{{ $x = "z" }}{{ end }}{{ $x }}', data) == "z"
    assert render_strict("{{ with .A }}{{ . }}{{ else }}none{{ end }}", data) == "none"
    assert render_strict("{{ if .A }}1{{ else if .B }}2{{ else }}3{{ end }}", data) == "2"
    assert (
        render_strict("{{ range .L }}{{ if eq . 2 }}{{ break }}{{ end }}{{ . }}{{ end }}", data)
        == "1"
    )
    assert render_strict('{{ .B | printf "<%s>" }}{{/* gone */}}', data) == "<b>"
    assert render_strict("{{ false }} {{ not .A }} {{ len .L }}", data) == "false true 3"


@pytest.mark.parametrize(
    "template",
    ["{{ if .A }}", "{{ undefinedfn .A }}", "{{ $nope }}", "{{ break }}", "{{ index .L 9 }}"],
)
def test_what_it_cannot_render_raises(template) -> None:
    with pytest.raises(GoTemplateError):
        render_strict(template, {"A": 1, "L": [1]})


def test_output_is_bounded() -> None:
    nested = "{{ range .L }}{{ range $.L }}{{ range $.L }}{{ $.S }}{{ end }}{{ end }}{{ end }}"
    with pytest.raises(GoTemplateError, match="too large"):
        render_strict(nested, {"L": list(range(200)), "S": "x" * 10})


def test_go_or_jinja() -> None:
    assert is_go_template("{{ .Prompt }}")
    assert not is_go_template("{% for m in messages %}{{ m.content }}{% endfor %}")
    assert not is_go_template("plain text")


class _Engine(InferenceEngine):
    """Records what reaches the model."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, object]] = []

    def load(self, model_path: str, **kwargs) -> None: ...

    def unload(self) -> None: ...

    def generate(self, prompt, config=None) -> GenerationResult:
        self.seen.append(("generate", prompt, config))
        return GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1)

    def generate_stream(self, prompt, config=None) -> Iterator[str]:
        self.seen.append(("generate_stream", prompt, config))
        yield "ok"

    def chat(self, messages, config=None, tools=None) -> GenerationResult:
        self.seen.append(("chat", messages, config))
        return GenerationResult(text="own", tokens_generated=1, tokens_prompt=1)

    def chat_stream(self, messages, config=None, tools=None) -> Iterator[str]:
        self.seen.append(("chat_stream", messages, config))
        yield "own"

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def is_loaded(self) -> bool:
        return True


def test_every_engines_chat_goes_through_the_template() -> None:
    engine = _Engine()
    cfg = GenerationConfig(modelfile_template=LEGACY)
    engine.chat([_m("user", "hi")], cfg)
    list(engine.chat_stream([_m("user", "hi")], cfg))
    (kind1, prompt1, cfg1), (kind2, prompt2, _) = engine.seen
    assert (kind1, kind2) == ("generate", "generate_stream")
    assert prompt1 == prompt2 == "### User\nhi\n\n### Bot\n"
    assert cfg1.modelfile_template is None and cfg1.template_override is None  # not twice


@pytest.mark.parametrize(
    ("messages", "template"),
    [
        ([_m("user", "hi", images=[b"\x89PNG"])], LEGACY),  # a text prompt cannot carry it
        ([_m("user", "hi")], "{{ if .Prompt }}unclosed"),  # the model's own, not a broken prompt
    ],
    ids=["images", "unrenderable"],
)
def test_otherwise_the_models_own(messages, template) -> None:
    engine = _Engine()
    engine.chat(messages, GenerationConfig(modelfile_template=template))
    assert engine.seen[0][0] == "chat"


def test_without_one_nothing_changes() -> None:
    engine = _Engine()
    engine.chat([_m("user", "hi")], GenerationConfig())
    assert engine.seen[0][0] == "chat"


def test_a_created_models_template_reaches_the_chat() -> None:
    from types import SimpleNamespace

    from hfl.api.modelfile_defaults import apply_to_chat

    go, jinja = GenerationConfig(), GenerationConfig()
    apply_to_chat(SimpleNamespace(chat_template=LEGACY), [_m("user", "hi")], go, ())
    apply_to_chat(SimpleNamespace(chat_template="{% if x %}{% endif %}"), [], jinja, ())
    assert go.modelfile_template == LEGACY and jinja.modelfile_template is None


THINKY = (
    "{{- if .IsThinkSet }}[set:{{ .Think }}:{{ .ThinkLevel }}]{{ end }}"
    "{{ range .Messages }}<{{ .Role }}>{{ .Content }}{{ end }}<assistant>"
)


@pytest.mark.parametrize(
    ("reasoning", "from_bool", "ollama"),
    [
        ("medium", True, "[set:true:]<user>hi<assistant>"),  # think: true names no level
        ("off", True, "[set:false:]<user>hi<assistant>"),
        ("high", False, "[set:true:high]<user>hi<assistant>"),
        (None, False, "<user>hi<assistant>"),
    ],
)
def test_the_thinking_fields_as_ollama_gives_them(reasoning, from_bool, ollama) -> None:
    assert render_chat(THINKY, [_m("user", "hi")], None, reasoning, from_bool) == ollama


def test_the_native_route_says_when_think_was_a_boolean(temp_config) -> None:
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from hfl.api.server import app
    from hfl.api.state import get_state, reset_state
    from hfl.models.manifest import ModelManifest

    reset_state()
    engine = MagicMock(is_loaded=True)
    engine.chat = MagicMock(
        return_value=GenerationResult(text="ok", tokens_generated=1, tokens_prompt=1)
    )
    state = get_state()
    state.engine = engine
    state.current_model = ModelManifest(name="m", repo_id="o/m", local_path="/m", format="gguf")
    client = TestClient(app)
    seen = []
    for think in (True, "high"):
        body = {
            "model": "m",
            "stream": False,
            "think": think,
            "messages": [{"role": "user", "content": "q"}],
        }
        client.post("/api/chat", json=body)
        seen.append(engine.chat.call_args[0][1].reasoning_from_bool)
    assert seen == [True, False]
    reset_state()

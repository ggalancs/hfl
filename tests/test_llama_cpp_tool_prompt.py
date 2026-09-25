# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""What the default GGUF engine does so a model can call a tool at all.

Each rule comes from a real model on this engine: Hermes-3 and DeepSeek-R1
never saw the tools (their templates drop them); Hermes-3 got no BOS and its
``<tool_call>`` token was dropped from the text; GLM-4-0414 never saw the
tool's result, and went on past ``<|observation|>``; gpt-oss showed its
reasoning channel as the answer.
"""

from __future__ import annotations

import importlib.util
import json
from unittest.mock import MagicMock

import pytest

from hfl.engine.base import ChatMessage, GenerationConfig
from hfl.engine.llama_cpp import (
    _filter_gemma4_stream,
    _history_for_template,
    _render_special_tokens,
    _strip_harmony_channels,
    _template_renders_tools,
    _tools_as_text,
)

_HAS_LLAMA_CPP = importlib.util.find_spec("llama_cpp") is not None

WEATHER = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
CALL = {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}


@pytest.mark.parametrize(
    ("template", "chat_format", "knows"),
    [
        ("{% for tool in tools %}{{ tool }}{% endfor %}", None, True),
        ("{% if tools %}...{% endif %}", None, True),
        # Hermes-3's plain ChatML; DeepSeek's renders past calls, not the list.
        ("{% for message in messages %}{{ message['content'] }}{% endfor %}", None, False),
        ("{% if message['tool_calls'] %}...{% endif %}", None, False),
        ("", "chatml", False),
        ("", "chatml-function-calling", True),
    ],
)
def test_which_templates_list_the_tools(template, chat_format, knows):
    assert _template_renders_tools(template, chat_format) is knows


class TestToolsAsText:
    def test_the_tools_go_into_the_system_message(self):
        out = _tools_as_text(
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
            WEATHER,
        )
        assert out[0]["role"] == "system"
        assert out[0]["content"].startswith("Be brief.\n\n")
        # Hermes' own prompt, word for word: a paraphrase was measured to fail.
        assert "You are a function calling AI model." in out[0]["content"]
        assert json.dumps(WEATHER[0]) in out[0]["content"]
        assert out[1] == {"role": "user", "content": "hi"}

    def test_without_a_system_message_one_is_added(self):
        out = _tools_as_text([{"role": "user", "content": "hi"}], WEATHER)
        assert [m["role"] for m in out] == ["system", "user"]

    def test_past_calls_and_results_become_text(self):
        """Consecutive results are one user turn: strict templates demand
        alternating turns."""
        out = _tools_as_text(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [CALL, CALL]},
                {"role": "tool", "content": "31C"},
                {"role": "tool", "content": "rain"},
            ],
            None,
        )
        assert [m["role"] for m in out] == ["user", "assistant", "user"]  # no tools: no prompt
        assert out[1]["content"].count("<tool_call>") == 2 and "tool_calls" not in out[1]
        assert json.loads(out[1]["content"].split("\n")[1]) == {
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }
        assert out[2]["content"] == (
            "<tool_response>\n31C\n</tool_response>\n<tool_response>\nrain\n</tool_response>"
        )

    def test_string_arguments_are_written_as_json(self):
        call = {"function": {"name": "f", "arguments": '{"a": 1}'}}
        out = _tools_as_text([{"role": "assistant", "content": "", "tool_calls": [call]}], None)
        assert '"arguments": {"a": 1}' in out[0]["content"]


# GLM-4-0414's template, the parts that matter: a call is an assistant turn
# with ``metadata``, a result an ``observation`` turn.
GLM4 = (
    '{%- set meta = message.get("metadata", "") %}'
    "{%- elif role == 'assistant' and meta %}<|assistant|>{{ meta }}"
    "{%- elif role == 'observation' %}<|observation|>"
)


def test_glm4_history_in_the_shape_its_template_renders():
    out = _history_for_template(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "content": "31C"},
        ],
        GLM4,
    )
    assert out[1] == {
        "role": "assistant",
        "metadata": "get_weather",
        "content": '{"city": "Paris"}',
    }
    assert out[2] == {"role": "observation", "content": "31C"}


def test_other_templates_keep_the_history():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [CALL]}]
    assert _history_for_template(msgs, "{{ tools }}") is msgs


class _Model:
    """Enough of ``llama_cpp.Llama`` for the special-token switch."""

    def detokenize(self, tokens, prev_tokens=None, special=False):
        return b"<tool_call>" if special else b""


def test_special_tokens_are_kept_only_while_asked():
    model = _Model()
    _render_special_tokens(model, True)
    assert model.detokenize([1]) == b"<tool_call>"
    _render_special_tokens(model, False)
    assert model.detokenize([1]) == b""
    _render_special_tokens(model, False)  # already off: nothing to undo
    assert "detokenize" not in vars(model)


class TestHarmony:
    TEXT = (
        "<|channel|>analysis<|message|>They greet.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello!"
    )
    CALL = (
        "<|channel|>analysis<|message|>Need the tool.<|end|><|start|>assistant"
        '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"P"}'
    )

    def test_the_answer_is_the_final_channel(self):
        assert _strip_harmony_channels(self.TEXT) == "Hello!"

    def test_a_call_is_left_for_the_parser(self):
        assert _strip_harmony_channels(self.CALL) == (
            "<|channel|>commentary to=functions.get_weather <|constrain|>json"
            '<|message|>{"city":"P"}'
        )

    @pytest.mark.parametrize("size", [1, 2, 3, 7])
    def test_a_stream_split_anywhere(self, size):
        chunks = [self.TEXT[i : i + size] for i in range(0, len(self.TEXT), size)]
        assert "".join(_filter_gemma4_stream(iter(chunks), harmony=True)) == "Hello!"


@pytest.mark.skipif(not _HAS_LLAMA_CPP, reason="llama-cpp-python not installed ([llama] extra)")
class TestEngine:
    def _engine(self, *, knows_tools=True, template="", architecture=None, text="Hi"):
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        seen: dict = {}

        class Model(_Model):
            def create_chat_completion(self, **kwargs):
                seen.update(kwargs, special=self.detokenize([1]))
                if kwargs.get("stream"):
                    return iter([{"choices": [{"delta": {"content": text}}]}])
                return {"choices": [{"message": {"content": text}}], "usage": {}}

        engine._model = Model()
        engine._model_path = "/fake/m.gguf"
        engine._template_knows_tools = knows_tools
        engine._chat_template = template
        engine._architecture = architecture
        return engine, seen

    def test_a_template_that_lists_tools_gets_them(self):
        engine, seen = self._engine()
        engine.chat([ChatMessage(role="user", content="w?")], tools=WEATHER)
        assert seen["tools"] == WEATHER and seen["messages"][0]["role"] == "user"

    def test_a_template_without_tools_gets_them_written_in(self):
        engine, seen = self._engine(knows_tools=False)
        engine.chat([ChatMessage(role="user", content="w?")], tools=WEATHER)
        assert "tools" not in seen
        assert "<tools>" in seen["messages"][0]["content"]

    def test_glm4_template_gets_its_history_shape(self):
        engine, seen = self._engine(template=GLM4)
        engine.chat(
            [
                ChatMessage(role="user", content="q"),
                ChatMessage(role="assistant", content="", tool_calls=[CALL]),
                ChatMessage(role="tool", content="31C", name="get_weather"),
            ],
            tools=WEATHER,
        )
        assert seen["messages"][2] == {"role": "observation", "content": "31C"}

    def test_the_markers_reach_the_text_only_with_tools(self):
        engine, seen = self._engine()
        engine.chat([ChatMessage(role="user", content="w?")], tools=WEATHER)
        assert seen["special"] == b"<tool_call>"
        assert engine._model.detokenize([1]) == b""  # and only during the call
        engine.chat([ChatMessage(role="user", content="hi")])
        assert seen["special"] == b""

    def test_an_abandoned_stream_does_not_leave_them_on(self):
        engine, seen = self._engine()
        stream = engine.chat_stream([ChatMessage(role="user", content="w?")], tools=WEATHER)
        next(iter(stream))
        assert engine._model.detokenize([1]) == b"<tool_call>"  # on while streaming
        engine.chat([ChatMessage(role="user", content="hi")])  # the next request
        assert seen["special"] == b""

    def test_a_finished_stream_turns_them_off(self):
        engine, _ = self._engine()
        list(engine.chat_stream([ChatMessage(role="user", content="w?")], tools=WEATHER))
        assert engine._model.detokenize([1]) == b""

    def test_glm_stops_where_it_hands_over_to_the_tool(self):
        engine, seen = self._engine()
        engine.chat([ChatMessage(role="user", content="w?")], tools=WEATHER)
        assert "<|observation|>" in seen["stop"]
        engine.chat([ChatMessage(role="user", content="hi")])
        assert "<|observation|>" not in seen["stop"]

    def test_no_repeat_penalty_on_a_tool_turn_unless_asked(self):
        engine, seen = self._engine()
        engine.chat([ChatMessage(role="user", content="w?")], tools=WEATHER)
        assert seen["repeat_penalty"] == 1.0
        engine.chat([ChatMessage(role="user", content="hi")])
        assert seen["repeat_penalty"] == 1.1
        chosen = GenerationConfig(repeat_penalty=1.3, repeat_penalty_chosen=True)
        engine.chat([ChatMessage(role="user", content="w?")], chosen, tools=WEATHER)
        assert seen["repeat_penalty"] == 1.3

    def test_gpt_oss_shows_the_answer_not_its_reasoning(self):
        engine, _ = self._engine(architecture="gpt-oss", text=TestHarmony.TEXT)
        assert engine.chat([ChatMessage(role="user", content="hi")]).text == "Hello!"
        stream = engine.chat_stream([ChatMessage(role="user", content="hi")])
        assert "".join(stream) == "Hello!"
        raw = GenerationConfig(expose_reasoning=True)
        assert "They greet." in engine.chat([ChatMessage(role="user", content="hi")], raw).text

    def test_bos_is_given_to_a_template_that_forgets_it(self, monkeypatch):
        from llama_cpp import llama_chat_format

        from hfl.engine.llama_cpp import _give_templates_a_bos

        made: list[str] = []

        class Formatter:
            def __init__(self, template, **kw):
                made.append(template)

            def to_chat_handler(self):
                return "handler"

        monkeypatch.setattr(llama_chat_format, "Jinja2ChatFormatter", Formatter)
        model = MagicMock()
        model._model.add_bos_token.return_value = True
        model._model.token_bos.return_value = 1
        model._model.token_eos.return_value = 2
        model._model.token_get_text.side_effect = {1: "<s>", 2: "</s>"}.get
        model._chat_handlers = {}
        model.metadata = {"tokenizer.chat_template": "{{ x }}"}
        assert _give_templates_a_bos(model) is True
        assert made == ["{{ bos_token }}{{ x }}"]
        assert model._chat_handlers == {"chat_template.default": "handler"}
        # Written by the template already, or not wanted: left alone.
        for template in ("{{ bos_token }}{{ x }}", "<s>{{ x }}"):
            model.metadata = {"tokenizer.chat_template": template}
            assert _give_templates_a_bos(model) is False
        model._model.add_bos_token.return_value = False
        model.metadata = {"tokenizer.chat_template": "{{ x }}"}
        assert _give_templates_a_bos(model) is False
        assert len(made) == 1

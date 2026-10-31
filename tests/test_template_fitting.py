# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The messages fitted to the model's chat template, for every engine.

Each engine did part of it, and the gaps failed silently: MLX swapped a
Mistral template that refused two user turns in a row for an invented
format (a DEBUG line); SmolVLM's template drew no text from a plain-string
message, and the model answered a question it never got; Transformers
crashed on a model with no template and handed tools to templates that
never show them. Plus the two silences around them: images an engine
cannot see, and replies of no tokens.
"""

from __future__ import annotations

import importlib.util
import logging

import pytest

from hfl.engine import llama_cpp as lc
from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult, InferenceEngine

# Mistral's shape: roles must alternate from a user turn, and no system role.
ALTERNATING = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if (m['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if m['role'] == 'user' %}[INST] {{ m['content'] }} [/INST]"
    "{% else %}{{ m['content'] }}</s>{% endif %}{% endfor %}"
)
# SmolVLM-256M-Instruct's, from its GGUF: typed parts only.
TYPED_ONLY = (
    "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}"
    "{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}"
    "{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}"
    "{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}"
    "<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)
# Rendering a template needs jinja2: not a core dependency (it comes with
# every backend that renders one in-process), so not in the CI venv.
needs_jinja2 = pytest.mark.skipif(
    importlib.util.find_spec("jinja2") is None, reason="jinja2 not installed"
)
CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}<|im_start|>assistant\n"
)


def _render(template: str, messages: list[dict]) -> str:
    from hfl.models.chat_template import template_env

    env = template_env()
    return str(
        env.from_string(template).render(
            messages=messages, add_generation_prompt=True, bos_token="<s>", eos_token="</s>"
        )
    )


@needs_jinja2
class TestProbe:
    def test_what_each_template_refuses(self):
        assert lc._probe_turns(ALTERNATING) == (True, True, False)
        assert lc._probe_turns(TYPED_ONLY) == (False, False, True)
        assert lc._probe_turns(CHATML) == (False, False, False)


@needs_jinja2
class TestFitTemplate:
    def test_turns_in_a_row_and_a_system_become_what_mistral_takes(self):
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        fitted, _ = lc._fit_template(msgs, None, ALTERNATING, False)
        assert fitted == [{"role": "user", "content": "S\n\na\n\nb"}]
        assert _render(ALTERNATING, fitted) == "<s>[INST] S\n\na\n\nb [/INST]"

    def test_an_opening_assistant_turn_gets_a_user_turn_before_it(self):
        msgs = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "q"}]
        fitted, _ = lc._fit_template(msgs, None, ALTERNATING, False)
        assert [m["role"] for m in fitted] == ["user", "assistant", "user"]
        _render(ALTERNATING, fitted)  # no longer raises

    def test_calls_and_results_are_not_joined(self):
        call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}]}
        msgs = [{"role": "user", "content": "q"}, call, dict(call)]
        joined = lc._fit_turns(msgs, True, False)
        assert len(joined) == 3

    def test_plain_text_is_drawn_by_a_template_that_wants_parts(self):
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        assert "France" not in _render(TYPED_ONLY, msgs)  # the question vanished
        fitted, _ = lc._fit_template(msgs, None, TYPED_ONLY, False)
        assert "What is the capital of France?" in _render(TYPED_ONLY, fitted)
        # llama-server's llama.cpp adapts the content itself.
        kept, _ = lc._fit_template(msgs, None, TYPED_ONLY, False, typed_content=False)
        assert kept == msgs

    def test_a_template_that_takes_everything_is_left_alone(self):
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        assert lc._fit_template(msgs, None, CHATML, True)[0] == msgs

    def test_the_system_message_can_be_left_to_the_server(self):
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}]
        assert lc._fit_template(msgs, None, ALTERNATING, False, takes_system=True)[0] == msgs


class _Tokenizer:
    """A Hugging Face tokenizer as far as the engines use one."""

    def __init__(self, template: str | None):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        if self.chat_template is None:
            raise ValueError(
                "Cannot use chat template functions because tokenizer.chat_template is not set"
            )
        return _render(self.chat_template, messages)


@needs_jinja2
class TestMLX:
    def _engine(self, template):
        from hfl.engine.mlx_engine import MLXEngine

        engine = MLXEngine.__new__(MLXEngine)
        engine._tokenizer = _Tokenizer(template)
        return engine

    def test_two_user_turns_keep_the_models_format(self):
        msgs = [ChatMessage(role="user", content="hola"), ChatMessage(role="user", content="crea")]
        prompt = self._engine(ALTERNATING)._messages_to_prompt(msgs)
        assert prompt == "<s>[INST] hola\n\ncrea [/INST]"  # not "user: hola\nuser: crea"

    def test_a_template_that_refuses_anyway_is_said(self, caplog):
        refusing = "{{ raise_exception('no') }}"
        with caplog.at_level(logging.WARNING, logger="hfl.engine.mlx_engine"):
            prompt = self._engine(refusing)._messages_to_prompt(
                [ChatMessage(role="user", content="q")]
            )
        assert prompt.endswith("assistant:")
        assert "refused this conversation" in caplog.text


class TestTransformers:
    def _engine(self, template):
        from hfl.engine.transformers_engine import TransformersEngine

        engine = TransformersEngine.__new__(TransformersEngine)
        engine._tokenizer = _Tokenizer(template)
        return engine

    def test_a_model_without_a_template_still_chats(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hfl.engine.transformers_engine"):
            prompt = self._engine(None)._build_prompt([ChatMessage(role="user", content="hi")])
        assert "hi" in prompt  # it raised ValueError: a 500 on every chat
        assert "no chat template" in caplog.text

    @needs_jinja2
    def test_tools_its_template_cannot_show_are_written_in(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        prompt = self._engine(CHATML)._build_prompt([ChatMessage(role="user", content="w?")], tools)
        assert "get_weather" in prompt  # handed to the template, they were never shown

    @needs_jinja2
    def test_two_user_turns_fit_a_template_that_refuses_them(self):
        msgs = [ChatMessage(role="user", content="a"), ChatMessage(role="user", content="b")]
        assert self._engine(ALTERNATING)._build_prompt(msgs) == "<s>[INST] a\n\nb [/INST]"


class _TextOnly(InferenceEngine):
    def load(self, model_path, **kwargs):
        pass

    def unload(self):
        pass

    def generate(self, prompt, config=None):
        return GenerationResult(text="x")

    def generate_stream(self, prompt, config=None):
        yield "x"

    def chat(self, messages, config=None, tools=None):
        return GenerationResult(text="answered without seeing it")

    def chat_stream(self, messages, config=None, tools=None):
        yield "answered without seeing it"

    @property
    def model_name(self):
        return "text"

    @property
    def is_loaded(self):
        return True


class TestImagesAModelCannotSee:
    def test_are_refused_not_dropped(self):
        from hfl.exceptions import ValidationError

        engine = _TextOnly()
        message = ChatMessage(role="user", content="colour?", images=[b"\x89PNG"])
        for call in (engine.chat, engine.chat_stream):
            with pytest.raises(ValidationError, match="cannot see images") as refused:
                call([message], GenerationConfig())
            assert refused.value.status_code == 400
        assert engine.chat([ChatMessage(role="user", content="q")]).text  # text still goes


class TestNothingGenerated:
    def test_a_reply_of_no_tokens_is_a_warning(self, caplog):
        from hfl.api.helpers import _account_generation

        with caplog.at_level(logging.WARNING, logger="hfl.api.helpers"):
            _account_generation(
                GenerationResult(text="", tokens_generated=0, tokens_prompt=2090), "chat"
            )
        assert "generated nothing" in caplog.text and "2090" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="hfl.api.helpers"):
            _account_generation(
                GenerationResult(text="ok", tokens_generated=3, tokens_prompt=9), "chat"
            )
        assert "generated nothing" not in caplog.text

    def test_a_stream_of_no_tokens_too(self, caplog, monkeypatch):
        import hfl.engine.base as base
        from hfl.api.streaming import _record_stream_generation

        monkeypatch.setattr(base, "stream_counts", lambda s: (2090, 0))
        with caplog.at_level(logging.WARNING, logger="hfl.api.helpers"):
            _record_stream_generation(iter(()), 0, 12.0, None)
        assert "generated nothing" in caplog.text

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``/v1/messages/count_tokens``: the tokens a message would cost, exactly.

Checked for real (2026-09-26) with Qwen2.5-0.5B-Instruct through the
official ``anthropic`` SDK, on the default GGUF backend, llama-server and
MLX: plain, with a system prompt and turns, with tools and with thinking
off, ``count_tokens`` equalled the ``input_tokens`` of the same
``/v1/messages`` request every time.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state
from hfl.engine.base import ChatMessage, GenerationConfig

USER = [{"role": "user", "content": "Weather in Paris?"}]
WEATHER = {
    "name": "get_weather",
    "description": "Current weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}


@pytest.fixture
def engine():
    state = get_state()
    fake = MagicMock()
    fake.is_loaded = True
    fake.count_prompt_tokens.return_value = 42
    model = MagicMock()
    model.name = "m"
    state.engine, state.current_model, state.api_key = fake, model, None
    yield fake
    state.engine = state.current_model = None


def _count(**body):
    return TestClient(app).post(
        "/v1/messages/count_tokens", json={"model": "m", "messages": USER, **body}
    )


class TestRoute:
    def test_the_engine_counts_what_messages_would_send(self, engine):
        response = _count(tools=[WEATHER], thinking={"type": "disabled"}, system="Be terse.")
        assert response.json() == {"input_tokens": 42}
        messages, config, tools = engine.count_prompt_tokens.call_args.args
        assert [m.role for m in messages] == ["system", "user"]
        assert config.reasoning == "off"
        assert tools and tools[0]["function"]["name"] == "get_weather"
        assert not engine.chat.called  # nothing generated

    def test_tool_choice_none_counts_no_tools(self, engine):
        _count(tools=[WEATHER], tool_choice={"type": "none"})
        assert engine.count_prompt_tokens.call_args.args[2] is None

    def test_a_backend_that_cannot_count_says_so(self, engine):
        engine.count_prompt_tokens.side_effect = NotImplementedError("vllm")
        response = _count()
        assert response.status_code == 501
        assert response.json()["error"]["type"] == "api_error"


def test_an_engine_that_cannot_count_raises_rather_than_guess():
    from hfl.engine.base import InferenceEngine

    class Engine(InferenceEngine):
        load = unload = generate = generate_stream = chat = chat_stream = MagicMock()
        model_name = "m"
        is_loaded = True

    with pytest.raises(NotImplementedError):
        Engine().count_prompt_tokens([ChatMessage(role="user", content="hi")])


@pytest.mark.skipif(
    importlib.util.find_spec("llama_cpp") is None, reason="llama-cpp-python not installed"
)
def test_llama_cpp_renders_with_its_formatter_and_tokenizes_as_chat_does():
    from hfl.engine.llama_cpp import LlamaCppEngine

    engine = LlamaCppEngine()
    seen: dict = {}

    class Formatter:
        hfl_name = "chat_template.default"
        template_vars: dict = {}

        def __call__(self, **kwargs):
            seen["rendered_with"] = {**self.template_vars, **kwargs}
            return SimpleNamespace(prompt="a b c", added_special=seen.get("added", False))

    def tokenize(data, add_bos, special):
        seen.update(add_bos=add_bos, special=special)
        return list(range(len(data.split()) + (1 if add_bos else 0)))

    engine._model = SimpleNamespace(tokenize=tokenize)
    engine._formatters = [Formatter()]
    engine._template_knows_tools = True
    engine._chat_template = ""
    hi = [ChatMessage(role="user", content="hi")]
    assert engine.count_prompt_tokens(hi) == 4  # BOS added: the template did not write it
    assert (seen["add_bos"], seen["special"]) == (True, True)
    seen["added"] = True
    tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    assert engine.count_prompt_tokens(hi, GenerationConfig(reasoning="off"), [tool]) == 3
    assert seen["rendered_with"]["tools"] == [tool]
    assert seen["rendered_with"]["enable_thinking"] is False


@pytest.mark.parametrize("module", ["mlx_engine", "transformers_engine"])
def test_mlx_and_transformers_count_the_prompt_they_render(module):
    """Called on an instance (a count_prompt_tokens left under another
    method's ``@staticmethod`` got the messages as ``self`` — a 500 on MLX)."""
    import importlib

    mod = importlib.import_module(f"hfl.engine.{module}")
    if module == "mlx_engine":
        engine = mod.MLXEngine()
        engine._tokenizer = SimpleNamespace(encode=lambda text: text.split())
        engine._messages_to_prompt = lambda messages, tools, reasoning: (
            "p " * (len(messages) + len(tools or []) + (reasoning == "off"))
        )
    else:
        engine = mod.TransformersEngine()
        engine._tokenizer = lambda text: {"input_ids": text.split()}
        engine._build_prompt = lambda messages, tools=None, reasoning=None: (
            "p " * (len(messages) + len(tools or []) + (reasoning == "off"))
        )
    two = [ChatMessage(role="user", content="a"), ChatMessage(role="user", content="b")]
    assert engine.count_prompt_tokens(two) == 2
    assert engine.count_prompt_tokens(two, GenerationConfig(reasoning="off"), [{}]) == 4

# SPDX-License-Identifier: HRUL-1.0
"""Tests that verify ``tools`` is propagated through the engine layer.

Covers:

- :class:`TransformersEngine` passes ``tools`` into
  ``tokenizer.apply_chat_template`` so qwen/llama3 templates can emit their
  native tool-call markers.
- :class:`LlamaCppEngine` forwards ``tools`` into
  ``create_chat_completion`` and normalises returned ``tool_calls``.
- :class:`FailoverEngine` and the async wrapper forward the kwarg.
- :class:`PromptBuilder.build` accepts the kwarg and injects a tool
  preamble.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import pytest

from hfl.engine.base import ChatMessage, GenerationResult
from hfl.engine.prompt_builder import PromptBuilder, PromptFormat

_HAS_LLAMA_CPP = importlib.util.find_spec("llama_cpp") is not None

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "write_wiki",
            "description": "Create or overwrite a wiki article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }
]


class TestTransformersEngineTools:
    def _engine_with_fake_tokenizer(self):
        from hfl.engine.transformers_engine import TransformersEngine

        engine = TransformersEngine()
        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template = MagicMock(return_value="PROMPT")
        return engine

    def test_build_prompt_forwards_tools_to_template(self):
        engine = self._engine_with_fake_tokenizer()
        out = engine._build_prompt(
            [ChatMessage(role="user", content="hi")],
            tools=TOOL_SPEC,
        )
        assert out == "PROMPT"
        call_kwargs = engine._tokenizer.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] == TOOL_SPEC
        assert call_kwargs["add_generation_prompt"] is True
        assert call_kwargs["tokenize"] is False

    def test_build_prompt_without_tools_does_not_pass_kwarg(self):
        engine = self._engine_with_fake_tokenizer()
        engine._build_prompt([ChatMessage(role="user", content="hi")])
        call_kwargs = engine._tokenizer.apply_chat_template.call_args.kwargs
        assert "tools" not in call_kwargs

    def test_build_prompt_falls_back_when_template_rejects_tools(self):
        """Older templates raise TypeError on unknown kwargs; we retry
        without ``tools`` instead of crashing the whole request."""
        engine = self._engine_with_fake_tokenizer()
        calls = {"n": 0}

        def _apply(msgs, **kw):
            calls["n"] += 1
            if "tools" in kw:
                raise TypeError("unexpected kwarg 'tools'")
            return "PROMPT"

        engine._tokenizer.apply_chat_template = _apply
        out = engine._build_prompt(
            [ChatMessage(role="user", content="hi")],
            tools=TOOL_SPEC,
        )
        assert out == "PROMPT"
        assert calls["n"] == 2  # initial try + fallback

    def test_build_prompt_preserves_tool_calls_and_name(self):
        engine = self._engine_with_fake_tokenizer()
        msgs = [
            ChatMessage(role="user", content="q"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Madrid"},
                        }
                    }
                ],
            ),
            ChatMessage(role="tool", content="22C sunny", name="get_weather"),
        ]
        engine._build_prompt(msgs)
        rendered = engine._tokenizer.apply_chat_template.call_args.args[0]
        assert rendered[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert rendered[2]["role"] == "tool"
        assert rendered[2]["name"] == "get_weather"

    def test_chat_forwards_tools_through_generate(self):
        engine = self._engine_with_fake_tokenizer()
        engine.generate = MagicMock(
            return_value=GenerationResult(text="<tool_call>", tokens_generated=1)
        )
        engine.chat([ChatMessage(role="user", content="hi")], tools=TOOL_SPEC)
        # build_prompt was invoked with tools (we check via tokenizer kwargs)
        call_kwargs = engine._tokenizer.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] == TOOL_SPEC


# The LlamaCppEngine suite below needs ``llama_cpp`` importable. CI's
# default ``[dev]`` install does not include the ``[llama]`` extra, so
# we skip the whole class in environments without it. The rest of the
# tool-calling test surface is exercised by the transformers, failover,
# and prompt-builder cases above.
@pytest.mark.skipif(
    not _HAS_LLAMA_CPP,
    reason="llama-cpp-python not installed (optional [llama] extra)",
)
class TestLlamaCppEngineTools:
    def _engine_with_fake_model(self, output: dict):
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine._model = MagicMock()
        engine._model.create_chat_completion.return_value = output
        engine._model_path = "/fake/qwen3.gguf"
        return engine

    def test_chat_forwards_tools_kwarg(self):
        engine = self._engine_with_fake_model(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "write_wiki",
                                        "arguments": '{"path":"a","content":"b"}',
                                    }
                                }
                            ],
                        }
                    }
                ],
                "usage": {"completion_tokens": 5, "prompt_tokens": 10},
            }
        )
        result = engine.chat([ChatMessage(role="user", content="hi")], tools=TOOL_SPEC)
        call_kwargs = engine._model.create_chat_completion.call_args.kwargs
        assert call_kwargs["tools"] == TOOL_SPEC
        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "write_wiki"
        # arguments string was parsed to dict
        assert result.tool_calls[0]["function"]["arguments"] == {
            "path": "a",
            "content": "b",
        }

    def test_chat_retries_without_tools_on_type_error(self):
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine._model = MagicMock()
        engine._model_path = "/fake/qwen3.gguf"
        calls: list[dict] = []

        def _ccc(**kw):
            calls.append(dict(kw))
            if "tools" in kw:
                raise TypeError("unexpected kwarg 'tools'")
            return {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {},
            }

        engine._model.create_chat_completion.side_effect = _ccc
        result = engine.chat([ChatMessage(role="user", content="hi")], tools=TOOL_SPEC)
        assert len(calls) == 2
        assert "tools" in calls[0]
        assert "tools" not in calls[1]
        assert result.text == "hi"
        assert result.tool_calls is None

    def test_chat_preserves_tool_and_assistant_tool_calls_in_messages(self):
        engine = self._engine_with_fake_model(
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {},
            }
        )
        engine.chat(
            [
                ChatMessage(role="user", content="q"),
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Madrid"},
                            }
                        }
                    ],
                ),
                ChatMessage(role="tool", content="22C", name="get_weather"),
            ]
        )
        sent = engine._model.create_chat_completion.call_args.kwargs["messages"]
        assert sent[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert sent[2]["role"] == "tool"
        assert sent[2]["name"] == "get_weather"


class TestFailoverEngineTools:
    def test_forwards_tools_to_underlying_engine(self):
        from hfl.engine.failover import FailoverEngine

        inner = MagicMock()
        inner.chat.return_value = GenerationResult(text="ok")
        fe = FailoverEngine([inner])
        fe.chat([ChatMessage(role="user", content="hi")], tools=TOOL_SPEC)
        call_kwargs = inner.chat.call_args.kwargs
        assert call_kwargs["tools"] == TOOL_SPEC


class TestPromptBuilderTools:
    def test_build_without_tools_unchanged(self):
        out = PromptBuilder.build(
            [ChatMessage(role="user", content="hi")],
            format=PromptFormat.CHATML,
        )
        assert "hi" in out
        assert "write_wiki" not in out

    def test_build_with_tools_injects_preamble(self):
        out = PromptBuilder.build(
            [ChatMessage(role="user", content="hi")],
            format=PromptFormat.CHATML,
            tools=TOOL_SPEC,
        )
        assert "write_wiki" in out
        # preamble is rendered in the CHATML system slot
        assert "system" in out

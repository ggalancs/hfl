# SPDX-License-Identifier: HRUL-1.0
"""Tests for the per-family tool-call parsers in :mod:`hfl.api.tool_parsers`.

Fixtures mirror the raw outputs documented in the HFL tool-calling spec
§1 (observed evidence) and §4 (per-family parsing rules).
"""

from __future__ import annotations

from hfl.api.tool_parsers import (
    dispatch,
    parse_fallback,
    parse_llama3,
    parse_mistral,
    parse_qwen,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_wiki",
            "description": "x",
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


# --- Qwen ---------------------------------------------------------------------


class TestQwen:
    def test_native_tool_call_tag(self):
        text = (
            '<tool_call>{"name": "write_wiki", "arguments": '
            '{"path": "topics/hello.md", "content": "Hello world"}}</tool_call>'
        )
        cleaned, calls = parse_qwen(text)
        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0] == {
            "function": {
                "name": "write_wiki",
                "arguments": {"path": "topics/hello.md", "content": "Hello world"},
            }
        }

    def test_thinking_block_is_stripped(self):
        text = (
            "<think>I should call it</think>\n"
            '<tool_call>{"name": "write_wiki", "arguments": {"path": "a", "content": "b"}}</tool_call>'
        )
        cleaned, calls = parse_qwen(text)
        assert "think" not in cleaned.lower()
        assert calls[0]["function"]["arguments"] == {"path": "a", "content": "b"}

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
        )
        _, calls = parse_qwen(text)
        assert [c["function"]["name"] for c in calls] == ["a", "b"]

    def test_no_tool_call(self):
        cleaned, calls = parse_qwen("just plain text")
        assert cleaned == "just plain text"
        assert calls == []

    def test_arguments_as_string_is_reparsed(self):
        text = '<tool_call>{"name": "x", "arguments": "{\\"k\\": 1}"}</tool_call>'
        _, calls = parse_qwen(text)
        assert calls[0]["function"]["arguments"] == {"k": 1}


# --- Llama 3 ------------------------------------------------------------------


class TestLlama3:
    def test_python_tag(self):
        text = (
            '<|python_tag|>{"name": "write_wiki", '
            '"parameters": {"path": "a", "content": "b"}}<|eom_id|>'
        )
        cleaned, calls = parse_llama3(text)
        assert cleaned == ""
        # parameters must be normalised to arguments
        assert calls[0]["function"]["arguments"] == {"path": "a", "content": "b"}
        assert calls[0]["function"]["name"] == "write_wiki"

    def test_function_tag(self):
        text = (
            '<function=write_wiki>{"path": "a", "content": "b"}</function>'
        )
        cleaned, calls = parse_llama3(text)
        assert cleaned == ""
        assert calls[0]["function"]["name"] == "write_wiki"
        assert calls[0]["function"]["arguments"] == {"path": "a", "content": "b"}

    def test_no_marker(self):
        cleaned, calls = parse_llama3("hi there")
        assert cleaned == "hi there"
        assert calls == []


# --- Mistral ------------------------------------------------------------------


class TestMistral:
    def test_tool_calls_array(self):
        text = (
            '[TOOL_CALLS][{"name": "write_wiki", '
            '"arguments": {"path": "a", "content": "b"}}]'
        )
        cleaned, calls = parse_mistral(text)
        assert cleaned == ""
        assert calls[0]["function"]["name"] == "write_wiki"
        assert calls[0]["function"]["arguments"] == {"path": "a", "content": "b"}

    def test_multiple_in_one_array(self):
        text = (
            '[TOOL_CALLS][{"name": "a", "arguments": {}}, '
            '{"name": "b", "arguments": {"x": 1}}]'
        )
        _, calls = parse_mistral(text)
        assert [c["function"]["name"] for c in calls] == ["a", "b"]


# --- Fallback -----------------------------------------------------------------


class TestFallback:
    def test_tool_call_envelope_from_spec(self):
        """Exact envelope observed in spec §1 Test 2."""
        text = (
            "<think>\nOkay, the user wants me to save a file...\n</think>\n\n"
            '{"tool_call": {"name": "write_wiki", "args": '
            '{"path": "topics/hello.md", "content": "Hello world."}}}'
        )
        cleaned, calls = parse_fallback(text)
        assert calls[0]["function"]["name"] == "write_wiki"
        assert calls[0]["function"]["arguments"] == {
            "path": "topics/hello.md",
            "content": "Hello world.",
        }
        assert "think" not in cleaned.lower()

    def test_name_arguments_envelope(self):
        text = '{"name": "write_wiki", "arguments": {"path": "a", "content": "b"}}'
        _, calls = parse_fallback(text)
        assert calls[0]["function"]["arguments"] == {"path": "a", "content": "b"}

    def test_fenced_json(self):
        text = (
            "Here is the call:\n```json\n"
            '{"name": "write_wiki", "arguments": {"path": "a", "content": "b"}}\n'
            "```"
        )
        _, calls = parse_fallback(text)
        assert calls[0]["function"]["name"] == "write_wiki"

    def test_no_envelope(self):
        cleaned, calls = parse_fallback("just a plain reply")
        assert calls == []
        assert cleaned == "just a plain reply"


# --- Dispatch -----------------------------------------------------------------


class TestDispatch:
    def test_routes_to_qwen(self):
        text = '<tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>'
        cleaned, calls = dispatch(text, model_name="qwen3-32b-q4_k_m", tools=TOOLS)
        assert len(calls) == 1
        assert cleaned == ""

    def test_routes_to_llama3(self):
        text = (
            '<|python_tag|>{"name": "x", "parameters": {"a": 1}}<|eom_id|>'
        )
        _, calls = dispatch(text, model_name="llama-3.2-70b", tools=TOOLS)
        assert calls[0]["function"]["arguments"] == {"a": 1}

    def test_routes_to_mistral(self):
        text = '[TOOL_CALLS][{"name": "x", "arguments": {"a": 1}}]'
        _, calls = dispatch(text, model_name="mistral-7b", tools=TOOLS)
        assert calls[0]["function"]["name"] == "x"

    def test_falls_back_to_envelope_when_family_parser_empty(self):
        """Spec §1 Test 2: qwen3 emitted a non-standard envelope because
        its chat template was not applied. Dispatch must still recover
        the call via the fallback parser when ``tools`` is non-empty."""
        text = (
            "<think>reasoning</think>\n\n"
            '{"tool_call": {"name": "write_wiki", "args": '
            '{"path": "topics/hello.md", "content": "Hello world."}}}'
        )
        cleaned, calls = dispatch(
            text, model_name="qwen3-32b-q4_k_m", tools=TOOLS
        )
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "write_wiki"
        assert calls[0]["function"]["arguments"]["path"] == "topics/hello.md"

    def test_plain_text_without_tools_is_not_misparsed(self):
        """A plain chat reply that happens to contain a JSON object must
        not be misinterpreted as a tool call when the client did not send
        ``tools``."""
        text = '{"key": "value"}'  # just JSON, no tools advertised
        cleaned, calls = dispatch(text, model_name="qwen3", tools=None)
        assert calls == []
        assert cleaned == '{"key": "value"}'

    def test_plain_text_with_tools_empty(self):
        cleaned, calls = dispatch("hi", model_name="qwen3", tools=TOOLS)
        assert calls == []
        assert cleaned == "hi"

    def test_empty_input(self):
        cleaned, calls = dispatch("", model_name="qwen3", tools=TOOLS)
        assert cleaned == ""
        assert calls == []

    def test_canonical_shape_is_enforced(self):
        """Rule C2: each entry is ``{function: {name, arguments}}``
        and ``arguments`` is a dict (rule C3)."""
        text = '<tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>'
        _, calls = dispatch(text, model_name="qwen3", tools=TOOLS)
        for tc in calls:
            assert set(tc.keys()) == {"function"}
            fn = tc["function"]
            assert set(fn.keys()) == {"name", "arguments"}
            assert isinstance(fn["name"], str)
            assert isinstance(fn["arguments"], dict)

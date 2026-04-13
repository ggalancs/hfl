# SPDX-License-Identifier: HRUL-1.0
"""Tests for the per-family tool-call parsers in :mod:`hfl.api.tool_parsers`.

Fixtures mirror the raw outputs documented in the HFL tool-calling spec
§1 (observed evidence) and §4 (per-family parsing rules).
"""

from __future__ import annotations

from hfl.api.tool_parsers import (
    dispatch,
    parse_fallback,
    parse_gemma4,
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
            "<tool_call>"
            '{"name": "write_wiki", "arguments": {"path": "a", "content": "b"}}'
            "</tool_call>"
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
        text = '<function=write_wiki>{"path": "a", "content": "b"}</function>'
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
        text = '[TOOL_CALLS][{"name": "write_wiki", "arguments": {"path": "a", "content": "b"}}]'
        cleaned, calls = parse_mistral(text)
        assert cleaned == ""
        assert calls[0]["function"]["name"] == "write_wiki"
        assert calls[0]["function"]["arguments"] == {"path": "a", "content": "b"}

    def test_multiple_in_one_array(self):
        text = '[TOOL_CALLS][{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]'
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


# --- Gemma 4 -----------------------------------------------------------------


class TestGemma4:
    """Gemma 4 uses its own split-pipe DSL, not JSON:

        <|tool_call>call:NAME{key:<|"|>string<|"|>,num:42}<tool_call|>

    The string delimiter is the dedicated ``<|"|>`` token (ID 110 in
    the Gemma 4 vocabulary). The parser transforms the DSL to JSON
    and delegates to ``json.loads``.

    These fixtures are transcribed from real outputs produced by
    ``bartowski/google_gemma-4-31B-it-GGUF`` during the 0.3.1/0.3.2
    incident so any regression in the transform path reproduces the
    bug that broke tool calling end to end.
    """

    def test_real_world_single_call(self):
        """Exact output captured from the bartowski GGUF when asked
        about the weather in Barcelona with ``get_weather`` supplied
        as a tool."""
        text = (
            '<|tool_call>call:get_weather{city:<|"|>Barcelona<|"|>,'
            'unit:<|"|>celsius<|"|>}<tool_call|>'
        )
        cleaned, calls = parse_gemma4(text)
        assert cleaned == ""
        assert len(calls) == 1
        assert calls[0] == {
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Barcelona", "unit": "celsius"},
            }
        }

    def test_truncated_by_stop_token(self):
        """When the engine sets ``stop=["<tool_call|>"]`` the closing
        marker is consumed by the stop machinery and never reaches
        the output. The regex must still match the envelope, ending
        at end-of-string."""
        text = '<|channel>thought\n<channel|><|tool_call>call:get_weather{city:<|"|>Paris<|"|>}'
        cleaned, calls = parse_gemma4(text)
        # The thought wrapper is left for the channel-marker filter
        # to strip; parse_gemma4 only extracts tool calls.
        assert "thought" in cleaned
        assert calls[0]["function"] == {
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }

    def test_numeric_boolean_and_null_values(self):
        text = "<|tool_call>call:cfg{n:42,f:3.14,b:true,ok:false,missing:null}<tool_call|>"
        _, calls = parse_gemma4(text)
        args = calls[0]["function"]["arguments"]
        assert args == {"n": 42, "f": 3.14, "b": True, "ok": False, "missing": None}

    def test_nested_object_argument(self):
        text = '<|tool_call>call:configure{outer:{inner:<|"|>hello<|"|>,count:3}}<tool_call|>'
        _, calls = parse_gemma4(text)
        assert calls[0]["function"]["arguments"] == {"outer": {"inner": "hello", "count": 3}}

    def test_string_with_internal_colon(self):
        """Colons inside string values (e.g. URLs) must not confuse
        the bare-key transformer. The split-on-delimiter approach
        handles this because string content never reaches the
        key-quoting regex."""
        text = '<|tool_call>call:http_get{url:<|"|>http://example.com:8080/path<|"|>}<tool_call|>'
        _, calls = parse_gemma4(text)
        assert calls[0]["function"]["arguments"]["url"] == ("http://example.com:8080/path")

    def test_no_tool_call_is_passthrough(self):
        text = "Just a normal reply, no tool call here."
        cleaned, calls = parse_gemma4(text)
        assert cleaned == text
        assert calls == []

    def test_multiple_tool_calls(self):
        text = (
            "<|tool_call>call:a{x:1}<tool_call|>"
            "middle text "
            '<|tool_call>call:b{y:<|"|>z<|"|>}<tool_call|>'
        )
        cleaned, calls = parse_gemma4(text)
        assert "middle text" in cleaned
        assert [c["function"]["name"] for c in calls] == ["a", "b"]
        assert calls[0]["function"]["arguments"] == {"x": 1}
        assert calls[1]["function"]["arguments"] == {"y": "z"}

    def test_malformed_body_yields_empty_args(self):
        """A call whose body can't be decoded still surfaces the
        function name so the caller can at least see *what* was
        invoked, with empty arguments as a safe default."""
        text = "<|tool_call>call:broken{this is not valid}<tool_call|>"
        _, calls = parse_gemma4(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "broken"
        assert calls[0]["function"]["arguments"] == {}

    def test_dispatch_routes_gemma4_by_model_name(self):
        """``dispatch`` must pick ``parse_gemma4`` based on the
        ``model_name`` substring ``gemma-4`` (or ``gemma4``). Earlier
        Gemma versions (2, 3) use a different output format and must
        NOT be routed through this parser."""
        text = '<|tool_call>call:search{q:<|"|>python<|"|>}<tool_call|>'
        _, calls = dispatch(text, model_name="google_gemma-4-31b-it-q4_k_m", tools=TOOLS)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "search"

    def test_dispatch_does_not_route_gemma2_to_gemma4_parser(self):
        """A Gemma 2 or Gemma 3 model name must NOT land in
        ``parse_gemma4`` — those families have their own output
        format (or none) and accidental routing would misparse
        legitimate replies as tool calls."""
        # Plain text reply. No tool calls expected.
        text = "Hello, how can I help you today?"
        _, calls = dispatch(text, model_name="google/gemma-2-9b-it", tools=TOOLS)
        assert calls == []


# --- Dispatch -----------------------------------------------------------------


class TestDispatch:
    def test_routes_to_qwen(self):
        text = '<tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>'
        cleaned, calls = dispatch(text, model_name="qwen3-32b-q4_k_m", tools=TOOLS)
        assert len(calls) == 1
        assert cleaned == ""

    def test_routes_to_llama3(self):
        text = '<|python_tag|>{"name": "x", "parameters": {"a": 1}}<|eom_id|>'
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
        cleaned, calls = dispatch(text, model_name="qwen3-32b-q4_k_m", tools=TOOLS)
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

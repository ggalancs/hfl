# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the dialect-agnostic chat resolution core.

This is the single decision point shared by the OpenAI / Ollama / Anthropic
chat routes, so it gets covered directly (the per-route tests then only
exercise wire-format translation).
"""

from unittest.mock import MagicMock

from hfl.api.chat_core import resolve_chat_output

QWEN = "qwen3-32b"
WRITE_WIKI = [{"type": "function", "function": {"name": "write_wiki", "parameters": {}}}]
MARKER = '<tool_call>\n{"name": "write_wiki", "arguments": {"path": "x"}}\n</tool_call>'


def test_plain_text_no_tools():
    out = resolve_chat_output("just a reply", QWEN, None)
    assert out.content == "just a reply"
    assert out.tool_calls == []
    assert out.has_tool_calls is False


def test_parses_marker_into_canonical_tool_call():
    out = resolve_chat_output(MARKER, QWEN, WRITE_WIKI)
    assert out.has_tool_calls
    assert out.content == ""
    call = out.tool_calls[0]["function"]
    assert call["name"] == "write_wiki"
    # Canonical shape keeps arguments as a dict (routes serialise per dialect).
    assert call["arguments"] == {"path": "x"}


def test_engine_tool_calls_preferred_over_parsing():
    engine_calls = [{"function": {"name": "get_time", "arguments": {}}}]
    out = resolve_chat_output("ignored text", QWEN, WRITE_WIKI, engine_calls)
    assert out.tool_calls == engine_calls
    assert out.content == ""


def test_magicmock_engine_tool_calls_not_trusted():
    # A MagicMock result's `.tool_calls` auto-attr must not masquerade as data;
    # we fall through to parsing the text.
    out = resolve_chat_output("plain", QWEN, WRITE_WIKI, MagicMock())
    assert out.tool_calls == []
    assert out.content == "plain"


def test_tools_disabled_never_scans_markers():
    # tool_choice="none": even a literal marker stays verbatim as content.
    out = resolve_chat_output(MARKER, QWEN, WRITE_WIKI, tools_disabled=True)
    assert out.has_tool_calls is False
    assert "write_wiki" in out.content


def test_no_tools_does_not_parse_generic_json():
    # Without tools declared, ordinary JSON-ish text is not a tool call.
    out = resolve_chat_output('{"name": "x", "arguments": {}}', QWEN, None)
    assert out.tool_calls == []

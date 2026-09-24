# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Qwen3-Coder's XML tool calls become structured calls.

Qwen3-Coder does not emit ``<tool_call>{json}</tool_call>``; its template
teaches ``<function=NAME><parameter=KEY>value</parameter></function>``.
Before this parser Claude Code, pointed at HFL, received that markup as
plain text and never ran a tool — found by running it for real.
"""

from __future__ import annotations

from hfl.api.tool_parsers import dispatch


def _tool(name, **props):
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": props}},
    }


READ = _tool("Read", file_path={"type": "string"}, limit={"type": "integer"})

REAL_OUTPUT = (
    "I need to read the calc.py file first.\n"
    "<tool_call>\n<function=Read>\n<parameter=file_path>\ncalc.py\n</parameter>\n"
    "</function>\n</tool_call>"
)


def test_the_output_seen_from_claude_code():
    content, calls = dispatch(REAL_OUTPUT, "qwen-coder", [READ])
    assert calls == [{"function": {"name": "Read", "arguments": {"file_path": "calc.py"}}}]
    assert content == "I need to read the calc.py file first."


def test_values_are_typed_from_the_schema():
    tool = _tool(
        "T",
        n={"type": "integer"},
        x={"type": "number"},
        on={"type": "boolean"},
        tags={"type": "array"},
        opts={"type": "object"},
        s={"type": "string"},
    )
    text = (
        "<tool_call><function=T>"
        "<parameter=n>\n42\n</parameter>"
        "<parameter=x>\n1.5\n</parameter>"
        "<parameter=on>\ntrue\n</parameter>"
        '<parameter=tags>\n["a", "b"]\n</parameter>'
        '<parameter=opts>\n{"k": 1}\n</parameter>'
        "<parameter=s>\n007\n</parameter>"
        "</function></tool_call>"
    )
    _, calls = dispatch(text, "qwen3-coder", [tool])
    assert calls[0]["function"]["arguments"] == {
        "n": 42,
        "x": 1.5,
        "on": True,
        "tags": ["a", "b"],
        "opts": {"k": 1},
        "s": "007",  # a string stays a string, however numeric it looks
    }


def test_a_value_that_does_not_fit_its_type_is_kept_as_text():
    tool = _tool("T", n={"type": "integer"})
    text = "<tool_call><function=T><parameter=n>\nmany\n</parameter></function></tool_call>"
    _, calls = dispatch(text, "qwen3-coder", [tool])
    assert calls[0]["function"]["arguments"] == {"n": "many"}


def test_code_is_kept_byte_for_byte():
    """An Edit's old_string must match the file exactly: only the one
    newline the template puts on each side is removed."""
    edit = _tool("Edit", old_string={"type": "string"}, new_string={"type": "string"})
    old = "    return a - b\n"
    new = "\n    return a + b  \n\n"
    text = (
        "<tool_call>\n<function=Edit>\n"
        f"<parameter=old_string>\n{old}\n</parameter>\n"
        f"<parameter=new_string>\n{new}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, calls = dispatch(text, "qwen-coder", [edit])
    assert calls[0]["function"]["arguments"] == {"old_string": old, "new_string": new}


def test_several_calls_in_one_reply():
    text = (
        "<tool_call>\n<function=Read>\n<parameter=file_path>\na.py\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=Read>\n<parameter=file_path>\nb.py\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    content, calls = dispatch(text, "qwen-coder", [READ])
    assert [c["function"]["arguments"]["file_path"] for c in calls] == ["a.py", "b.py"]
    assert content == ""


def test_a_missing_closing_parameter_tag_is_tolerated():
    text = (
        "<tool_call><function=Read><parameter=file_path>\na.py\n"
        "<parameter=limit>\n10\n</function></tool_call>"
    )
    _, calls = dispatch(text, "qwen-coder", [READ])
    assert calls[0]["function"]["arguments"] == {"file_path": "a.py", "limit": 10}


def test_an_alias_that_hides_the_family_still_parses_when_tools_are_sent():
    _, calls = dispatch(REAL_OUTPUT, "coder", [READ])
    assert calls and calls[0]["function"]["name"] == "Read"


def test_the_json_form_still_works():
    text = '<tool_call>{"name": "Read", "arguments": {"file_path": "a.py"}}</tool_call>'
    _, calls = dispatch(text, "qwen-coder", [READ])
    assert calls[0]["function"]["arguments"] == {"file_path": "a.py"}


def test_llama3_function_tags_are_not_taken_for_xml():
    text = '<function=get_weather>{"city": "Paris"}</function>'
    _, calls = dispatch(text, "llama-3.1-8b", [])
    assert calls == [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]


def test_without_tools_an_unknown_family_is_left_alone():
    content, calls = dispatch(REAL_OUTPUT, "coder", None)
    assert calls == []
    assert "<function=Read>" in content

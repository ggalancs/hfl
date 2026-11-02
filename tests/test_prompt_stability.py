# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The same conversation gives the same prompt, so the server's cache works.

Measured on 2026-11-01: Project Hub (a Swift app) sent its tools with the
keys in a different order each turn — Swift's JSONEncoder wrote one
dictionary in 3 orders in 50 encodings — and Qwen3.8's template puts the
tools first. llama-server found no slot sharing more than a tenth of the
prompt ("selected slot by LRU") and re-read all of it every turn: 20-38 s
for 4-7k tokens, where the new part was ~170.
"""

from __future__ import annotations

import json

from hfl.engine import llama_cpp as lc
from hfl.engine.base import ChatMessage, GenerationConfig


def _tool(name: str, order: tuple[str, ...]) -> dict:
    parts = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} a project",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "domain": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
    function = parts["function"]
    parameters = function["parameters"]
    # The same tool, its members in another order at every level.
    reordered_params = {k: parameters[k] for k in reversed(list(parameters))}
    reordered_params["properties"] = dict(reversed(list(parameters["properties"].items())))
    reordered = {k: function[k] for k in order}
    reordered["parameters"] = reordered_params
    return {"function": reordered, "type": "function"}


def test_reordered_tools_come_out_as_first_sent():
    lc._TOOLS_SEEN.clear()
    first = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search a project",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "domain": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {"type": "function", "function": {"name": "get", "parameters": {"type": "object"}}},
    ]
    assert lc._stable_tools(first) is first  # a client that keeps one order keeps it
    later = [
        {"function": {"parameters": {"type": "object"}, "name": "get"}, "type": "function"},
        _tool("search", ("parameters", "description", "name")),
    ]
    assert json.dumps(later) != json.dumps(first)
    assert json.dumps(lc._stable_tools(later)) == json.dumps(first)


def test_other_tools_keep_their_own_order():
    lc._TOOLS_SEEN.clear()
    a = [_tool("a", ("name", "description", "parameters"))]
    b = [_tool("b", ("parameters", "name", "description"))]
    lc._stable_tools(a)
    assert json.dumps(lc._stable_tools(b)) == json.dumps(b)


def test_a_copy_is_handed_out_and_the_memory_is_bounded():
    lc._TOOLS_SEEN.clear()
    first = [_tool("a", ("name", "description", "parameters"))]
    lc._stable_tools(first)
    again = lc._stable_tools([_tool("a", ("parameters", "name", "description"))])
    again[0]["type"] = "changed by the engine"
    third = lc._stable_tools([_tool("a", ("description", "parameters", "name"))])
    assert third[0]["type"] == "function"
    for i in range(200):
        lc._stable_tools([_tool(f"t{i}", ("name", "description", "parameters"))])
    assert len(lc._TOOLS_SEEN) == 64


def test_llama_server_sends_the_same_tools_every_turn():
    from tests.test_cov_llama_server_units import _engine

    lc._TOOLS_SEEN.clear()
    engine = _engine()
    msgs = [ChatMessage(role="user", content="find hfl")]
    turns = [
        [_tool("search", order)]
        for order in (
            ("name", "description", "parameters"),
            ("parameters", "name", "description"),
            ("description", "parameters", "name"),
        )
    ]
    sent = [json.dumps(engine._chat_body(msgs, GenerationConfig(), t)["tools"]) for t in turns]
    assert len({json.dumps(t) for t in turns}) == 3  # three orders in
    assert len(set(sent)) == 1  # one prompt out

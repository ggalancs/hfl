# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Security: tool-call parsing is linear in the reply.

The reply is steered by whoever writes the prompt, and it is parsed on the
event loop. Lazy patterns that never met their closer scanned to the end of
the text from every opener: ``<tool_call>{`` repeated took 2.9 s at 48 KB
and 46.5 s at 192 KB; Harmony's call markers were cubic (37 KB: 180 s);
DeepSeek's fence over a run of whitespace was cubic too (10 KB: 226 s).
Each case below now parses in well under a second; the budget is wide so a
slow machine does not fail it, and still far below what the old code took.
"""

from __future__ import annotations

import random
import time

import pytest

from hfl.api import tool_parsers
from hfl.api.tool_parsers import dispatch

TOOLS = [{"type": "function", "function": {"name": "a", "parameters": {}}}]
MODELS = ["qwen", "llama3", "gpt-oss", "deepseek", "glm", "gemma4", "mistral", "other"]
BUDGET_S = 3.0  # all eight families together

N = 16_000
CASES = {
    "qwen-unclosed": "<tool_call>{" * N,
    "llama3-python-tag": "<|python_tag|>{" * N,
    "llama3-function": "<function=a>{" * N,
    "qwen-xml-names": "<function=" * N,
    "qwen-xml-params": "<function=a>" + "<parameter=" * N + "</function>",
    "gemma4-unclosed": "<|tool_call>call:a{" * N,
    "harmony-messages": "<|channel|>to=functions.a<|message|>{" * (N // 4),
    "harmony-recipients": "to=functions.a " * N,
    "deepseek-begin": "<｜tool▁call▁begin｜>" * N,
    "deepseek-fence": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>a\n```json" + " " * N * 4 + "x",
    "glm-keys": "<tool_call>a " + "<arg_key>" * N,
    "glm-values": "<tool_call>a " + "<arg_key>k</arg_key><arg_value>" * N,
    "think-unclosed": "<think>" * N,
    "fence-unclosed": "```json {" * N,
    "dangling-whitespace": " " * N * 10 + "x",
    "braces": "{" * N * 4,
    "strings": '{"' * N,
    "escapes": '{"\\' * N,
}


@pytest.mark.parametrize("name", list(CASES))
def test_adversarial_reply_parses_in_linear_time(name):
    text = CASES[name]
    started = time.perf_counter()
    for model in MODELS:
        dispatch(text, model, TOOLS)
    elapsed = time.perf_counter() - started
    assert elapsed < BUDGET_S, f"{name}: {len(text)} chars took {elapsed:.1f} s"


def _reference_end(text: str, start: int) -> int | None:
    """The old per-start scan: where the braces opened at ``start`` balance."""
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def test_one_pass_brace_matching_agrees_with_the_per_start_scan():
    """The single pass tracks every start at once; for each ``{`` it must
    give what scanning from that ``{`` alone gives."""
    rng = random.Random(20261003)
    for _ in range(20_000):
        text = "".join(rng.choice('{}"\\a') for _ in range(rng.randint(0, 20)))
        expected = {
            s: end
            for s, ch in enumerate(text)
            if ch == "{" and (end := _reference_end(text, s)) is not None
        }
        assert tool_parsers._balanced_ends(text) == expected, text


def test_results_are_unchanged_on_ordinary_replies():
    qwen = 'Sure.<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
    assert dispatch(qwen, "qwen", TOOLS) == (
        "Sure.",
        [{"function": {"name": "a", "arguments": {"x": 1}}}],
    )
    harmony = '<|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x": 2}<|call|>'
    assert dispatch(harmony, "gpt-oss", TOOLS)[1] == [
        {"function": {"name": "a", "arguments": {"x": 2}}}
    ]
    deepseek = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>a\n"
        '```json\n{"x": 3}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )
    assert dispatch(deepseek, "deepseek", TOOLS) == (
        "",
        [{"function": {"name": "a", "arguments": {"x": 3}}}],
    )
    glm = "<tool_call>a<arg_key>x</arg_key><arg_value>4</arg_value></tool_call>"
    assert dispatch(glm, "glm", TOOLS)[1] == [{"function": {"name": "a", "arguments": {"x": 4}}}]
    # An unclosed call before a closed one: the closed one is still found.
    late = "<think>plan" + " " * 50 + "</think>" + '```json {"name": "a", "arguments": {}} ```'
    assert dispatch(late, "other", TOOLS)[1] == [{"function": {"name": "a", "arguments": {}}}]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The compatibility matrix's checks fail when a model gets it wrong.

``docs/compatibility.md`` is only worth what its checks are: one that
passes whatever the model does would publish a green table about nothing.
Each check runs here against a fake server answering right and wrong.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compat_matrix", Path(__file__).resolve().parents[1] / "scripts" / "compat_matrix.py"
)
cm = importlib.util.module_from_spec(_SPEC)
sys.modules["compat_matrix"] = cm  # dataclasses look their module up there
_SPEC.loader.exec_module(cm)

CALL = {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
GOOD_ANSWER = "It is 31°C with a thunderstorm in Paris."


def _client(reply):
    """A fake HFL whose /api/chat answers with ``reply(body)`` →
    (content, tool_calls, eval_count)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content, calls, count = reply(body)
        message = {"role": "assistant", "content": content, "tool_calls": calls}
        return httpx.Response(200, json={"message": message, "eval_count": count, "done": True})

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://hfl")


def _tools_reply(first_calls, answer):
    def reply(body):
        turn_two = any(m.get("role") == "tool" for m in body["messages"])
        return (answer, [], 20) if turn_two else ("", first_calls, 10)

    return reply


def test_tools_pass_when_the_model_calls_and_answers():
    http = _client(_tools_reply([CALL], GOOD_ANSWER))
    assert cm.check_tools(http, "m", "ollama", False)["ok"] is True


@pytest.mark.parametrize(
    ("calls", "answer"),
    [
        ([], GOOD_ANSWER),  # answered without calling
        ([CALL, CALL], GOOD_ANSWER),  # called twice
        ([{"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}], GOOD_ANSWER),
        ([CALL], "I cannot fulfill your request."),  # did not use the result
        ([CALL], "<tool_call>31 thunderstorm"),  # a marker leaked
    ],
)
def test_tools_fail_when_it_does_not(calls, answer):
    http = _client(_tools_reply(calls, answer))
    assert cm.check_tools(http, "m", "ollama", False)["ok"] is False


def test_reasoning_off_needs_fewer_tokens_and_the_answer():
    def reply(tokens_off, answer):
        return lambda body: (
            (answer, [], tokens_off) if body.get("think") is False else ("391", [], 1500)
        )

    assert cm.check_reasoning_off(_client(reply(23, "391")), "m")["ok"] is True
    assert cm.check_reasoning_off(_client(reply(1500, "391")), "m")["ok"] is False
    assert cm.check_reasoning_off(_client(reply(23, "400")), "m")["ok"] is False


@pytest.mark.parametrize(
    ("text", "ok"),
    [("pong", True), ("<think>hm</think>pong", False), ("ping", False)],
)
def test_chat(text, ok):
    assert cm.check_chat(_client(lambda body: (text, [], 1)), "m")["ok"] is ok


def test_a_server_error_is_a_failure_not_a_crash():
    def handler(request):
        return httpx.Response(500, text="boom")

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://hfl")
    result = cm.check_chat(http, "m")
    assert result["ok"] is False and result["detail"]


def test_the_table_counts_what_passed():
    ok, bad = {"ok": True, "detail": ""}, {"ok": False, "detail": "x"}
    results = {
        "a": {
            "reference": "hf.co/a",
            "family": "A",
            "backends": {"default": {"chat": ok, "tools": {"1": ok, "2": bad}, "vision": bad}},
        },
        "b": {"reference": "hf.co/b", "family": "B", "backends": {}, "skipped": "license"},
    }
    table = cm.render(results, {"date": "d", "machine": "m", "hfl": "v", "raw": "r.json"})
    assert "| `hf.co/a` | A | default | ✓ | 1/6 | — | ✗ |" in table
    assert "| `hf.co/b` | B | — | license |" in table

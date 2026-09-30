# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The streaming machinery the four chat dialects share.

Each dialect used to carry its own copy: the not-loaded answer, the
engine's stream with or without tools, backpressure, an error that never
leaks internals, the dispatcher slot released whatever happens."""

from __future__ import annotations

import asyncio
import logging

from hfl.api.chat_stream import run_chat_stream
from hfl.engine.base import ChatMessage, GenerationConfig

SECRET = "/Users/someone/private/model.gguf"


class _Renderer:
    log_label = "test"

    def not_loaded(self) -> str:
        return "NOT LOADED"

    def failed(self) -> str:
        return "FAILED"

    def start(self, turn) -> str:
        return "START "

    def token(self, turn, token: str) -> str:
        return f"<{token}>"

    def done(self, turn) -> str:
        return f" DONE({turn.text()}, {turn.emitted})"


class _Slot:
    released = 0

    async def __aexit__(self, *args) -> None:
        _Slot.released += 1


def _run(engine, tools=None) -> tuple[str, int]:
    _Slot.released = 0

    async def go() -> str:
        out = ""
        async for chunk in run_chat_stream(
            _Renderer(), engine=engine, model="m",
            messages=[ChatMessage(role="user", content="hi")],
            config=GenerationConfig(), tools=tools, slot_cm=_Slot(),
        ):  # fmt: skip
            out += chunk
        return out

    return asyncio.run(go()), _Slot.released


class _Engine:
    def __init__(self, tokens, fail_after=None, tools_ok=True):
        self.tokens, self.fail_after, self.tools_ok = tokens, fail_after, tools_ok
        self.calls = []

    def chat_stream(self, messages, config, **kw):
        if "tools" in kw and not self.tools_ok:
            raise TypeError("chat_stream() got an unexpected keyword argument 'tools'")
        self.calls.append(kw.get("tools"))

        def gen():
            for i, token in enumerate(self.tokens):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError(f"broke reading {SECRET}")
                yield token

        return gen()


def test_a_turn_streams_start_tokens_and_done_and_releases_the_slot():
    out, released = _run(_Engine(["a", "b"]))
    assert out == "START <a><b> DONE(ab, 2)" and released == 1


def test_no_model_loaded_answers_so_and_releases_the_slot():
    out, released = _run(None)
    assert out == "NOT LOADED" and released == 1


def test_a_failure_mid_stream_says_so_without_internals(caplog):
    with caplog.at_level(logging.ERROR, logger="hfl.api.chat_stream"):
        out, released = _run(_Engine(["a", "b", "c"], fail_after=1))
    assert out.endswith("FAILED") and SECRET not in out and released == 1
    assert SECRET in caplog.text  # the traceback goes to the server log


def test_an_engine_without_tools_gets_the_two_argument_call():
    engine = _Engine(["x"], tools_ok=False)
    out, _ = _run(engine, tools=[{"type": "function", "function": {"name": "f"}}])
    assert out == "START <x> DONE(x, 1)" and engine.calls == [None]

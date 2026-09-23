# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""`HFL_STREAM_QUEUE_GET_TIMEOUT`, documented since forever, read by nothing.

`docs/env-vars.md` describes it as "seconds the consumer waits for the
next token". `config` exposed `stream_queue_get_timeout` and no module
read it: `simple_stream_async` — the path behind five OpenAI and Ollama
streaming helpers — did a bare `await queue.get()` and waited forever. If
the producer thread wedged inside the engine its `finally` never posted
the sentinel, and the request hung for as long as the client held the
socket.

`vllm_engine` was worse than absent: its consumer `get()` passed
`stream_queue_put_timeout`, so the producer's knob governed the consumer
and the documented one governed nothing at all.

The subtlety that decides whether this fix is an improvement or a new
bug: the bound applies **from the second token onward**. The first one is
the end of prompt processing, which legitimately takes minutes on a large
model, so capping it at 30 seconds would kill healthy requests — a
regression dressed as a fix. The documented wording says "the *next*
token", and that is what is enforced.
"""

from __future__ import annotations

import inspect
import time

import pytest

import hfl.config as hfl_config
from hfl.api.streaming import StreamTimeoutError, simple_stream_async


def _fmt(item):
    return f"data: {item}\n"


def _done():
    return "data: [DONE]\n"


async def _drain(iterator, limit=50):
    out = []
    async for chunk in iterator:
        out.append(chunk)
        if len(out) >= limit:
            break
    return out


@pytest.fixture
def short_get_timeout(monkeypatch):
    monkeypatch.setattr(hfl_config.config, "stream_queue_get_timeout", 0.3)
    return 0.3


class TestASlowFirstTokenIsNotCut:
    """The regression this fix had to avoid.

    Prompt processing on a large model takes far longer than the
    inter-token budget. A 70B can spend minutes before the first token.
    """

    @pytest.mark.asyncio
    async def test_a_first_token_slower_than_the_timeout_still_arrives(self, short_get_timeout):
        def slow_first():
            time.sleep(short_get_timeout * 3)  # "prefill"
            yield "first"
            yield "second"

        chunks = await _drain(simple_stream_async(slow_first(), _fmt, _done))
        assert "data: first\n" in chunks, (
            "a prompt that took longer than the inter-token budget was cut off "
            "— the bound must not apply to the first token"
        )
        assert chunks[-1] == _done()


class TestAStalledProducerIsCutOff:
    @pytest.mark.asyncio
    async def test_a_gap_after_the_first_token_raises(self, short_get_timeout):
        """The hang this exists to stop."""

        def stalls_after_one():
            yield "first"
            time.sleep(short_get_timeout * 10)
            yield "never seen"

        started = time.perf_counter()
        with pytest.raises(StreamTimeoutError) as caught:
            await _drain(simple_stream_async(stalls_after_one(), _fmt, _done))
        elapsed = time.perf_counter() - started

        assert "mid-stream" in str(caught.value)
        assert elapsed < short_get_timeout * 6, (
            f"waited {elapsed:.1f}s for a {short_get_timeout}s budget — the "
            "timeout is not the one being applied"
        )

    @pytest.mark.asyncio
    async def test_a_normal_stream_is_untouched(self, short_get_timeout):
        def quick():
            yield from ("a", "b", "c")

        chunks = await _drain(simple_stream_async(quick(), _fmt, _done))
        assert chunks == ["data: a\n", "data: b\n", "data: c\n", _done()]


class TestTheKnobIsTheOneBeingRead:
    def test_the_consumer_reads_the_consumer_knob(self):
        """Checked per line, not over the whole function.

        ``simple_stream_async`` contains the producer too, and that side
        legitimately uses the PUT knob — a blanket "put must not appear"
        assertion failed on correct code.
        """
        lines = inspect.getsource(simple_stream_async).splitlines()
        bounded_get = [line for line in lines if "queue.get()" in line and "timeout=" in line]
        assert bounded_get, "the consumer no longer bounds its wait at all"
        for line in bounded_get:
            assert "stream_queue_get_timeout" in line, (
                f"a bounded queue get() using the wrong knob: {line.strip()}"
            )

    def test_vllm_consumer_reads_the_consumer_knob(self):
        """It used to pass the PUT timeout to a `get()`."""
        from hfl.engine import vllm_engine

        source = inspect.getsource(vllm_engine)
        get_lines = [
            line
            for line in source.splitlines()
            if "token_queue.get(" in line and "timeout=" in line
        ]
        assert get_lines, "the vLLM consumer no longer passes a timeout at all"
        for line in get_lines:
            assert "stream_queue_get_timeout" in line, (
                f"a queue get() bounded by the wrong knob: {line.strip()}"
            )

    @pytest.mark.asyncio
    async def test_changing_the_value_changes_the_deadline(self, monkeypatch):
        """Not merely "a timeout happens" — the configured number is used."""

        def stalls():
            yield "first"
            time.sleep(5.0)

        monkeypatch.setattr(hfl_config.config, "stream_queue_get_timeout", 0.2)
        started = time.perf_counter()
        with pytest.raises(StreamTimeoutError):
            await _drain(simple_stream_async(stalls(), _fmt, _done))
        fast = time.perf_counter() - started

        monkeypatch.setattr(hfl_config.config, "stream_queue_get_timeout", 1.0)
        started = time.perf_counter()
        with pytest.raises(StreamTimeoutError):
            await _drain(simple_stream_async(stalls(), _fmt, _done))
        slow = time.perf_counter() - started

        assert slow > fast * 2, (
            f"0.2s budget cut at {fast:.2f}s and 1.0s budget at {slow:.2f}s — "
            "the configured value is not the one being applied"
        )


class TestTheDocumentationMatches:
    def test_the_docs_describe_the_next_token_not_the_first(self):
        from pathlib import Path

        docs = Path(__file__).resolve().parents[1] / "docs" / "env-vars.md"
        row = next(
            line
            for line in docs.read_text(encoding="utf-8").splitlines()
            if "HFL_STREAM_QUEUE_GET_TIMEOUT" in line
        )
        assert "next token" in row.lower(), (
            "the documented wording is what makes first-token exemption correct; "
            "if it changes, the implementation has to change with it"
        )

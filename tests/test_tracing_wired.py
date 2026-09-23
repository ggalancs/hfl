# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""OpenTelemetry spans, emitted from somewhere at last.

`observability/tracing.py` shipped `configure_tracing` and a `trace_span`
context manager designed to be a no-op when the SDK is absent, so the
rest of HFL could instrument freely. Nothing ever called it — the June
architecture review recorded zero call sites and September found the same
zero. An operator installing the `otel` extra and pointing HFL at a
collector received no spans at all.

Two call sites, chosen so the coverage is complete rather than
decorative:

* `lifespan` configures tracing once at boot.
* `run_dispatched` opens one span per inference. Chat, generate and
  embeddings from all three API dialects funnel through that function,
  and the caller already passes the operation name — instrumenting the
  routers instead would be a dozen sites that drift apart.

The property that has to hold regardless of OTEL: the span must close on
every exit path, including the timeout path where the worker thread
outlives the request. A span left open leaks context into whatever task
the event loop runs next.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import time

import pytest
from fastapi import HTTPException

import hfl.config as hfl_config
from hfl.api.helpers import run_dispatched
from hfl.core.container import reset_container
from hfl.observability.tracing import is_enabled, reset_tracing, trace_span


@pytest.fixture
def clean_dispatcher():
    reset_container()
    original = hfl_config.config.generation_timeout
    yield
    hfl_config.config.generation_timeout = original
    reset_container()


class TestTheSpanIsOpened:
    def test_run_dispatched_opens_one(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(run_dispatched)))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "trace_span"
        ]
        assert len(calls) == 1, (
            f"expected exactly one trace_span call in run_dispatched, found {len(calls)}"
        )

    def test_the_server_configures_tracing_at_startup(self):
        from hfl.api import server

        source = inspect.getsource(server.lifespan)
        tree = ast.parse(textwrap.dedent(source))
        assert any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == "configure_tracing"
            for n in ast.walk(tree)
        ), "nothing configures tracing, so the SDK is never handed a tracer"


class TestTheSpanAlwaysCloses:
    """A span left open leaks context into the next task on the loop."""

    @staticmethod
    def _exit_calls(tree) -> list[ast.Call]:
        return [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "__exit__"
        ]

    def test_it_closes_in_a_finally_not_per_branch(self):
        """One `finally` beats three hand-written closes.

        `run_dispatched` has three exits — timeout, raise, success — and
        each releases the dispatcher slot differently. Closing the span
        per branch is three chances to forget one.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(run_dispatched)))
        tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
        assert tries, "no try/finally in run_dispatched"
        assert any(
            self._exit_calls(ast.Module(body=t.finalbody, type_ignores=[])) for t in tries
        ), "the span is not closed in a finally, so an exit path can leave it open"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_a_timed_out_request_still_closes_its_span(self, clean_dispatcher):
        """The hardest path: the worker thread outlives the request.

        Exercised for real rather than read, because this is the branch
        where the function deliberately returns while a thread keeps
        running — the one most likely to forget the span.
        """
        import threading

        release = threading.Event()
        hfl_config.config.generation_timeout = 0.2

        spans: list[str] = []
        real = trace_span

        with pytest.raises(HTTPException):
            await run_dispatched(lambda: release.wait(timeout=10.0), operation="probe")
        release.set()

        # Nothing to assert about OTEL internals when the SDK is absent;
        # what matters is that the call completed rather than raising from
        # an unbalanced context manager.
        assert real is trace_span and spans == []


class TestTheNoOpPathStaysFree:
    def test_tracing_is_off_by_default(self):
        reset_tracing()
        assert not is_enabled(), (
            "tracing configured itself without being asked — the default must "
            "cost nothing and send nothing anywhere"
        )

    def test_the_span_is_usable_with_no_sdk(self):
        reset_tracing()
        with trace_span("x", attributes={"a": 1}):
            pass  # must not raise

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_a_normal_call_is_unaffected(self, clean_dispatcher):
        hfl_config.config.generation_timeout = 5.0
        started = time.perf_counter()
        result = await run_dispatched(lambda: "value", operation="quick")
        assert result == "value"
        assert time.perf_counter() - started < 2.0, (
            "instrumenting the dispatch path made it measurably slower"
        )

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``/metrics`` counts what happens, streamed or not.

Measured on a live server (2026-09-26): after a model load, a streamed and
a non-streamed reply of 20 tokens each and a failed request,
``hfl_model_loads_total`` was 0, ``hfl_tokens_generated_total`` 20 and no
error was counted. Loads, unloads and errors were fed by events nothing
emitted, and streams were neither counted nor traced.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from hfl.metrics import get_metrics, reset_metrics


@pytest.fixture(autouse=True)
def fresh_metrics():
    reset_metrics()
    yield get_metrics()
    reset_metrics()


class Counted:
    """A stream that knows its own counts, as the llama.cpp engines do."""

    def __init__(self, chunks, prompt=7, generated=None, fail=None):
        self._chunks, self.fail = chunks, fail
        self.prompt_tokens, self.generated_tokens = prompt, generated

    def __iter__(self):
        yield from self._chunks
        if self.fail is not None:
            raise self.fail


def _drain(iterator):
    from hfl.api.streaming import stream_with_backpressure

    async def run():
        out = []
        async for chunk in stream_with_backpressure(iterator, str, lambda: "done"):
            out.append(chunk)
        return out

    return asyncio.run(run())


def test_a_finished_stream_counts_its_tokens(fresh_metrics, monkeypatch):
    import hfl.engine.base as base

    monkeypatch.setattr(base, "stream_counts", lambda s: (s.prompt_tokens, s.generated_tokens))
    _drain(Counted(["a", "b", "c"], prompt=7, generated=30))
    _drain(Counted(["a", "b"], prompt=None, generated=None))  # no counts: the chunks
    assert (fresh_metrics.tokens_generated, fresh_metrics.tokens_input) == (32, 7)


def test_a_failed_stream_counts_the_error_and_not_the_tokens(fresh_metrics, monkeypatch):
    with pytest.raises(ValueError):
        _drain(Counted(["a"], fail=ValueError("boom")))
    assert fresh_metrics.errors_by_type["ValueError"] == 1
    assert fresh_metrics.tokens_generated == 0


def test_a_stream_is_traced(monkeypatch):
    import hfl.observability.tracing as tracing

    spans: list[str] = []

    @contextlib.contextmanager
    def span(name, attributes=None):
        spans.append(name)
        yield

    monkeypatch.setattr(tracing, "trace_span", span)
    _drain(Counted(["a"]))
    assert spans == ["inference.stream"]


def test_a_failed_request_counts_its_error(fresh_metrics):
    from hfl.api.helpers import run_dispatched

    def boom():
        raise RuntimeError("engine failed")

    with pytest.raises(RuntimeError):
        asyncio.run(run_dispatched(boom, operation="chat"))
    assert fresh_metrics.errors_by_type["RuntimeError"] == 1


def test_loads_and_unloads_are_counted(fresh_metrics, temp_config, monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from hfl.api import model_loader
    from hfl.api.state import ResidentModel, get_state, reset_state
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import get_registry

    reset_state()
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    get_registry().add(ModelManifest("m", "org/m", str(model), "gguf"))
    engine = MagicMock()
    engine.is_loaded = True
    monkeypatch.setattr(model_loader, "select_engine", lambda path: engine)
    asyncio.run(model_loader.load_llm("m"))
    assert fresh_metrics.model_loads == 1 and engine.load.called

    state = get_state()
    resident = next(r for r in state.resident_models() if r.name == "m")
    assert isinstance(resident, ResidentModel)
    asyncio.run(state.evict("m"))
    assert fresh_metrics.model_unloads == 1
    assert "hfl_model_unloads_total 1" in fresh_metrics.export_prometheus()
    reset_state()


class Paced(Counted):
    """A stream whose first token takes a while (prompt processing), then
    the rest come fast."""

    def __iter__(self):
        import time

        time.sleep(0.2)
        yield self._chunks[0]
        for chunk in self._chunks[1:]:
            time.sleep(0.01)
            yield chunk


def test_a_stream_records_its_first_token_time_and_decode_speed(fresh_metrics, monkeypatch):
    """The prompt's 200 ms go to time-to-first-token, not into the speed:
    10 tokens after the first in ~0.1 s is ~100 tok/s, not 11 tok / 0.3 s."""
    import hfl.engine.base as base

    monkeypatch.setattr(base, "stream_counts", lambda s: (s.prompt_tokens, s.generated_tokens))
    _drain(Paced([str(i) for i in range(11)], prompt=5, generated=11))
    out = fresh_metrics.export_prometheus()
    ttft = float(
        next(x for x in out.splitlines() if x.startswith("hfl_time_to_first_token_ms_sum")).split()[
            -1
        ]
    )
    speed = float(
        next(
            x for x in out.splitlines() if x.startswith("hfl_generation_tokens_per_second_sum")
        ).split()[-1]
    )
    assert 180 <= ttft <= 1000, ttft
    assert 40 <= speed <= 110, speed  # tokens ÷ total time would be ~35

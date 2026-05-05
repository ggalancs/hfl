# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for ``hfl/engine/benchmark.py`` — V4 F3.2."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hfl.engine.benchmark import (
    BenchmarkSummary,
    _summarise,
    run_benchmark,
    run_benchmark_stream,
)


def _engine(*, tokens_per_call: int = 8):
    """Mock engine whose ``generate`` returns a predictable token count."""
    engine = MagicMock(spec=["generate"])  # no generate_stream
    result = MagicMock()
    result.tokens_generated = tokens_per_call
    result.text = "x" * tokens_per_call
    engine.generate = MagicMock(return_value=result)
    return engine


class TestRunBenchmarkSync:
    def test_runs_each_prompt_length(self):
        engine = _engine()
        out = run_benchmark(
            engine,
            model_name="test",
            runs_per_length=2,
            max_tokens=8,
            prompt_lengths=(16, 256),
        )
        assert out.model == "test"
        # 2 lengths × 2 runs = 4 raw measurements, 2 summaries.
        assert len(out.raw) == 4
        assert len(out.summaries) == 2
        assert {s.prompt_length for s in out.summaries} == {16, 256}

    def test_summary_has_percentiles(self):
        engine = _engine()
        out = run_benchmark(
            engine,
            model_name="test",
            runs_per_length=3,
            prompt_lengths=(16,),
        )
        summary = out.summaries[0]
        # All p50/p95/mean fields populated.
        assert summary.runs == 3
        assert summary.tps_mean > 0
        assert summary.total_p50_ms >= 0


class TestSummariseEdgeCases:
    def test_empty_runs_returns_zero_summary(self):
        s = _summarise(16, [])
        assert isinstance(s, BenchmarkSummary)
        assert s.runs == 0
        assert s.ttft_p50_ms == 0.0
        assert s.tps_mean == 0.0


class TestStreamingBenchmark:
    @pytest.mark.asyncio
    async def test_emits_starting_run_summary_done(self):
        engine = _engine()
        events = []
        async for event in run_benchmark_stream(
            engine,
            model_name="test",
            runs_per_length=2,
            prompt_lengths=(16,),
        ):
            events.append(event)

        statuses = [e["status"] for e in events]
        assert statuses[0] == "starting"
        # 2 runs + 1 summary + 1 done (after one prompt length).
        assert statuses.count("run") == 2
        assert statuses.count("summary") == 1
        assert statuses[-1] == "done"

    @pytest.mark.asyncio
    async def test_run_event_carries_metrics(self):
        engine = _engine()
        events = [
            event
            async for event in run_benchmark_stream(
                engine,
                model_name="test",
                runs_per_length=1,
                prompt_lengths=(16,),
            )
        ]
        run_events = [e for e in events if e["status"] == "run"]
        assert run_events
        for r in run_events:
            assert "ttft_ms" in r
            assert "total_ms" in r
            assert "tokens_per_second" in r

    @pytest.mark.asyncio
    async def test_summary_event_includes_p95(self):
        engine = _engine()
        events = [
            event
            async for event in run_benchmark_stream(
                engine,
                model_name="test",
                runs_per_length=3,
                prompt_lengths=(16,),
            )
        ]
        summary_events = [e for e in events if e["status"] == "summary"]
        assert summary_events
        s = summary_events[0]
        assert "ttft_p95_ms" in s
        assert "total_p95_ms" in s
        assert "tps_mean" in s

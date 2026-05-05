# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Model benchmark — V4 F3.2.

Runs deterministic prompts of varying lengths against a loaded
engine and reports TTFT, tok/s and total latency. Uses fixed seeds
so two consecutive runs are comparable; reports per-prompt-length
percentiles so a regression in long-context latency stands out from
short-prompt noise.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from hfl.engine.base import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """A single (prompt_length, repetition) measurement."""

    prompt_length: int
    tokens_generated: int
    ttft_ms: float | None
    """Time to first token (ms). ``None`` when the engine doesn't
    expose ``generate_stream`` — we cannot honestly measure TTFT
    without an incremental token signal, so the field is left blank
    rather than aliased to ``total_ms``."""

    total_ms: float
    """Total wall-clock for the generate call."""

    tokens_per_second: float
    measurement_mode: str = "stream"
    """``"stream"`` when TTFT was measured against the first yielded
    token; ``"no-stream"`` when only ``total_ms`` was timed and TTFT
    is ``None``."""


@dataclass
class BenchmarkSummary:
    """Aggregate results per prompt length — what the client renders."""

    prompt_length: int
    runs: int
    ttft_p50_ms: float | None
    """``None`` when no run produced a real TTFT measurement (engine
    without ``generate_stream``)."""
    ttft_p95_ms: float | None
    total_p50_ms: float
    total_p95_ms: float
    tps_mean: float
    tps_min: float
    tps_max: float
    measurement_mode: str = "stream"
    """``"stream"`` when at least one run produced a real TTFT;
    ``"no-stream"`` when every run was timed via ``generate`` only."""


@dataclass
class BenchmarkResult:
    model: str
    summaries: list[BenchmarkSummary] = field(default_factory=list)
    raw: list[BenchmarkRun] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Golden prompts (deterministic)
# ---------------------------------------------------------------------------


_GOLDEN = {
    16: "Briefly describe the colour blue.",
    256: (
        "Write a short paragraph that explains the difference between "
        "supervised and unsupervised learning, in plain language, with "
        "concrete examples for each. Aim for one paragraph; do not list "
        "bullet points. Make sure the explanation is suitable for a "
        "non-technical reader and avoid mathematical notation. End the "
        "paragraph with a single, concise summary sentence."
    ),
    2048: (
        # ~2000-character prompt: a concrete editing/critique task.
        "You are reviewing a draft technical README for an open-source "
        "project. The project is a local LLM server with REST endpoints "
        "compatible with Ollama, OpenAI and Anthropic. The README must "
        "explain installation, model pulling, basic chat, embeddings, "
        "and structured output, while also mentioning the multimodal "
        "capabilities and the tool calling format. The current draft is "
        "verbose and inconsistent in voice; some sections speak in the "
        "first person ('we'), others in the imperative ('install the "
        "package'). Re-write it so that all instructions use the "
        "imperative voice, every code block has a one-line preamble "
        "explaining what it does, and headings follow the pattern "
        "'## Verb + object' (e.g. '## Pull a model'). Trim repetitive "
        "phrases. Preserve the original example commands verbatim. "
        "Do not add new sections. Do not invent feature claims. End the "
        "rewrite with a one-paragraph 'Why this project exists' note "
        "that contrasts it briefly with Ollama, focusing on the "
        "HuggingFace Hub integration. The output must be valid Markdown "
        "and not exceed 60 lines. Keep the project name as it appears "
        "in the original. If you encounter ambiguous wording in the "
        "draft, prefer clarity over fidelity. Stop at the end of the "
        "rewrite — do not add commentary about the rewrite itself."
    ),
}


# ---------------------------------------------------------------------------
# Engine probing
# ---------------------------------------------------------------------------


def _measure_one(engine: "InferenceEngine", prompt: str, max_tokens: int) -> BenchmarkRun:
    """Run one generate call, deriving TTFT from a streaming pass
    when available.

    For engines that don't expose ``generate_stream``, the run falls
    back to ``generate`` and reports ``ttft_ms=None`` plus
    ``measurement_mode="no-stream"`` — we do NOT alias ``total_ms``
    into the TTFT slot because that would silently inflate the
    first-token latency percentile.
    """
    from hfl.engine.base import GenerationConfig

    cfg = GenerationConfig(max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=42)

    if hasattr(engine, "generate_stream"):
        start = time.perf_counter()
        first = None
        tokens = 0
        try:
            stream = engine.generate_stream(prompt, cfg)  # type: ignore[attr-defined]
            for tok in stream:
                if first is None:
                    first = time.perf_counter()
                tokens += 1
        except Exception:  # pragma: no cover — fall back
            return _measure_via_generate(engine, prompt, cfg)
        end = time.perf_counter()
        ttft_value = (first - start) * 1000 if first is not None else None
        total_ms = (end - start) * 1000
        tps = (tokens / (total_ms / 1000)) if total_ms > 0 else 0.0
        return BenchmarkRun(
            prompt_length=len(prompt),
            tokens_generated=tokens,
            ttft_ms=round(ttft_value, 2) if ttft_value is not None else None,
            total_ms=round(total_ms, 2),
            tokens_per_second=round(tps, 2),
            measurement_mode="stream",
        )

    return _measure_via_generate(engine, prompt, cfg)


def _measure_via_generate(engine: "InferenceEngine", prompt: str, cfg) -> BenchmarkRun:
    start = time.perf_counter()
    result = engine.generate(prompt, cfg)
    end = time.perf_counter()
    total_ms = (end - start) * 1000
    tokens = int(getattr(result, "tokens_generated", 0) or 0)
    tps = (tokens / (total_ms / 1000)) if total_ms > 0 else 0.0
    return BenchmarkRun(
        prompt_length=len(prompt),
        tokens_generated=tokens,
        # No incremental token signal on this path → TTFT cannot be
        # honestly measured. Report ``None`` rather than aliasing to
        # ``total_ms`` (which would silently inflate the percentile).
        ttft_ms=None,
        total_ms=round(total_ms, 2),
        tokens_per_second=round(tps, 2),
        measurement_mode="no-stream",
    )


def _summarise(prompt_length: int, runs: list[BenchmarkRun]) -> BenchmarkSummary:
    """Convert per-run measurements to p50/p95 aggregates.

    TTFT is reported as ``None`` when no run produced an honest
    measurement (engine has no ``generate_stream``). We do NOT
    average ``total_ms`` into the TTFT slot — that would silently
    overstate first-token latency.
    """
    if not runs:
        return BenchmarkSummary(
            prompt_length=prompt_length,
            runs=0,
            ttft_p50_ms=None,
            ttft_p95_ms=None,
            total_p50_ms=0.0,
            total_p95_ms=0.0,
            tps_mean=0.0,
            tps_min=0.0,
            tps_max=0.0,
            measurement_mode="no-stream",
        )

    total = sorted(r.total_ms for r in runs)
    tps = [r.tokens_per_second for r in runs]

    ttft_values = [r.ttft_ms for r in runs if r.ttft_ms is not None]
    if ttft_values:
        ttft_sorted = sorted(ttft_values)
        ttft_p50 = round(statistics.median(ttft_sorted), 2)
        ttft_p95 = round(
            ttft_sorted[int(len(ttft_sorted) * 0.95) - 1]
            if len(ttft_sorted) >= 2
            else ttft_sorted[-1],
            2,
        )
        mode = "stream"
    else:
        ttft_p50 = None
        ttft_p95 = None
        mode = "no-stream"

    return BenchmarkSummary(
        prompt_length=prompt_length,
        runs=len(runs),
        ttft_p50_ms=ttft_p50,
        ttft_p95_ms=ttft_p95,
        total_p50_ms=round(statistics.median(total), 2),
        total_p95_ms=round(total[int(len(total) * 0.95) - 1] if len(total) >= 2 else total[-1], 2),
        tps_mean=round(sum(tps) / len(tps), 2),
        tps_min=round(min(tps), 2),
        tps_max=round(max(tps), 2),
        measurement_mode=mode,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_benchmark(
    engine: "InferenceEngine",
    *,
    model_name: str,
    runs_per_length: int = 3,
    max_tokens: int = 64,
    prompt_lengths: tuple[int, ...] = (16, 256, 2048),
) -> BenchmarkResult:
    """Drive the benchmark synchronously.

    The streaming variant lives in :func:`run_benchmark_stream` and
    yields per-run progress so the CLI / NDJSON endpoint can render
    a bar instead of a long quiet wait.
    """
    raw: list[BenchmarkRun] = []
    summaries: list[BenchmarkSummary] = []
    for length in prompt_lengths:
        prompt = _GOLDEN.get(length, "Hello.")
        runs_for_length: list[BenchmarkRun] = []
        for _ in range(runs_per_length):
            runs_for_length.append(_measure_one(engine, prompt, max_tokens))
        raw.extend(runs_for_length)
        summaries.append(_summarise(length, runs_for_length))
    return BenchmarkResult(model=model_name, summaries=summaries, raw=raw)


async def run_benchmark_stream(
    engine: "InferenceEngine",
    *,
    model_name: str,
    runs_per_length: int = 3,
    max_tokens: int = 64,
    prompt_lengths: tuple[int, ...] = (16, 256, 2048),
) -> AsyncIterator[dict]:
    """Yield per-run NDJSON-shaped progress events.

    Event grammar:

      {"status": "starting", "prompt_lengths": [...], "runs_per_length": ...}
      {"status": "run", "prompt_length": ..., "run": ..., "ttft_ms": ..., "tps": ...}
      {"status": "summary", "prompt_length": ..., "ttft_p50_ms": ..., ...}
      {"status": "done", "model": "..."}
    """
    import asyncio

    yield {
        "status": "starting",
        "model": model_name,
        "prompt_lengths": list(prompt_lengths),
        "runs_per_length": runs_per_length,
        "max_tokens": max_tokens,
    }

    for length in prompt_lengths:
        prompt = _GOLDEN.get(length, "Hello.")
        runs: list[BenchmarkRun] = []
        for i in range(runs_per_length):
            run = await asyncio.to_thread(_measure_one, engine, prompt, max_tokens)
            runs.append(run)
            yield {
                "status": "run",
                "prompt_length": length,
                "run": i + 1,
                "ttft_ms": run.ttft_ms,
                "total_ms": run.total_ms,
                "tokens_per_second": run.tokens_per_second,
            }
        summary = _summarise(length, runs)
        yield {"status": "summary", **asdict(summary)}

    yield {"status": "done", "model": model_name}

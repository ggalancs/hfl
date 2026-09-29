#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""How well a running HFL server serves requests at the same time.

    python scripts/bench_concurrency.py http://127.0.0.1:11434 --model chat \\
        --other-model coder --out concurrency.json \\
        [--min-parallel-speedup 1.5] [--min-two-model-speedup 1.2]

Three measurements, each the median of ``--runs``:

- ``one``: a single request to ``--model``;
- ``parallel``: ``--concurrency`` requests to ``--model`` at once, against
  that many one after another;
- ``two_models``: one request to ``--model`` and one to ``--other-model`` at
  once, against the two one after another.

The speed-up is throughput (tokens per second) at once over throughput one
after another: 1.0 means the requests were served in turn. Throughput, not
time: a reply can end before ``--tokens``, and comparing times then
compared different amounts of work. Every prompt starts with a unique
prefix, so no request reuses another's cached prompt. Tokens are the
server's own counts; the comparison is always between runs on the same
server. ``--min-*`` turn it into a check: exit 1 when a speed-up is below
its floor.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

# Long enough to use up ``--tokens`` (a count got "1, 2, 3, ..., 1000" from a
# small model: 24 tokens).
PROMPT = "Write a long, detailed story about a lighthouse keeper and a storm."


def _ask(base: str, model: str, tokens: int) -> int:
    """One chat request; the completion tokens the server reports."""
    body = {
        "model": model,
        "stream": False,
        "options": {"num_predict": tokens, "temperature": 0},
        # A unique prefix: no request can reuse another's cached prompt.
        "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {PROMPT}"}],
    }
    r = httpx.post(f"{base}/api/chat", json=body, timeout=900)
    r.raise_for_status()
    return int(r.json().get("eval_count") or 0)


def _timed(calls: list) -> tuple[float, int]:
    """Run ``calls`` at once; wall seconds and total tokens."""
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(len(calls)) as pool:
        tokens = sum(pool.map(lambda call: call(), calls))
    return time.monotonic() - started, tokens


def _sequential(calls: list) -> tuple[float, int]:
    started = time.monotonic()
    tokens = sum(call() for call in calls)
    return time.monotonic() - started, tokens


def measure(base: str, model: str, other: str | None, n: int, tokens: int, runs: int) -> dict:
    def ask(m: str):
        return lambda: _ask(base, m, tokens)

    # Warm-up: load the models and fill every code path once.
    _ask(base, model, 8)
    if other:
        _ask(base, other, 8)
    one, par, seq, two, two_seq = [], [], [], [], []
    for _ in range(runs):
        wall, got = _sequential([ask(model)])
        one.append({"seconds": wall, "tokens": got})
        seq.append(_sequential([ask(model)] * n))
        par.append(_timed([ask(model)] * n))
        if other:
            two_seq.append(_sequential([ask(model), ask(other)]))
            two.append(_timed([ask(model), ask(other)]))

    def med(values: list[float]) -> float:
        return round(statistics.median(values), 3)

    out: dict = {
        "one": {
            "seconds": med([r["seconds"] for r in one]),
            "tokens_per_second": med([r["tokens"] / r["seconds"] for r in one]),
        },
        "parallel": {
            "requests": n,
            "sequential_seconds": med([s for s, _ in seq]),
            "concurrent_seconds": med([s for s, _ in par]),
            "tokens_per_second": med([t / s for s, t in par]),
        },
    }
    out["parallel"]["sequential_tokens_per_second"] = med([t / s for s, t in seq])
    out["parallel"]["speedup"] = round(
        out["parallel"]["tokens_per_second"] / out["parallel"]["sequential_tokens_per_second"], 2
    )
    if other:
        out["two_models"] = {
            "sequential_seconds": med([s for s, _ in two_seq]),
            "concurrent_seconds": med([s for s, _ in two]),
            "sequential_tokens_per_second": med([t / s for s, t in two_seq]),
            "tokens_per_second": med([t / s for s, t in two]),
        }
        out["two_models"]["speedup"] = round(
            out["two_models"]["tokens_per_second"]
            / out["two_models"]["sequential_tokens_per_second"],
            2,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base", help="the server, e.g. http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--other-model", help="a second model, for two models at once")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-parallel-speedup", type=float)
    parser.add_argument("--min-two-model-speedup", type=float)
    args = parser.parse_args()

    result = measure(
        args.base.rstrip("/"), args.model, args.other_model, args.concurrency, args.tokens,
        args.runs,
    )  # fmt: skip
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    failed = []
    if args.min_parallel_speedup and result["parallel"]["speedup"] < args.min_parallel_speedup:
        failed.append(
            f"parallel speedup {result['parallel']['speedup']} < {args.min_parallel_speedup}"
        )
    two = result.get("two_models")
    if args.min_two_model_speedup and (not two or two["speedup"] < args.min_two_model_speedup):
        failed.append(f"two-model speedup {two and two['speedup']} < {args.min_two_model_speedup}")
    for line in failed:
        print("BAD", line, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

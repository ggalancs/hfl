#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Compare HFL, Ollama and llama-server on the same GGUF file.

    python scripts/bench_compare.py hf.co/bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M

The model is pulled with ``hfl pull`` (into ``HFL_HOME``) and that one file
is served by all three: HFL by name, llama-server by path, Ollama through a
Modelfile ``FROM`` it (into a throwaway ``OLLAMA_MODELS``, never ~/.ollama).
Every server runs with its own defaults except the context (``--ctx``) and
full GPU offload. All three are started here and stay up together, but
requests go one at a time, and the rounds are interleaved in a rotating
order so a machine whose speed drifts during the run (a desktop in use)
does not favour whichever server happened to go last.

All three are measured through the same OpenAI-compatible streaming
endpoint, from the client's side. Every request starts with a unique
prefix, so no server can answer from a cached prompt, and every reply is
counted in tokens by this script with the model's own tokenizer (the
GGUF's vocabulary, through llama-cpp-python) — never trusting a server's
own count or its chunking:

- time to first token, for a short prompt and a long one (prefill);
- decode rate: completion tokens after the first, over the time between the
  first and the last content chunk;
- throughput with ``--concurrency`` requests at once: all tokens over the
  wall time.

Reported as medians of ``--runs`` runs, after one warm-up request.
Results go to stdout as a Markdown table and to ``--out`` as JSON.
Check the power source first: on a laptop, battery can cut decode speed
several times.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

SHORT = "Write a detailed paragraph about the history of the bicycle."
PARAGRAPH = (
    "The river town grew around a stone bridge built by merchants who traded wool, "
    "salt and timber. Each spring the floods reshaped the banks, and each summer the "
    "market filled the square with carts from three valleys. The records of the guilds "
    "describe disputes over tolls, the founding of a school and a fire that burned the "
    "granary. "
)
LONG = PARAGRAPH * 28 + "\n\nSummarize the text above in five sentences."
_REQUEST = itertools.count(1)
_TOKENIZER = None  # set in main(): the GGUF's vocabulary


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.tokenize(text.encode(), add_bos=False, special=False))


def _unique(prompt: str) -> str:
    """A different first line per request: no prefix is ever reused."""
    return f"Request {next(_REQUEST)}.\n{prompt}"


@dataclass
class Server:
    name: str
    port: int
    model: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    setup: list[list[str]] = field(default_factory=list)  # run once the server is up


def _ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _stream(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    """One streamed completion, timed from the client."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": _unique(prompt)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    sent = time.perf_counter()
    first = last = None
    parts: list[str] = []
    with httpx.stream("POST", f"{base}/v1/chat/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            for choice in event.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    parts.append(content)
    tokens = _count_tokens("".join(parts))
    decode = (tokens - 1) / (last - first) if first and last and last > first else 0.0
    return {
        "ttft_s": (first - sent) if first else None,
        "tokens": tokens,
        "decode_tok_s": decode,
        "total_s": time.perf_counter() - sent,
    }


def _concurrent(base: str, model: str, n: int, max_tokens: int) -> dict:
    results: list[dict] = []
    errors: list[str] = []

    def one() -> None:
        try:
            results.append(_stream(base, model, SHORT, max_tokens))
        except Exception as exc:  # reported, not hidden
            errors.append(repr(exc))

    threads = [threading.Thread(target=one) for _ in range(n)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)
    wall = time.perf_counter() - started
    tokens = sum(r["tokens"] for r in results)
    return {
        "requests": n,
        "ok": len(results),
        "errors": errors,
        "wall_s": wall,
        "aggregate_tok_s": tokens / wall if wall else 0.0,
    }


def _median(runs: list[dict], key: str) -> float | None:
    values = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(values) if values else None


def _start(server: Server, log_dir: Path) -> tuple[subprocess.Popen, object]:
    log = open(log_dir / f"{server.name}.log", "wb")
    proc = subprocess.Popen(
        server.argv,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env={**os.environ, **server.env},
    )
    base = f"http://127.0.0.1:{server.port}"
    deadline = time.monotonic() + 120
    while not _ok(f"{base}/v1/models"):
        if proc.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(f"{server.name} did not start; see {log.name}")
        time.sleep(0.5)
    for step in server.setup:
        subprocess.run(
            step,
            check=True,
            env={**os.environ, **server.env},
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    _stream(base, server.model, SHORT, 16)  # warm-up: load the model
    return proc, log


def bench(servers: list[Server], runs: int, max_tokens: int, concurrency: int) -> dict:
    """Measure every server, interleaved.

    All servers are up at once (requests still go one at a time), and each
    round visits them in a rotated order: a machine that slows down or
    speeds up during the run — a desktop in use does — shifts every server
    alike instead of whichever happened to be measured last.
    """
    samples: dict[str, dict[str, list[dict]]] = {
        s.name: {"short": [], "long": [], "concurrent": []} for s in servers
    }
    for round_ in range(runs):
        order = servers[round_ % len(servers) :] + servers[: round_ % len(servers)]
        for server in order:
            base = f"http://127.0.0.1:{server.port}"
            samples[server.name]["short"].append(_stream(base, server.model, SHORT, max_tokens))
            samples[server.name]["long"].append(_stream(base, server.model, LONG, max_tokens))
            samples[server.name]["concurrent"].append(
                _concurrent(base, server.model, concurrency, max_tokens)
            )
    results = {}
    for server in servers:
        short, long = samples[server.name]["short"], samples[server.name]["long"]
        conc = samples[server.name]["concurrent"]
        results[server.name] = {
            "short_ttft_s": _median(short, "ttft_s"),
            "long_ttft_s": _median(long, "ttft_s"),
            "decode_tok_s": _median(short + long, "decode_tok_s"),
            "tokens_short": _median(short, "tokens"),
            "tokens_long": _median(long, "tokens"),
            "concurrent": {
                "aggregate_tok_s": _median(conc, "aggregate_tok_s"),
                "failed_requests": sum(c["requests"] - c["ok"] for c in conc),
                "errors": [e for c in conc for e in c["errors"]],
            },
            "runs": {"short": short, "long": long, "concurrent": conc},
        }
    return results


def _version(argv: list[str]) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "not found"
    text = (out.stdout + out.stderr).strip()
    # ``ollama --version`` also asks whatever server answers on its default
    # port for *its* version; the client's own is on the "client version" line.
    for line in text.splitlines():
        if "client version is" in line:
            return line.split("client version is", 1)[1].strip()
    return next((ln for ln in text.splitlines() if any(c.isdigit() for c in ln)), "?").strip()


def _machine() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    if sys.platform == "darwin":
        for key, cmd in {
            "chip": ["sysctl", "-n", "machdep.cpu.brand_string"],
            "memory_bytes": ["sysctl", "-n", "hw.memsize"],
            "power": ["pmset", "-g", "ps"],
        }.items():
            try:
                info[key] = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
            except OSError:
                info[key] = "?"
        info["power"] = info["power"].splitlines()[0] if info.get("power") else "?"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", help="Hub reference, e.g. hf.co/org/repo-GGUF:Q4_K_M")
    parser.add_argument("--servers", default="hfl,ollama,llama-server")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--out", type=Path, default=Path("bench_results.json"))
    args = parser.parse_args()

    hfl = [sys.executable, "-c", "from hfl.cli.main import app; app()"]
    subprocess.run([*hfl, "pull", args.reference], check=True, stdin=subprocess.DEVNULL)
    from hfl.hub.resolver import parse_model_spec
    from hfl.models.registry import ModelRegistry

    spec = parse_model_spec(args.reference)
    manifest = ModelRegistry().find_pulled(spec.repo_id, spec.quantization)
    if manifest is None or not str(manifest.local_path).endswith(".gguf"):
        print("the reference did not resolve to a local GGUF file", file=sys.stderr)
        return 1
    gguf = str(manifest.local_path)
    global _TOKENIZER
    from llama_cpp import Llama  # the [llama] extra

    _TOKENIZER = Llama(model_path=gguf, vocab_only=True, verbose=False)

    with tempfile.TemporaryDirectory(prefix="hfl-bench-") as scratch:
        work = Path(scratch)
        modelfile = work / "Modelfile"
        modelfile.write_text(f"FROM {gguf}\nPARAMETER num_ctx {args.ctx}\n")
        ollama_env = {"OLLAMA_HOST": "127.0.0.1:11602", "OLLAMA_MODELS": str(work / "ollama")}
        servers = {
            "hfl": Server(
                "hfl",
                11601,
                manifest.name,
                [*hfl, "serve", "--port", "11601", "--ctx", str(args.ctx)],
            ),
            "hfl-llama-server": Server(
                "hfl-llama-server",
                11604,
                manifest.name,
                [*hfl, "serve", "--port", "11604", "--ctx", str(args.ctx)],
                {"HFL_LLM_LIBRARY": "llama-server"},
            ),
            "ollama": Server(
                "ollama",
                11602,
                "bench",
                ["ollama", "serve"],
                ollama_env,
                setup=[["ollama", "create", "bench", "-f", str(modelfile)]],
            ),
            "llama-server": Server(
                "llama-server",
                11603,
                "bench",
                [
                    "llama-server",
                    "-m",
                    gguf,
                    "--port",
                    "11603",
                    "-c",
                    str(args.ctx),
                    "-ngl",
                    "99",
                    "--alias",
                    "bench",
                ],
            ),
        }
        chosen = [s.strip() for s in args.servers.split(",") if s.strip()]
        missing = [n for n in chosen if n != "hfl" and shutil.which(servers[n].argv[0]) is None]
        if missing:
            print(f"not installed: {', '.join(missing)}", file=sys.stderr)
            return 1
        report = {
            "reference": args.reference,
            "gguf": gguf,
            "settings": {
                "ctx": args.ctx,
                "max_tokens": args.max_tokens,
                "runs": args.runs,
                "concurrency": args.concurrency,
                "order": "interleaved, rotated each round",
                "temperature": 0,
            },
            "machine": _machine(),
            "versions": {
                "hfl": _version([*hfl, "version"]),
                "ollama": _version(["env", "OLLAMA_HOST=127.0.0.1:9", "ollama", "--version"]),
                "llama-server": _version(["llama-server", "--version"]),
            },
            "results": {},
        }
        started: list[tuple[subprocess.Popen, object]] = []
        try:
            for name in chosen:
                print(f"starting {name}...", file=sys.stderr, flush=True)
                started.append(_start(servers[name], work))
            print("measuring (interleaved)...", file=sys.stderr, flush=True)
            report["results"] = bench(
                [servers[n] for n in chosen], args.runs, args.max_tokens, args.concurrency
            )
        finally:
            for proc, log in started:
                _stop(proc)
                log.close()
            for log in work.glob("*.log"):
                shutil.copy(log, args.out.with_name(f"{args.out.stem}-{log.name}"))

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n| | {' | '.join(chosen)} |\n|---|{'---|' * len(chosen)}")
    rows = [
        ("Time to first token, short prompt (s)", lambda r: r.get("short_ttft_s"), "{:.3f}"),
        ("Time to first token, ~2K-token prompt (s)", lambda r: r.get("long_ttft_s"), "{:.3f}"),
        ("Decode (tokens/s)", lambda r: r.get("decode_tok_s"), "{:.1f}"),
        (
            f"{args.concurrency} requests at once (tokens/s, total)",
            lambda r: (r.get("concurrent") or {}).get("aggregate_tok_s"),
            "{:.1f}",
        ),
    ]
    for label, get, fmt in rows:
        cells = []
        for name in chosen:
            value = get(report["results"][name])
            cells.append(fmt.format(value) if isinstance(value, (int, float)) else "error")
        print(f"| {label} | {' | '.join(cells)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

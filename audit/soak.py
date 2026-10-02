#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Hours of mixed load on one HFL server, measured as it goes (soak test).

    python audit/soak.py --work ~/hfl-audit-X --hours 12

Uses the audit's install and models (``local_audit.py --setup`` and a run
that pulled them). Four clients send, at random but always the same mix:
chat, streams, tool calls, embeddings, Anthropic messages, and streams
abandoned after a few tokens; models come and go (``HFL_MAX_LOADED_MODELS``
and a short ``keep_alive``). Every ``--interval`` seconds it records HFL's
memory (with its children), open files, child processes, latency and
errors to ``<work>/soak/samples.jsonl`` (one line each, written at once).

The verdict, on the samples after the first hour (warm-up):
- HFL's own memory, the open files and the child processes do not keep
  growing (the trend over the run: under 5 % of the memory's level, under
  20 files, under one process). The children's memory — models loaded and
  unloaded all the time — is recorded, not judged;
- no 5xx except capacity (429 and 503 are the server saying "later");
- p95 latency of the same request drifts under 10 %;
- the server ran the whole time, and nothing is left running after it.

Exit 0 only when all hold. Its own deadline (hours + 30 min) stops it
whatever happens; run it under ``audit/watchdog.py`` too.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import psutil

MODELS = ["chat", "think", "stories"]  # GGUF, in and out of memory
EMBED = "embed"
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Load:
    """The clients, and what they saw."""

    def __init__(self, base: str, seed: int) -> None:
        self.base = base
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.window: list[dict] = []  # requests since the last sample
        self.errors_path: Path | None = None

    def record(self, op: str, status: int | str, seconds: float, detail: str = "") -> None:
        entry = {"t": time.time(), "op": op, "status": status, "s": round(seconds, 3)}
        with self.lock:
            self.window.append(entry)
        bad = not (isinstance(status, int) and (status < 500 or status in (503,)))
        if bad and self.errors_path is not None:
            with open(self.errors_path, "a") as out:
                out.write(json.dumps({**entry, "detail": detail[:500]}) + "\n")

    def take(self) -> list[dict]:
        with self.lock:
            window, self.window = self.window, []
        return window

    # -- the requests ------------------------------------------------------

    def _post(self, op: str, path: str, body: dict, timeout: float = 300) -> None:
        started = time.monotonic()
        try:
            r = httpx.post(self.base + path, json=body, timeout=timeout)
            self.record(op, r.status_code, time.monotonic() - started, r.text[:300])
        except httpx.HTTPError as exc:
            self.record(op, type(exc).__name__, time.monotonic() - started, str(exc))

    def chat_probe(self) -> None:
        """Always the same request: the latency the drift is measured on."""
        body = {
            "model": "chat", "stream": False,
            "messages": [{"role": "user", "content": "Name three colours."}],
            "options": {"num_predict": 32, "temperature": 0, "seed": 1},
        }  # fmt: skip
        self._post("probe", "/api/chat", body)

    def chat(self) -> None:
        model = self.rng.choice(MODELS)
        body = {
            "model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": "Tell me a fact about the sea."}],
            "options": {"num_predict": 64},
        }  # fmt: skip
        self._post(f"chat:{model}", "/api/chat", body)

    def stream(self, abandon: bool) -> None:
        model = self.rng.choice(MODELS)
        body = {
            "model": model, "stream": True, "max_tokens": 128,
            "messages": [{"role": "user", "content": "Write a short story about a lighthouse."}],
        }  # fmt: skip
        op = f"{'abandon' if abandon else 'stream'}:{model}"
        started = time.monotonic()
        try:
            with httpx.stream(
                "POST", self.base + "/v1/chat/completions", json=body, timeout=300
            ) as r:
                status = r.status_code
                for n, _ in enumerate(r.iter_lines()):
                    if abandon and n >= 3:
                        break  # the client goes away mid-stream
            self.record(op, status, time.monotonic() - started)
        except httpx.HTTPError as exc:
            self.record(op, type(exc).__name__, time.monotonic() - started, str(exc))

    def tools(self) -> None:
        body = {
            "model": "chat", "max_tokens": 128, "tools": [TOOL],
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        }  # fmt: skip
        self._post("tools", "/v1/chat/completions", body)

    def embed(self) -> None:
        texts = [f"sentence number {self.rng.randint(0, 10**6)}" for _ in range(8)]
        self._post("embed", "/api/embed", {"model": EMBED, "input": texts})

    def anthropic(self) -> None:
        body = {
            "model": "chat", "max_tokens": 48,
            "messages": [{"role": "user", "content": "Say hello in French."}],
        }  # fmt: skip
        self._post("anthropic", "/v1/messages", body)

    def client(self) -> None:
        ops = [
            (self.chat_probe, 2), (self.chat, 4), (lambda: self.stream(False), 3),
            (lambda: self.stream(True), 1), (self.tools, 2), (self.embed, 2),
            (self.anthropic, 1),
        ]  # fmt: skip
        population = [op for op, weight in ops for _ in range(weight)]
        while not self.stop.is_set():
            with self.lock:
                op = self.rng.choice(population)
            op()


def _tree(pid: int) -> tuple[int, int, int, int]:
    """HFL's own RSS, the RSS of it and its children, their open files, and
    the number of children. HFL's own memory is where a leak would show: a
    child is a model's weights, loaded and unloaded all the time."""
    proc = psutil.Process(pid)
    procs = [proc, *proc.children(recursive=True)]
    own = rss = fds = 0
    for p in procs:
        try:
            mem = p.memory_info().rss
            rss += mem
            own += mem if p.pid == pid else 0
            fds += p.num_fds() if hasattr(p, "num_fds") else p.num_handles()
        except psutil.Error:
            pass
    return own, rss, fds, len(procs) - 1


def _p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=20)[18]


def _trend(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys over xs (per unit of x)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def verdict(samples: list[dict], hours: float) -> tuple[bool, list[str]]:
    lines: list[str] = []
    steady = [s for s in samples if s["h"] >= 1.0] or samples
    ok = True
    if len(steady) < 3:
        return False, ["fewer than 3 samples after the first hour: nothing to judge"]
    hs = [s["h"] for s in steady]
    span = hs[-1] - hs[0]
    for key, unit, limit_rel, limit_abs in (
        ("hfl_mb", "MB", 0.05, None),
        ("fds", "files", None, 20),
        ("children", "processes", None, 1),
    ):
        ys = [float(s[key]) for s in steady]
        growth = _trend(hs, ys) * span
        level = statistics.fmean(ys)
        bound = limit_abs if limit_abs is not None else level * (limit_rel or 0)
        good = growth <= bound
        ok &= good
        lines.append(
            f"{'OK ' if good else 'BAD'} {key}: trend {growth:+.1f} {unit} over {span:.1f} h "
            f"(level {level:.0f}, allowed {bound:.0f})"
        )
    server_errors = sum(s["errors_5xx"] for s in samples)
    good = server_errors == 0
    ok &= good
    lines.append(f"{'OK ' if good else 'BAD'} non-capacity 5xx and failures: {server_errors}")
    early = [v for s in steady[: max(1, len(steady) // 6)] for v in s["probe_latencies"]]
    late = [v for s in steady[-max(1, len(steady) // 6) :] for v in s["probe_latencies"]]
    if early and late:
        drift = (_p95(late) - _p95(early)) / _p95(early)
        good = drift < 0.10
        ok &= good
        lines.append(
            f"{'OK ' if good else 'BAD'} p95 of the probe: {_p95(early):.2f}s early, "
            f"{_p95(late):.2f}s late ({drift:+.0%})"
        )
    else:
        ok = False
        lines.append("BAD no probe latencies to compare")
    alive = all(s["server_alive"] for s in samples)
    ok &= alive
    lines.append(f"{'OK ' if alive else 'BAD'} the server ran the whole time")
    return ok, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--interval", type=float, default=300.0)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    hfl = work / "venv" / ("Scripts/hfl.exe" if os.name == "nt" else "bin/hfl")
    if not hfl.exists():
        print(f"no audit install in {work}: run local_audit.py --setup first", file=sys.stderr)
        return 2
    out = work / "soak"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    samples_path = out / f"samples-{stamp}.jsonl"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HFL_", "OLLAMA_"))}
    env.pop("HF_TOKEN", None)
    env.update(
        HFL_HOME=str(work / "home"), HF_HOME=str(work / "hf"), HFL_LANG="en",
        HFL_RATE_LIMIT_ENABLED="false", HFL_MAX_LOADED_MODELS="2", HFL_KEEP_ALIVE="2m",
        PYTHONUNBUFFERED="1",
    )  # fmt: skip
    port = _port()
    log = open(out / f"serve-{stamp}.log", "wb")
    server = subprocess.Popen(
        [str(hfl), "serve", "--port", str(port)], env=env, stdout=log,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    started = time.monotonic()
    hard_stop = started + args.hours * 3600 + 1800
    load = Load(base, args.seed)
    load.errors_path = out / f"errors-{stamp}.jsonl"
    samples: list[dict] = []
    try:
        while True:
            if server.poll() is not None:
                print("the server exited at start", file=sys.stderr)
                return 1
            try:
                if httpx.get(base + "/healthz", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        clients = [threading.Thread(target=load.client, daemon=True) for _ in range(args.clients)]
        for c in clients:
            c.start()
        end = started + args.hours * 3600
        next_sample = time.monotonic() + args.interval
        while time.monotonic() < end and time.monotonic() < hard_stop:
            time.sleep(min(5.0, max(0.0, next_sample - time.monotonic())))
            if time.monotonic() < next_sample:
                continue
            next_sample += args.interval
            window = load.take()
            alive = server.poll() is None
            own, rss, fds, children = _tree(server.pid) if alive else (0, 0, 0, 0)

            def failed(r: dict) -> bool:
                status = r["status"]
                return not (isinstance(status, int) and (status < 500 or status == 503))

            sample = {
                "h": round((time.monotonic() - started) / 3600, 3),
                "hfl_mb": round(own / 2**20, 1), "rss_mb": round(rss / 2**20, 1),
                "fds": fds, "children": children,
                "requests": len(window),
                "errors_5xx": sum(1 for r in window if failed(r)),
                "capacity": sum(1 for r in window if r["status"] in (429, 503)),
                "probe_latencies": [r["s"] for r in window if r["op"] == "probe"
                                    and r["status"] == 200],
                "server_alive": alive,
            }  # fmt: skip
            samples.append(sample)
            with open(samples_path, "a") as f:
                f.write(json.dumps(sample) + "\n")
            print(
                f"{sample['h']:6.2f} h  hfl {sample['hfl_mb']:7.1f} MB  "
                f"all {sample['rss_mb']:8.1f} MB  fds {fds:4d}  "
                f"children {children}  req {len(window):4d}  5xx {sample['errors_5xx']}",
                flush=True,
            )
            if not alive:
                break
    finally:
        load.stop.set()
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)
        log.close()
    time.sleep(3)
    leftovers = [
        p.pid
        for p in psutil.process_iter(["name", "cmdline"])
        if (p.info["name"] or "").startswith("llama-server")
        and any(str(work / "home") in c for c in (p.info["cmdline"] or []))
    ]
    ok, lines = verdict(samples, args.hours)
    lines.append(f"{'OK ' if not leftovers else 'BAD'} nothing left running: {leftovers or 'none'}")
    ok &= not leftovers
    report = "\n".join(lines)
    (out / f"verdict-{stamp}.txt").write_text(report + "\n")
    print(report)
    print("SOAK OK" if ok else "SOAK FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

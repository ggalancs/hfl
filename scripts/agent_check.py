#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Coding agents on HFL, checked for real: can they finish a task?

Starts ``hfl serve`` with a model, then asks each agent — Claude Code
(through ``hfl launch``) and Codex (its ``codex`` binary, or ``npx`` if it
is not installed) — to fix a planted bug, each in a fresh folder, and
judges by running the code, not by the agent's words. It also flags a final
message that leaked tool-call markup.

    python scripts/agent_check.py --model qwen3-coder
    python scripts/agent_check.py --model qwen3-coder --home /tmp/hfl-home

What this found the first time (2026-09-26): an HFL that crashed under
Claude Code — a dropped stream let two requests decode on one model — and
a stray ``<tool_call>`` ending both agents' answers. Each agent may send
requests the model never finishes, so a run takes minutes, not seconds.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

HFL = str(Path(sys.executable).with_name("hfl"))
TASK = "calc.py has a bug: add(2, 3) must return 5. Fix it by editing calc.py."
BUGGY = "def add(a, b):\n    return a - b\n"
MARKUP = re.compile(r"<tool_call>|<function=|<\|[a-z_]+\|>")


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fixed(folder: Path) -> bool:
    code = "from calc import add; raise SystemExit(0 if add(2, 3) == 5 else 1)"
    # -B: no cached bytecode. "a - b" and "a + b" are the same size, so a
    # fix within the same second looked unchanged to a cached .pyc.
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=folder).returncode == 0


def _claude(port: int, model: str, folder: Path, env: dict) -> list[str]:
    return [
        HFL, "launch", "claude", "-m", model, "--port", str(port), "--",
        "-p", TASK, "--allowedTools", "Read,Edit",
    ]  # fmt: skip


def _codex(port: int, model: str, folder: Path, env: dict) -> list[str]:
    codex = shutil.which("codex") or None
    base = [codex] if codex else ["npx", "-y", "@openai/codex"]
    url = f"http://127.0.0.1:{port}/v1"
    provider = f'model_providers.hfl={{name="HFL", base_url="{url}", wire_api="responses"}}'
    return [
        *base, "exec", "--skip-git-repo-check", "-s", "workspace-write",
        "-c", "model_provider=hfl", "-c", provider, "-m", model, TASK,
    ]  # fmt: skip


AGENTS = {"claude": _claude, "codex": _codex}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="local model name or alias")
    parser.add_argument("--home", type=Path, default=None, help="HFL home (default: HFL_HOME)")
    parser.add_argument("--agents", default="claude,codex")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per agent")
    args = parser.parse_args()

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if args.home:
        env["HFL_HOME"] = str(args.home)
    work = Path(tempfile.mkdtemp(prefix="hfl-agents-"))
    port = _port()
    log = open(work / "serve.log", "wb")
    server = subprocess.Popen(
        [HFL, "serve", "--port", str(port)], env=env, stdout=log, stderr=subprocess.STDOUT
    )
    failed = False
    try:
        deadline = time.monotonic() + 120
        while True:
            if server.poll() is not None:
                print(f"hfl serve exited; see {work / 'serve.log'}")
                return 1
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                print("hfl serve did not start")
                return 1
            time.sleep(0.5)
        for name in [a.strip() for a in args.agents.split(",") if a.strip()]:
            folder = work / name
            folder.mkdir()
            (folder / "calc.py").write_text(BUGGY)
            argv = AGENTS[name](port, args.model, folder, env)
            started = time.monotonic()
            try:
                out = subprocess.run(
                    argv, cwd=folder, env=env, capture_output=True, text=True,
                    timeout=args.timeout, stdin=subprocess.DEVNULL,
                )  # fmt: skip
                text = out.stdout + out.stderr
            except subprocess.TimeoutExpired as exc:
                text = f"timed out after {exc.timeout}s"
            (work / f"{name}.log").write_text(text)
            fixed = _fixed(folder)
            leaked = bool(MARKUP.search(text.strip().splitlines()[-1] if text.strip() else ""))
            alive = server.poll() is None
            failed |= not (fixed and not leaked and alive)
            print(
                f"{name:7} {'fixed' if fixed else 'NOT fixed':9} "
                f"{'markup leaked' if leaked else 'clean answer':14} "
                f"server {'up' if alive else 'DOWN'}  {time.monotonic() - started:.0f}s",
                flush=True,
            )
            if not alive:
                break
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
        log.close()
    print(f"logs: {work}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HFL on this machine, checked for real: install to answer, end to end.

What the unit tests cannot say — does HFL, as installed here, pull models,
serve them and answer over its APIs on this OS and CPU? In a fresh HFL home
(deleted at the end unless ``--keep``), with two small Apache-2.0 models:

    python scripts/platform_check.py            # uses the ``hfl`` beside this Python
    python scripts/platform_check.py --keep     # leave the home for inspection

Checks: ``hfl pull`` of a chat GGUF and an embedding GGUF; ``hfl serve``;
a plain answer over the Ollama API, a streamed one over OpenAI's, one over
Anthropic's; unit-length embeddings; ``hfl outdated`` (the Hub reachable);
``hfl rm`` deleting what HFL owns. Each step has its own deadline, and the
server this starts is stopped whatever happens. Exit 0 only when every
check passed; the platform and the backend used are printed first, so a
result says where it holds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

import httpx

# Where pip put the ``hfl`` of this Python: its scripts folder (beside
# python on POSIX venvs, ``Scripts\`` on Windows).
HFL = str(Path(sysconfig.get_path("scripts")) / ("hfl.exe" if os.name == "nt" else "hfl"))
CHAT = ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "Q4_K_M", "chat")  # Apache-2.0, ~400 MB
EMBED = ("nomic-ai/nomic-embed-text-v1.5-GGUF", "Q4_K_M", "embed")  # Apache-2.0, ~80 MB
QUESTION = "What is the capital of France? Answer with one word."


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Run:
    def __init__(self, env: dict[str, str]) -> None:
        self.env, self.failed = env, False

    def check(self, label: str, ok: bool, detail: object = "") -> bool:
        self.failed |= not ok
        print(f"{'OK ' if ok else 'BAD'} {label}: {str(detail)[:160]}", flush=True)
        return ok

    def hfl(self, *args: str, timeout: float = 900) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [HFL, *args], env=self.env, capture_output=True, encoding="utf-8",
                errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(exc.cmd, 124, "", f"timed out after {timeout}s")


def _checks(run: Run, http: httpx.Client) -> None:
    asked = [{"role": "user", "content": QUESTION}]
    body = {"model": "chat", "stream": False, "options": {"temperature": 0}, "messages": asked}
    reply = http.post("/api/chat", json=body)
    text = reply.json().get("message", {}).get("content", "") if reply.status_code == 200 else ""
    run.check("Ollama /api/chat", "paris" in text.lower(), text or reply.text)

    streamed = ""
    with http.stream(
        "POST", "/v1/chat/completions",
        json={"model": "chat", "stream": True, "temperature": 0,
              "messages": [{"role": "user", "content": QUESTION}]},
    ) as response:  # fmt: skip
        for line in response.iter_lines():
            if line.startswith("data: {"):
                delta = json.loads(line[6:])["choices"]
                streamed += (delta[0]["delta"].get("content") or "") if delta else ""
    run.check("OpenAI /v1/chat/completions stream", "paris" in streamed.lower(), streamed)

    message = http.post(
        "/v1/messages",
        json={"model": "chat", "max_tokens": 50, "temperature": 0,
              "messages": [{"role": "user", "content": QUESTION}]},
    )  # fmt: skip
    blocks = message.json().get("content", []) if message.status_code == 200 else []
    said = "".join(b.get("text", "") for b in blocks)
    run.check("Anthropic /v1/messages", "paris" in said.lower(), said or message.text)

    embedded = http.post("/api/embed", json={"model": "embed", "input": ["a cat", "a dog"]})
    vectors = embedded.json().get("embeddings", []) if embedded.status_code == 200 else []
    norms = [round(sum(x * x for x in v) ** 0.5, 3) for v in vectors]
    unit = norms == [1.0, 1.0] and len(vectors[0]) == 768
    run.check("embeddings", unit, norms or embedded.text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep", action="store_true", help="keep the HFL home it creates")
    args = parser.parse_args()

    home = Path(tempfile.mkdtemp(prefix="hfl-platform-"))
    # UTF-8 everywhere: a Windows console's cp1252 cannot print rich's tables.
    env = {**os.environ, "HFL_HOME": str(home), "PYTHONUNBUFFERED": "1", "HFL_LANG": "en",
           "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # fmt: skip
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run = Run(env)
    version = run.hfl("version", timeout=60)
    hfl_version = (version.stdout.strip().splitlines() or ["?"])[0]
    print(
        f"platform: {platform.system()} {platform.release()} {platform.machine()} · "
        f"Python {platform.python_version()} · {hfl_version}",
        flush=True,
    )
    server: subprocess.Popen[bytes] | None = None
    log = None
    try:
        for repo, quant, alias in (CHAT, EMBED):
            pulled = run.hfl("pull", repo, "-q", quant, "--alias", alias, "--skip-license")
            if not run.check(f"hfl pull {repo}", pulled.returncode == 0, pulled.stderr[-300:]):
                return 1
        port = _port()
        log = open(home / "serve.log", "wb")
        server = subprocess.Popen(
            [HFL, "serve", "--port", str(port)], env=env, stdout=log, stderr=subprocess.STDOUT
        )
        http = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=600)
        deadline = time.monotonic() + 120
        while True:
            if server.poll() is not None or time.monotonic() > deadline:
                said = (home / "serve.log").read_text(encoding="utf-8", errors="replace")
                run.check("hfl serve", False, said[-300:])
                return 1
            try:
                if http.get("/healthz", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        run.check("hfl serve", True, f"port {port}")
        _checks(run, http)
        served = (home / "serve.log").read_text(encoding="utf-8", errors="replace")
        backend = next(
            (b for b in ("llama-server", "llama.cpp", "MLX", "Transformers") if b in served), "?"
        )
        print(f"backend seen in the server log: {backend}", flush=True)
        outdated = run.hfl("outdated", timeout=120)
        run.check("hfl outdated", outdated.returncode == 0, outdated.stdout.strip()[-200:])
        removed = run.hfl("rm", "embed", "--yes", timeout=60)
        files = list((home / "models").rglob("*nomic*.gguf"))
        run.check("hfl rm", removed.returncode == 0 and not files, removed.stdout.strip()[-120:])
    finally:
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
        if log is not None:
            log.close()
        if args.keep:
            print(f"home kept: {home}")
        else:
            shutil.rmtree(home, ignore_errors=True)
    print("ALL OK" if not run.failed else "FAILURES")
    return 1 if run.failed else 0


if __name__ == "__main__":
    sys.exit(main())

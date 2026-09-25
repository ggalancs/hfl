#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Which models work with HFL, checked for real: ``docs/compatibility.md``.

HFL's promise is any model on the Hugging Face Hub, and each family writes
its prompts, tool calls and reasoning its own way. This script pulls a list
of models, serves each on every backend that can run it, and checks what a
client relies on:

- **chat** — a plain question gets a plain answer (no markers leaking);
- **tools** — the model calls a tool and then answers from its result, over
  the Ollama, OpenAI and Anthropic APIs, streamed and not (6 checks);
- **reasoning off** — for a model that reasons, ``think: false`` makes it
  generate far fewer tokens and still answer;
- **vision** — for a model that reports it, it reads an image (a red circle
  and "HFL 42", drawn here).

Models are pulled into ``--home`` (a scratch HFL home by default, never
~/.hfl) with ``hfl pull --yes``. A model whose license must be accepted is
not accepted here: that is its owner's call, so it is listed as such.

    python scripts/compat_matrix.py                 # the default list
    python scripts/compat_matrix.py --only qwen3    # names containing "qwen3"

Each server is started and stopped by the script, one at a time, with one
model resident. Check the power source first: this is not a benchmark, but
a laptop on battery makes it slow.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
# The ``hfl`` installed with this interpreter (the repo's venv).
HFL = [str(Path(sys.executable).with_name("hfl"))]


@dataclass(frozen=True)
class Model:
    alias: str
    reference: str
    family: str
    gguf: bool = True
    reasons: bool = False  # reasons by default, so "reasoning off" applies
    stops_reasoning: bool = True  # False: the family always reasons (R1)


# The most downloaded families, in sizes that run on a laptop. Licenses that
# need acceptance (Gemma, Llama) are left for their owner to accept.
DEFAULT_MODELS = [
    Model("c-qwen3", "hf.co/Qwen/Qwen3-1.7B-GGUF:Q8_0", "Qwen3", reasons=True),
    # 7B: Qwen's 3B builds carry its research license; these are Apache-2.0
    # (and Qwen2.5-7B's Q4_K_M comes in two parts).
    Model("c-qwen2.5", "hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", "Qwen2.5"),
    Model(
        "c-qwen2.5-coder",
        "hf.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M",
        "Qwen2.5-Coder",
    ),
    Model("c-llama3.2", "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", "Llama 3.2"),
    Model("c-mistral", "hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M", "Mistral"),
    Model("c-phi4-mini", "hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M", "Phi-4-mini"),
    Model("c-granite", "hf.co/ibm-granite/granite-3.3-2b-instruct-GGUF:Q4_K_M", "Granite 3.3"),
    Model("c-smollm3", "hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M", "SmolLM3", reasons=True),
    Model("c-hermes3", "hf.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF:Q4_K_M", "Hermes 3"),
    Model(
        "c-r1",
        "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M",
        "DeepSeek-R1 distill",
        reasons=True,
        stops_reasoning=False,
    ),
    Model("c-glm4", "hf.co/bartowski/THUDM_GLM-4-9B-0414-GGUF:Q4_K_M", "GLM-4"),
    Model("c-gpt-oss", "hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M", "gpt-oss", reasons=True),
    Model("c-qwen2.5-vl", "hf.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M", "Qwen2.5-VL"),
    Model(
        "c-qwen3-mlx",
        "hf.co/mlx-community/Qwen3-1.7B-4bit",
        "Qwen3 (MLX)",
        gguf=False,
        reasons=True,
    ),
    Model(
        "c-llama3.2-mlx",
        "hf.co/mlx-community/Llama-3.2-3B-Instruct-4bit",
        "Llama 3.2 (MLX)",
        gguf=False,
    ),
]

WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
TOOLS = [{"type": "function", "function": WEATHER}]
ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": WEATHER["description"],
        "input_schema": WEATHER["parameters"],
    }
]
ASK = "What is the weather in Paris right now? Use the tool."
RESULT = '{"temperature_c": 31, "sky": "thunderstorm"}'
MARKERS = re.compile(r"<\|[a-z_]+\|>|<tool_call>|</?think>|\[TOOL_CALLS\]")


# ------------------------------------------------------------------ server


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    """``hfl serve`` for one backend; stopped by the script that started it."""

    def __init__(self, home: Path, backend: str, log: Path) -> None:
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        env = {
            **os.environ,
            "HFL_HOME": str(home),
            "HFL_MAX_LOADED_MODELS": "1",
            "PYTHONUNBUFFERED": "1",
        }
        argv = [*HFL, "serve", "--port", str(self.port)]
        if backend != "default":
            argv += ["--backend", backend]
        self._log = open(log, "ab")
        self.proc = subprocess.Popen(
            argv, env=env, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        )
        deadline = time.monotonic() + 120
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"hfl serve exited; see {log}")
            try:
                if httpx.get(self.base + "/healthz", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"hfl serve did not start; see {log}")
            time.sleep(0.5)
        self.http = httpx.Client(base_url=self.base, timeout=600)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self._log.close()


# ------------------------------------------------------------------ checks


def _check(fn):
    """A check that raises is a failed check, with the reason kept."""

    def run(*args):
        try:
            ok, detail = fn(*args)
        except Exception as exc:  # noqa: BLE001 - the matrix records, not raises
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:200]
        return {"ok": bool(ok), "detail": str(detail)[:300]}

    return run


@_check
def check_chat(http: httpx.Client, model: str):
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "options": {"temperature": 0, "num_predict": 1500},
    }
    text = http.post("/api/chat", json=body).json()["message"]["content"]
    return "pong" in text.lower() and not MARKERS.search(text), text


def _ollama(http, model, messages, stream):
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": stream,
        "options": {"temperature": 0, "num_predict": 3000},
    }
    if not stream:
        m = http.post("/api/chat", json=body).json()["message"]
        return m.get("content", ""), m.get("tool_calls") or []
    content, calls = "", []
    with http.stream("POST", "/api/chat", json=body) as r:
        for line in r.iter_lines():
            if line:
                m = json.loads(line).get("message") or {}
                content += m.get("content") or ""
                calls += m.get("tool_calls") or []
    return content, calls


def _openai(http, model, messages, stream):
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "stream": stream,
        "temperature": 0,
        "max_tokens": 3000,
    }
    if not stream:
        m = http.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
        return m.get("content") or "", m.get("tool_calls") or []
    content, calls = "", {}
    with http.stream("POST", "/v1/chat/completions", json=body) as r:
        for line in r.iter_lines():
            if not line.startswith("data: {"):
                continue
            for choice in json.loads(line[6:]).get("choices") or []:
                delta = choice.get("delta") or {}
                content += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    entry = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    entry["name"] += fn.get("name") or ""
                    entry["arguments"] += fn.get("arguments") or ""
    return content, [{"function": c} for c in calls.values()]


def _anthropic(http, model, messages, stream):
    body = {
        "model": model,
        "messages": messages,
        "tools": ANTHROPIC_TOOLS,
        "stream": stream,
        "temperature": 0,
        "max_tokens": 3000,
    }
    if not stream:
        d = http.post("/v1/messages", json=body).json()
        text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
        calls = [
            {"function": {"name": b["name"], "arguments": b["input"]}}
            for b in d["content"]
            if b["type"] == "tool_use"
        ]
        return text, calls
    text, blocks = "", {}
    with http.stream("POST", "/v1/messages", json=body) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            e = json.loads(line[6:])
            if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use":
                blocks[e["index"]] = {"name": e["content_block"]["name"], "arguments": ""}
            elif e["type"] == "content_block_delta":
                d = e["delta"]
                if d["type"] == "text_delta":
                    text += d["text"]
                elif d["type"] == "input_json_delta":
                    blocks[e["index"]]["arguments"] += d["partial_json"]
    return text, [{"function": b} for b in blocks.values()]


def _history(api: str):
    if api == "ollama":
        call = {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        return [
            {"role": "user", "content": ASK},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": RESULT},
        ]
    if api == "anthropic":
        return [
            {"role": "user", "content": ASK},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": RESULT}],
            },
        ]
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }
    return [
        {"role": "user", "content": ASK},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": RESULT},
    ]


def _paris(calls) -> bool:
    if len(calls) != 1:
        return False
    fn = calls[0]["function"]
    args = fn["arguments"]
    args = json.loads(args) if isinstance(args, str) else args
    return fn["name"] == "get_weather" and str(args.get("city", "")).lower() == "paris"


@_check
def check_tools(http: httpx.Client, model: str, api: str, stream: bool):
    fn = {"ollama": _ollama, "openai": _openai, "anthropic": _anthropic}[api]
    _, calls = fn(http, model, [{"role": "user", "content": ASK}], stream)
    if not _paris(calls):
        return False, f"call: {calls}"
    answer, again = fn(http, model, _history(api), stream)
    good = "31" in answer and "thunder" in answer.lower() and not again
    return good and not MARKERS.search(answer), f"answer: {answer[:120]!r}"


@_check
def check_reasoning_off(http: httpx.Client, model: str):
    def ask(think):
        body = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
            "options": {"temperature": 0, "num_predict": 1500},
        }
        if think is not None:
            body["think"] = think
        d = http.post("/api/chat", json=body).json()
        return d.get("eval_count") or 0, d["message"].get("content", "")

    default, _ = ask(None)
    off, text = ask(False)
    return off < default / 2 and "391" in text, f"tokens {default} -> {off}, answer {text[:60]!r}"


def _image() -> str:
    """A red circle over the text "HFL 42", as base64 PNG."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 60, 260, 260), fill=(220, 20, 20))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 64)
    except OSError:
        font = ImageFont.load_default(size=64)
    draw.text((80, 320), "HFL 42", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@_check
def check_vision(http: httpx.Client, model: str):
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": "What colour is the circle, and what text is in the image? Be brief.",
                "images": [_image()],
            }
        ],
        "options": {"temperature": 0, "num_predict": 300},
    }
    text = http.post("/api/chat", json=body).json()["message"]["content"]
    return "red" in text.lower() and "42" in text, text


# -------------------------------------------------------------------- run


def pull(model: Model, home: Path, log: Path) -> str | None:
    """``None`` when pulled; otherwise why not."""
    env = {**os.environ, "HFL_HOME": str(home)}
    with open(log, "ab") as out:
        proc = subprocess.run(
            [*HFL, "pull", model.reference, "--alias", model.alias, "--yes"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=3600,
        )
    if proc.returncode == 0:
        return None
    text = log.read_text(errors="replace")[-2000:]
    if "accept the terms" in text:
        return "license to be accepted by its owner"
    return f"pull failed (exit {proc.returncode}); see {log.name}"


def run_model(server: Server, model: Model) -> dict:
    http = server.http
    caps = http.post("/api/show", json={"model": model.alias}).json().get("capabilities") or []
    result: dict = {"chat": check_chat(http, model.alias), "tools": {}}
    for api in ("ollama", "openai", "anthropic"):
        for stream in (False, True):
            result["tools"][f"{api}{' stream' if stream else ''}"] = check_tools(
                http, model.alias, api, stream
            )
    if model.reasons and model.stops_reasoning:
        result["reasoning_off"] = check_reasoning_off(http, model.alias)
    elif model.reasons:
        result["reasoning_off"] = {"ok": None, "detail": "always reasons (no switch)"}
    if "vision" in caps:
        result["vision"] = check_vision(http, model.alias)
    return result


def _cell(check: dict | None) -> str:
    if check is None:
        return "—"
    if check["ok"] is None:
        return "always"
    return "✓" if check["ok"] else "✗"


def render(results: dict, meta: dict) -> str:
    lines = [
        "# Model compatibility",
        "",
        "Generated by `scripts/compat_matrix.py` — every cell is a real request",
        "to a real model, not a claim. ✓ passed, ✗ failed (the reason is in the",
        f"[raw data]({meta['raw']})), — not applicable.",
        "",
        f"{meta['date']} · {meta['machine']} · HFL {meta['hfl']}",
        "",
        "- **Chat**: a plain question gets a plain answer, no markers leaking.",
        "- **Tools**: calls the tool, then answers from its result — Ollama,",
        "  OpenAI and Anthropic APIs, streamed and not (of 6).",
        "- **Reasoning off**: `think: false` cuts the tokens by half or more and",
        "  the answer is still right (models that reason by default); *always*:",
        "  the family has no switch (DeepSeek-R1 distills).",
        '- **Vision**: reads a red circle and the text "HFL 42" from an image.',
        "",
        "| Model | Family | Backend | Chat | Tools | Reasoning off | Vision |",
        "|---|---|---|:-:|:-:|:-:|:-:|",
    ]
    for key, entry in results.items():
        if "skipped" in entry:
            lines.append(
                f"| `{entry['reference']}` | {entry['family']} | — | {entry['skipped']} | | | |"
            )
            continue
        for backend, r in entry["backends"].items():
            if "error" in r:
                why = r["error"][:60]
                lines.append(
                    f"| `{entry['reference']}` | {entry['family']} | {backend} | ✗ {why} | | | |"
                )
                continue
            tools = sum(c["ok"] for c in r["tools"].values())
            lines.append(
                f"| `{entry['reference']}` | {entry['family']} | {backend} | {_cell(r['chat'])} | "
                f"{tools}/6 | {_cell(r.get('reasoning_off'))} | {_cell(r.get('vision'))} |"
            )
    return "\n".join(lines) + "\n"


def _hfl_version() -> str:
    out = subprocess.run([*HFL, "version"], capture_output=True, text=True).stdout
    found = re.search(r"v?(\d+\.\d+\.\d+\S*)", out)
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", "src"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    # A run on uncommitted code must not claim to be that commit.
    suffix = " + uncommitted changes" if dirty else ""
    return f"{found.group(1) if found else '?'} ({commit}{suffix})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--home", type=Path, default=None, help="HFL home (default: a temp dir)")
    parser.add_argument("--only", default="", help="only models whose alias contains this")
    parser.add_argument("--backends", default="default,llama-server")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "compatibility.md")
    args = parser.parse_args()

    home = args.home or Path(tempfile.mkdtemp(prefix="hfl-compat-"))
    home.mkdir(parents=True, exist_ok=True)
    logs = home / "compat-logs"
    logs.mkdir(exist_ok=True)
    models = [m for m in DEFAULT_MODELS if args.only in m.alias]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"HFL home: {home}", flush=True)

    results: dict = {}
    for model in models:
        why = pull(model, home, logs / f"pull-{model.alias}.log")
        entry = {"reference": model.reference, "family": model.family, "backends": {}}
        if why:
            entry["skipped"] = why
            print(f"{model.alias}: {why}", flush=True)
        results[model.alias] = entry
    for backend in backends:
        todo = [m for m in models if "skipped" not in results[m.alias]]
        todo = [m for m in todo if m.gguf or backend == "default"]
        if not todo:
            continue
        server = Server(home, backend, logs / f"serve-{backend}.log")
        try:
            for model in todo:
                started = time.monotonic()
                try:
                    r = run_model(server, model)
                except Exception as exc:  # noqa: BLE001
                    r = {"error": f"{type(exc).__name__}: {exc}"}
                r["seconds"] = round(time.monotonic() - started)
                results[model.alias]["backends"][backend] = r
                tools = sum(c["ok"] for c in r.get("tools", {}).values())
                print(f"{model.alias:18} {backend:12} tools {tools}/6  {r['seconds']}s", flush=True)
        finally:
            server.stop()

    raw = args.out.with_suffix(".json")
    meta = {
        "date": date.today().isoformat(),
        "machine": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        or os.uname().machine,
        "hfl": _hfl_version(),
        "raw": raw.name,
    }
    raw.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    args.out.write_text(render(results, meta))
    print(f"wrote {args.out} and {raw}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

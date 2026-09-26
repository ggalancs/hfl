# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section A: every ``hfl`` command, run as a user runs it."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import httpx
import tomllib
from local_audit import MODELS, QUESTION, Audit, Parts, check, expect

REPO = Path(__file__).resolve().parents[2]


# Pulled first: every other check uses these models.
@check("A24", "hfl pull")
def pull(a: Audit) -> str:
    done = []
    for alias, repo, quant, fmt in MODELS:
        out = a.ok("pull", repo, "-q", quant, "--alias", alias, "--format", fmt, timeout=1800)
        expect("Model ready" in out, f"{repo}: {out[-300:]}")
        done.append(alias)
    missing = a.fails_cleanly("pull", "hfl-audit/this-repo-does-not-exist-9f2c", timeout=120)
    return f"pulled {', '.join(done)}; a missing repo: {missing.strip().splitlines()[-1][:80]}"


@check("A17", "hfl list")
def list_(a: Audit) -> str:
    out = a.ok("list")
    for alias, *_ in MODELS:
        expect(f" {alias} " in out, f"{alias} not listed: {out[-400:]}")
    supported = a.ok("list", "--supported-only")
    expect(" embed " in supported, "--supported-only hid the embedding model (a supported type)")
    return f"{len(MODELS)} models listed, with aliases"


@check("A7", "hfl cp")
def cp(a: Audit) -> str:
    a.ok("cp", "chat", "chat-copy")
    expect(" chat-copy " in a.ok("list") or "chat-copy" in a.ok("list"), "copy not listed")
    a.fails_cleanly("cp", "no-such-model", "x")
    a.fails_cleanly("cp", "chat", "chat-copy")  # the name is taken
    return "copied; missing source and taken name refused"


@check("A1", "hfl alias")
def alias(a: Audit) -> str:
    a.ok("alias", "chat-copy", "cc")
    expect(re.search(r"chat-copy\s.*cc|cc\s", a.ok("list")), "alias not shown in list")
    a.fails_cleanly("alias", "no-such-model", "zz")
    return "alias set and listed; missing model refused"


@check("A15", "hfl inspect")
def inspect(a: Audit) -> str:
    out = a.ok("inspect", "chat")
    expect("Qwen/Qwen2.5-0.5B-Instruct-GGUF" in out and "apache" in out.lower(), out[-400:])
    a.fails_cleanly("inspect", "no-such-model")
    return "repo, license shown; missing model refused"


@check("A34", "hfl show")
def show(a: Audit) -> str:
    part = Parts()
    part(
        "summary",
        lambda: "capabilities" in a.ok("show", "chat").lower() or expect(False, "no capabilities"),
    )
    part("--template", lambda: expect("{" in a.ok("show", "chat", "--template"), "empty"))
    part("--modelfile", lambda: expect("FROM" in a.ok("show", "chat", "--modelfile"), "no FROM"))
    part("--parameters", lambda: a.ok("show", "chat", "--parameters") is not None)
    part(
        "--license",
        lambda: expect("apache" in a.ok("show", "chat", "--license").lower(), "no license"),
    )
    part("missing model refused", lambda: a.fails_cleanly("show", "no-such-model"))
    return part.verdict()


@check("A28", "hfl run")
def run_(a: Audit) -> str:
    out = a.ok("run", "chat", stdin=f"{QUESTION}\n/bye\n", timeout=300)
    expect("paris" in out.lower(), f"no answer: {out[-400:]}")
    styled = a.ok("run", "chat", "--system", "Answer only in French.", stdin="Say hello.\n/bye\n")
    expect(re.search(r"bonjour|salut", styled, re.I), f"--system ignored: {styled[-300:]}")
    a.ok("run", "chat", "--session", "audit-s", stdin="Remember the word tangerine.\n/bye\n")
    a.fails_cleanly("run", "no-such-model-xyz", timeout=120)
    return "answered; --system honoured; --session recorded; missing model refused"


@check("A31", "hfl sessions list")
def sessions_list(a: Audit) -> str:
    out = a.ok("sessions", "list")
    expect("audit-s" in out, f"session not listed: {out[-300:]}")
    return "the session recorded by hfl run is listed"


@check("A33", "hfl sessions show")
def sessions_show(a: Audit) -> str:
    out = a.ok("sessions", "show", "audit-s")
    expect("tangerine" in out.lower(), f"messages not shown: {out[-300:]}")
    a.fails_cleanly("sessions", "show", "no-such-session")
    return "its messages; a missing session refused"


@check("A32", "hfl sessions rm")
def sessions_rm(a: Audit) -> str:
    a.ok("sessions", "rm", "audit-s")
    expect("audit-s" not in a.ok("sessions", "list"), "still listed after rm")
    return "removed"


@check("A39", "hfl verify")
def verify(a: Audit) -> str:
    part = Parts()
    part(
        "chat passes",
        lambda: expect("fail" not in a.ok("verify", "chat", timeout=300).lower(), "a check failed"),
    )
    part("missing model refused", lambda: a.fails_cleanly("verify", "no-such-model"))
    return part.verdict()


@check("A2", "hfl bench")
def bench(a: Audit) -> str:
    part = Parts()

    def measured() -> None:
        out = a.ok("bench", "chat", "-n", "1", "-t", "8", "--lengths", "16", timeout=300)
        row = re.search(r"│\s*16\s*│\s*1\s*│\s*([\d.]+)\s*│\s*[\d.]+\s*│\s*([\d.]+)", out)
        expect(row and float(row.group(2)) > 0, f"no tokens/s row: {out[-300:]}")

    part("measures ttft and tokens/s", measured)
    part("missing model refused", lambda: a.fails_cleanly("bench", "no-such-model"))
    return part.verdict()


@check("A3", "hfl check")
def check_(a: Audit) -> str:
    out = a.ok("check")
    expect("llama" in out.lower(), out[-300:])
    return out.strip().splitlines()[-1][:120]


@check("A11", "hfl doctor")
def doctor(a: Audit) -> str:
    out = a.ok("doctor")
    expect("metal" in out.lower(), f"Metal not detected: {out[-400:]}")
    return "Metal detected; extras listed"


@check("A9", "hfl debug")
def debug(a: Audit) -> str:
    out = a.ok("debug")
    expect(str(a.home) in out or "home" in out.lower(), out[-300:])
    return "runs"


@check("A6", "hfl config")
def config(a: Audit) -> str:
    out = a.ok("config")
    expect(str(a.home) in out, f"HFL_HOME not shown: {out[-400:]}")
    return "shows the audit's HFL_HOME"


@check("A13", "hfl help")
def help_(a: Audit) -> str:
    a.ok("help")
    out = a.ok("help", "--extras")
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    missing = [e for e in extras if e not in ("dev", "build", "all") and e not in out]
    expect(not missing, f"extras not in `hfl help --extras`: {missing}")
    return f"all {len(extras)} extras named"


@check("A40", "hfl version")
def version(a: Audit) -> str:
    out = a.ok("version")
    wanted = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    expect(wanted in out, f"version {wanted} not in: {out}")
    return out.strip().splitlines()[0]


@check("A4", "hfl compliance-dashboard")
def compliance_dashboard(a: Audit) -> str:
    out = a.ok("compliance-dashboard")
    expect("apache" in out.lower(), out[-400:])
    return "licenses shown"


@check("A5", "hfl compliance-report")
def compliance_report(a: Audit) -> str:
    part = Parts()
    target, md = a.scratch / "report.json", a.scratch / "report.md"

    def as_json() -> None:
        a.ok("compliance-report", "--output", str(target))
        expect(json.loads(target.read_text()), "empty report")

    def as_markdown() -> None:
        a.ok("compliance-report", "--output", str(md), "--format", "markdown")
        expect(md.read_text().strip().startswith("#"), md.read_text()[:100])

    part("json", as_json)
    part("markdown", as_markdown)
    part(
        "unknown format refused",
        lambda: a.fails_cleanly("compliance-report", "--output", str(md), "--format", "pdf"),
    )
    return part.verdict()


@check("A10", "hfl discover")
def discover(a: Audit) -> str:
    out = a.ok("discover", "qwen", "--limit", "3", timeout=120)
    expect("qwen" in out.lower(), out[-400:])
    return "Hub results"


@check("A26", "hfl recommend")
def recommend(a: Audit) -> str:
    out = a.ok("recommend", "-t", "chat", "-n", "3", timeout=180)
    expect("/" in out, f"no repo ids: {out[-400:]}")
    a.fails_cleanly("recommend", "-t", "no-such-task")
    return "suggestions; an unknown task refused"


@check("A12", "hfl draft-recommend")
def draft_recommend(a: Audit) -> str:
    out = a.ok("draft-recommend", "Qwen/Qwen2.5-7B-Instruct", timeout=180)
    expect("qwen" in out.lower(), out[-400:])
    return out.strip().splitlines()[-1][:120]


@check("A29", "hfl search")
def search(a: Audit) -> str:
    out = a.ok("search", "qwen2.5", "-l", "5", "-n", "5", "--gguf", stdin="q\n", timeout=120)
    expect("qwen" in out.lower(), out[-400:])
    a.fails_cleanly("search", "ab")  # fewer than 3 characters
    return "results; a too-short query refused"


@check("A14", "hfl import")
def import_(a: Audit) -> str:
    elsewhere = a.scratch / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    source = a.model_file("Qwen--Qwen2.5-0.5B-Instruct-GGUF")
    copy = elsewhere / "Imported-Qwen-Q4_K_M.gguf"
    shutil.copyfile(source, copy)
    a.ok("import", str(copy), "--alias", "imp")
    answer = a.ok("run", "imp", stdin=f"{QUESTION}\n/bye\n", timeout=300)
    expect("paris" in answer.lower(), answer[-300:])
    a.ok("rm", "imp", "--yes")
    expect(copy.exists(), "hfl rm deleted an imported file")
    a.fails_cleanly("import", str(elsewhere / "missing.gguf"))
    return "imported in place, answered; rm kept the file; a missing path refused"


@check("A27", "hfl rm")
def rm(a: Audit) -> str:
    a.ok("rm", "chat-copy", "--yes")
    expect("chat-copy" not in a.ok("list"), "still listed")
    a.fails_cleanly("rm", "no-such-model", "--yes")
    return "removed; missing model refused"


@check("A22", "hfl outdated")
def outdated(a: Audit) -> str:
    out = a.ok("outdated", timeout=180)
    expect("up to date" in out, out[-400:])
    offline = a.cli("outdated", env={"HF_ENDPOINT": "http://hub.invalid"}, timeout=120)
    expect(offline.returncode == 1 and "could not check" in offline.stdout, offline.stdout[-300:])
    return "up to date; offline says could not check (exit 1)"


@check("A18", "hfl login")
def login(a: Audit) -> str:
    bad = a.cli("login", "--token", "hf_not_a_real_token_000000000000000000", timeout=60)
    out = bad.stdout + bad.stderr
    expect("Traceback" not in out, out[-400:])
    expect(bad.returncode != 0, f"an invalid token was accepted: {out[-300:]}")
    return "an invalid token refused with a message (the user's own token is never touched)"


@check("A19", "hfl logout")
def logout(a: Audit) -> str:
    out = a.cli("logout", timeout=60)
    text = out.stdout + out.stderr
    expect("Traceback" not in text, text[-300:])
    return f"exit {out.returncode}: {text.strip().splitlines()[-1][:100] if text.strip() else ''}"


@check("A16", "hfl launch")
def launch(a: Audit) -> str:
    out = a.ok("launch", "claude", "-m", "chat", "--print", timeout=120)
    expect("ANTHROPIC_BASE_URL" in out, out[-400:])
    codex = a.ok("launch", "codex", "-m", "chat", "--print", timeout=120)
    expect("hfl" in codex.lower(), codex[-300:])
    a.fails_cleanly("launch", "not-an-agent", "-m", "chat")
    return (
        "settings printed for claude and codex; an unknown agent refused"
        " (full runs: agent_check.py)"
    )


@check("A30", "hfl serve")
def serve(a: Audit) -> str:
    with a.server("--model", "chat") as base:
        loaded = httpx.get(base + "/api/ps", timeout=30).json()
        expect(loaded.get("models"), f"--model did not preload: {loaded}")
    with a.server("--api-key", "audit-key") as base:
        expect(httpx.get(base + "/api/tags", timeout=30).status_code == 401, "no key: not 401")
        good = httpx.get(base + "/api/tags", headers={"Authorization": "Bearer audit-key"})
        expect(good.status_code == 200, good.text[:200])
    refused = a.cli("serve", "--host", "0.0.0.0", "--port", "1", timeout=60)
    expect(refused.returncode == 1, "bound 0.0.0.0 unattended with no key")
    return "--model preloads; --api-key enforced; unattended 0.0.0.0 refused"


@check("A23", "hfl ps")
def ps(a: Audit) -> str:
    with a.server("--model", "chat") as _:
        out = a.ok("ps", "--port", str(a.port))
        expect("qwen" in out.lower() or "chat" in out, out[-300:])
    down = a.cli("ps", "--port", "9")
    expect(
        down.returncode != 0 and "Traceback" not in down.stdout + down.stderr, down.stdout[-200:]
    )
    return "lists the loaded model; no server: a clear error"


@check("A37", "hfl stop")
def stop(a: Audit) -> str:
    with a.server("--model", "chat") as base:
        a.ok("stop", "chat", "--port", str(a.port))
        left = httpx.get(base + "/api/ps", timeout=30).json().get("models")
        expect(not left, f"still loaded: {left}")
    return "unloaded"


@check("A8", "hfl create")
def create(a: Audit) -> str:
    part = Parts()
    modelfile = a.scratch / "Modelfile"
    modelfile.write_text('FROM chat\nSYSTEM "Answer only in French."\nPARAMETER temperature 0\n')
    bad = a.scratch / "Bad"
    bad.write_text("FROM no-such-base\n")
    with a.server() as base:
        port = str(a.port)
        part(
            "created",
            lambda: a.ok("create", "french", "-f", str(modelfile), "--port", port, timeout=300),
        )

        def system_applied() -> None:
            reply = httpx.post(
                base + "/api/chat",
                json={
                    "model": "french",
                    "stream": False,
                    "messages": [{"role": "user", "content": "Say hello."}],
                },
                timeout=300,
            ).json()
            text = reply.get("message", {}).get("content", "")
            expect(re.search(r"bonjour|salut", text, re.I), f"SYSTEM not applied: {text[:80]}")

        part("its SYSTEM applies", system_applied)
        part(
            "missing base refused",
            lambda: a.fails_cleanly("create", "bad", "-f", str(bad), "--port", port, timeout=120),
        )
    a.cli("rm", "french", "--yes")
    return part.verdict()


@check("A20", "hfl lora")
def lora(a: Audit) -> str:
    """``hfl lora`` has no --host/--port: it loads its own copy of the model
    in the CLI process. Checked against a running server of the same home."""
    from huggingface_hub import hf_hub_download

    adapters = a.home / "adapters"
    adapters.mkdir(exist_ok=True)
    adapter = adapters / "moe_shakespeare15M.gguf"
    if not adapter.exists():
        source = hf_hub_download("ggml-org/stories15M_MOE", "moe_shakespeare15M.gguf")
        shutil.copyfile(source, adapter)
    with a.server() as base:
        httpx.post(
            base + "/api/generate",
            json={
                "model": "stories",
                "prompt": "a",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=120,
        )
        out = a.ok("lora", "apply", "stories", "--path", str(adapter))
        seen = httpx.get(base + "/api/lora/stories", timeout=30).json().get("adapters")
        expect(
            seen,
            "the command applied the adapter to a model it loaded itself, in the CLI "
            f"process, and exited; the running server has none ({out.strip()[-80:]})",
        )
    return "applied to the running server"


@check("A35", "hfl snapshot")
def snapshot(a: Audit) -> str:
    """Like ``hfl lora``: no --host/--port, a model of its own in the CLI."""
    with a.server("--model", "chat") as base:
        httpx.post(
            base + "/api/generate",
            json={
                "model": "chat",
                "prompt": "The capital of France",
                "stream": False,
                "options": {"num_predict": 8},
            },
            timeout=120,
        )
        out = a.ok("snapshot", "save", "chat", "--name", "audit-snap", timeout=300)
        tokens = re.search(r"tokens=(\d+)", out)
        expect(
            tokens and int(tokens.group(1)) > 0,
            "the CLI saved the KV of a model it loaded "
            f"itself — empty — not the server's: {out.strip()[-120:]}",
        )
        expect("audit-snap" in a.ok("snapshot", "list"), "not listed")
        a.ok("snapshot", "delete", "--name", "audit-snap")
    return "saved the server's KV, listed, deleted"


@check("A21", "hfl mcp")
def mcp(a: Audit) -> str:
    import subprocess

    part = Parts()
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "audit", "version": "1"},
        },
    }

    def serves() -> None:
        proc = subprocess.run(
            [a.hfl, "mcp", "serve", "--transport", "stdio"],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
            env=a.env,
        )
        reply = next((json.loads(x) for x in proc.stdout.splitlines() if x.startswith("{")), None)
        expect(
            reply and reply.get("result", {}).get("serverInfo"),
            f"no initialize reply: {(proc.stdout + proc.stderr).strip()[-200:]}",
        )

    def lists() -> None:
        listed = a.cli("mcp", "list", timeout=60)
        expect(
            listed.returncode == 0 and "Traceback" not in listed.stdout + listed.stderr,
            (listed.stdout + listed.stderr)[-200:],
        )

    part("serve over stdio answers initialize", serves)
    part("list", lists)
    part("unknown action refused", lambda: a.fails_cleanly("mcp", "not-an-action"))
    return part.verdict()


@check("A25", "hfl pull-smart")
def pull_smart(a: Audit) -> str:
    out = a.ok("pull-smart", "Qwen/Qwen2.5-0.5B-Instruct", timeout=1800)
    expect('"status":"success"' in out.replace(" ", ""), out[-400:])
    chosen = re.search(r"Downloading (\S+)", out)
    listed = a.ok("list")
    expect(chosen and chosen.group(1).split("/")[-1].lower()[:20] in listed.lower(), listed[-400:])
    return f"chose {chosen.group(1)} for this machine; registered"

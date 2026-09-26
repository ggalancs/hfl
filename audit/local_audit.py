#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HFL audited end to end on this machine: every command, every route.

    python audit/local_audit.py --work ~/hfl-audit --setup   # build + install once
    python audit/local_audit.py --work ~/hfl-audit           # every check
    python audit/local_audit.py --work ~/hfl-audit --only A,B3   # a section, or ids

See ``audit/README.md``. The report lands in ``<work>/REPORT.md``.

Each check runs the real thing — the installed ``hfl``, a real server, small
Apache-2.0 / MIT models — and records one of: ``OK`` (with its evidence),
``ROTO`` (with the output), ``NO COMPROBABLE AQUÍ`` (and why) or ``REQUIERE
PERMISO``. A check that raised is ``ROTO``, never skipped. Results merge into
``<work>/results.json`` and, with ``--doc``, into the audit's tables.

Isolation: its own ``HFL_HOME`` and ``HF_HOME`` under ``--work`` (``hfl
login``/``logout`` never touch the user's Hugging Face token), no ``HF_TOKEN``.
Every server it starts is stopped when the check ends.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

OK, BROKEN, UNCHECKABLE, PERMISSION = "OK", "ROTO", "NO COMPROBABLE AQUÍ", "REQUIERE PERMISO"


class Broken(AssertionError):
    """A check's finding: what it saw instead of what it expected."""


class Uncheckable(Exception):
    """Cannot be checked on this machine; the message says why."""


def expect(condition: object, evidence: object) -> None:
    if not condition:
        raise Broken(str(evidence)[:600])


APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def need_apple_silicon(what: str) -> None:
    """MLX, its hints and macOS speech exist only on Apple Silicon."""
    if not APPLE_SILICON:
        raise Uncheckable(f"{what}: needs macOS on Apple Silicon")


def need_llama_server() -> str:
    """llama.cpp's own server binary, which no Python extra installs."""
    found = shutil.which("llama-server")
    if found is None:
        raise Uncheckable("needs llama.cpp's llama-server on PATH (e.g. `brew install llama.cpp`)")
    return found


class Parts:
    """A check with several parts: each one checked, every failure listed —
    a check that stops at its first failure leaves the rest unknown."""

    def __init__(self) -> None:
        self.fine: list[str] = []
        self.problems: list[str] = []
        self.unchecked: list[str] = []

    def __call__(self, name: str, fn: Callable[[], object]) -> None:
        try:
            if fn() is False:  # None is fine: ``expect`` returns it when it holds
                raise Broken("false")
            self.fine.append(name)
        except Uncheckable as exc:  # neither fine nor broken: not looked at
            self.unchecked.append(f"{name} ({exc})")
        except Exception as exc:  # a part that crashed failed too
            self.problems.append(f"{name}: {' '.join(str(exc).split())[:160]}")

    def verdict(self) -> str:
        expect(not self.problems, "; ".join(self.problems) + f" (fine: {', '.join(self.fine)})")
        if not self.fine:
            raise Uncheckable("; ".join(self.unchecked) or "no part ran")
        tail = f" — NOT checked: {'; '.join(self.unchecked)}" if self.unchecked else ""
        return ", ".join(self.fine) + tail


@dataclass
class Check:
    cid: str
    title: str
    fn: Callable[["Audit"], str]
    needs: tuple[str, ...] = ()


REGISTRY: list[Check] = []


def check(
    cid: str, title: str, needs: tuple[str, ...] = ()
) -> Callable[[Callable[["Audit"], str]], Callable[["Audit"], str]]:
    """Register a check. ``needs``: checks whose results it reads (run first)."""

    def register(fn: Callable[["Audit"], str]) -> Callable[["Audit"], str]:
        REGISTRY.append(Check(cid, title, fn, needs))
        return fn

    return register


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Audit:
    def __init__(self, hfl: Path, work: Path) -> None:
        self.hfl = str(hfl)
        self.python = str(hfl.with_name("python"))
        self.work = work
        self.home = work / "home"
        self.scratch = work / "scratch"
        for folder in (self.home, self.scratch, work / "hf", work / "logs"):
            folder.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("HFL_", "OLLAMA_"))}
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
        self.env = {
            **env,
            "HFL_HOME": str(self.home),
            "HF_HOME": str(work / "hf"),
            "HFL_LANG": "en",
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "COLUMNS": "200",
            "NO_COLOR": "1",
        }
        self.port = 0  # of the running server, if any
        self._shared: tuple[Any, str] | None = None

    # -- the CLI --------------------------------------------------------------

    def cli(
        self, *args: str, stdin: str | None = None, timeout: float = 600, env: dict | None = None
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self.hfl, *args],
                input=stdin,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env={**self.env, **(env or {})},
                stdin=None if stdin is not None else subprocess.DEVNULL,
                cwd=self.scratch,
            )
        except subprocess.TimeoutExpired as exc:
            raise Broken(f"`hfl {' '.join(args)}` did not finish in {timeout}s") from exc

    def ok(self, *args: str, **kwargs: Any) -> str:
        """``hfl args`` must succeed; its output."""
        done = self.cli(*args, **kwargs)
        out = done.stdout + done.stderr
        expect(done.returncode == 0, f"`hfl {' '.join(args)}` exit {done.returncode}: {out[-500:]}")
        return out

    def fails_cleanly(self, *args: str, **kwargs: Any) -> str:
        """``hfl args`` must fail with a message, not a traceback."""
        done = self.cli(*args, **kwargs)
        out = done.stdout + done.stderr
        expect(done.returncode != 0, f"`hfl {' '.join(args)}` succeeded: {out[-300:]}")
        expect("Traceback" not in out, f"`hfl {' '.join(args)}` traceback: {out[-500:]}")
        expect(out.strip(), f"`hfl {' '.join(args)}` failed silently")
        return out

    # -- a server -------------------------------------------------------------

    @contextmanager
    def server(self, *args: str, env: dict | None = None, ready: float = 120) -> Iterator[str]:
        """``hfl serve`` on a free port; its base URL. Always stopped."""
        port = _port()
        log_path = self.work / "logs" / f"serve-{port}.log"
        log = open(log_path, "wb")
        proc = subprocess.Popen(
            [self.hfl, "serve", "--port", str(port), *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**self.env, **(env or {})},
            cwd=self.scratch,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + ready
            while True:
                if proc.poll() is not None:
                    raise Broken(f"hfl serve exited: {log_path.read_text(errors='replace')[-500:]}")
                try:
                    if httpx.get(base + "/healthz", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise Broken("hfl serve did not answer /healthz in time")
                time.sleep(0.3)
            self.port = port
            yield base
        finally:
            self.port = 0
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
            log.close()

    def shared(self) -> str:
        """One server for the checks that do not need one of their own."""
        if self._shared is None:
            # Without the rate limit: section B sends more than its default 60
            # requests a minute, and the checks after the 60th got 429s. The
            # limit itself is checked on servers of its own (D42-D44).
            manager = self.server(env={"HFL_RATE_LIMIT_ENABLED": "false"})
            self._shared = (manager, manager.__enter__())
        self.port = int(self._shared[1].rsplit(":", 1)[1])
        return self._shared[1]

    def close(self) -> None:
        if self._shared is not None:
            manager, _ = self._shared
            self._shared = None
            manager.__exit__(None, None, None)

    def http(self, base: str, timeout: float = 600, **kwargs: Any) -> httpx.Client:
        return httpx.Client(base_url=base, timeout=timeout, **kwargs)

    def wheel(self) -> Path:
        """The wheel ``--setup`` built and installed."""
        record = self.work / "wheel.txt"
        if not record.exists():
            raise Uncheckable("no wheel recorded: run with --setup first")
        return Path(record.read_text().strip())

    # -- models ---------------------------------------------------------------

    def model_file(self, repo_dir: str, pattern: str = "*.gguf") -> Path:
        files = sorted((self.home / "models" / repo_dir).rglob(pattern))
        files = [f for f in files if "mmproj" not in f.name]
        expect(files, f"no {pattern} under models/{repo_dir}")
        return files[0]


# The models the checks share: (alias, repo, pull arguments). All Apache-2.0 or
# MIT, so no license is accepted on anyone's behalf. Pulled once, before any
# check, so a check never depends on another having run first.
MODELS = [
    ("chat", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", ("-q", "Q4_K_M")),
    ("think", "Qwen/Qwen3-0.6B-GGUF", ("-q", "Q8_0")),
    ("vision", "ggml-org/SmolVLM-256M-Instruct-GGUF", ("-q", "Q8_0")),
    ("embed", "nomic-ai/nomic-embed-text-v1.5-GGUF", ("-q", "Q4_K_M")),
    ("stories", "ggml-org/stories15M_MOE", ("-q", "F16")),
    ("hfq", "Qwen/Qwen2.5-0.5B-Instruct", ("--format", "safetensors")),
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2", ("--format", "safetensors")),
    ("bark", "suno/bark-small", ()),
]
if APPLE_SILICON:
    MODELS.append(("mlxq", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", ()))
QUESTION = "What is the capital of France? Answer with one word."


def registered_aliases(home: Path) -> set[str]:
    path = home / "models.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    models = data if isinstance(data, list) else data.get("models", [])
    return {m["alias"] for m in models if m.get("alias")}


def prepare(audit: Audit) -> None:
    """Pull every shared model that is missing. A model that cannot be pulled
    stops the run: every result after it would be about the missing model."""
    have = registered_aliases(audit.home)
    for alias, repo, extra in MODELS:
        if alias in have:
            continue
        print(f"preparing: hfl pull {repo} --alias {alias}", flush=True)
        done = audit.cli("pull", repo, *extra, "--alias", alias, timeout=3600)
        if done.returncode != 0:
            raise SystemExit(f"cannot pull {repo}: {(done.stdout + done.stderr)[-400:]}")


def load_checks() -> None:
    """Import the check modules (each registers its checks)."""
    here = Path(__file__).resolve().parent / "audit_checks"
    sys.path.insert(0, str(here.parent))
    # The check modules import ``local_audit``; run as a script this module
    # is ``__main__``, and a second copy would have a registry of its own.
    sys.modules.setdefault("local_audit", sys.modules[__name__])
    for module in sorted(here.glob("[a-z]*.py")):
        __import__(f"audit_checks.{module.stem}")


def selected(only: str | None) -> list[Check]:
    """The checks asked for, each preceded by the checks it ``needs``."""
    if not only:
        return REGISTRY
    wanted = [w.strip() for w in only.split(",") if w.strip()]

    def match(cid: str) -> bool:
        return any(cid == w or (w.isalpha() and cid.startswith(w)) for w in wanted)

    by_id = {c.cid: c for c in REGISTRY}
    chosen: list[Check] = []

    def add(item: Check) -> None:
        if item in chosen:
            return
        for need in item.needs:
            add(by_id[need])
        chosen.append(item)

    for item in REGISTRY:
        if match(item.cid):
            add(item)
    return chosen


def run(audit: Audit, checks: list[Check]) -> dict[str, dict]:
    results_path = audit.work / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for item in checks:
        started = time.monotonic()
        try:
            evidence = item.fn(audit)
            status = OK
        except Broken as exc:
            status, evidence = BROKEN, str(exc)
        except Uncheckable as exc:
            status, evidence = UNCHECKABLE, str(exc)
        except PermissionError as exc:
            status, evidence = PERMISSION, str(exc)
        except Exception:  # a check that crashed found something too
            status, evidence = BROKEN, "check crashed: " + traceback.format_exc()[-600:]
        seconds = time.monotonic() - started
        results[item.cid] = {
            "title": item.title,
            "status": status,
            "evidence": " ".join(str(evidence).split())[:700],
            "seconds": round(seconds, 1),
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        print(
            f"{status:20} {item.cid:6} {item.title[:50]:50} {seconds:6.1f}s  "
            f"{results[item.cid]['evidence'][:110]}",
            flush=True,
        )
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def render(doc: Path, results: dict[str, dict]) -> None:
    """Write each result's status into an audit document's rows (by id)."""
    lines = doc.read_text().splitlines()
    out = []
    for line in lines:
        found = re.match(r"^\| ([A-F]\d+) \|", line)
        if found and found.group(1) in results:
            result = results[found.group(1)]
            cells = line.rstrip("|").split("|")
            evidence = result["evidence"].replace("|", "/")[:160]
            cells[-1] = f" **{result['status']}** — {evidence} "
            line = "|".join(cells) + "|"
        out.append(line)
    doc.write_text("\n".join(out) + "\n")


SECTIONS = {
    "A": "CLI commands",
    "B": "API routes",
    "C": "install extras",
    "D": "environment variables",
    "E": "engines",
    "F": "ways to install and run",
}


TABLE_HEAD = ["| id | check | status | evidence |", "|---|---|---|---|"]


def _order(cid: str) -> tuple[str, int]:
    return cid[0], int(re.sub(r"\D", "", cid) or 0)


def report(work: Path, results: dict[str, dict]) -> Path:
    """``<work>/REPORT.md``: the counts, what is not OK first, then every row."""
    counts: dict[str, int] = {}
    for result in results.values():
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    lines = [
        "# HFL local audit",
        "",
        f"{time.strftime('%Y-%m-%d %H:%M')} · {platform.platform()} · "
        + " · ".join(f"**{n} {s}**" for s, n in sorted(counts.items())),
        "",
        "Statuses: `OK` (with its evidence) · `ROTO` = broken (with the output) · "
        "`NO COMPROBABLE AQUÍ` = cannot be checked on this machine (and why) · "
        "`REQUIERE PERMISO` = would act outward or on the owner's behalf.",
        "",
    ]
    failing = [c for c in sorted(results, key=_order) if results[c]["status"] != OK]
    if failing:
        lines += ["## Not OK", "", *TABLE_HEAD]
        for cid in failing:
            r = results[cid]
            evidence = r["evidence"].replace("|", "/")
            lines.append(f"| {cid} | {r['title']} | {r['status']} | {evidence} |")
        lines.append("")
    for letter, name in SECTIONS.items():
        ids = [c for c in sorted(results, key=_order) if c.startswith(letter)]
        if not ids:
            continue
        lines += [f"## {letter}. {name}", "", *TABLE_HEAD]
        for cid in ids:
            r = results[cid]
            evidence = r["evidence"].replace("|", "/")[:200]
            lines.append(f"| {cid} | {r['title']} | {r['status']} | {evidence} |")
        lines.append("")
    path = work / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


# Every extra that installs on the platform; ``mlx`` only on Apple Silicon.
SETUP_EXTRAS = "llama,transformers,tts,stt,imagegen,convert,mcp,audio,tray,otel"


def setup(work: Path, python: str) -> None:
    """Build the wheel from this checkout and install it the way a user would:
    ``<work>/venv`` with the extras, ``<work>/venv-core`` with none (the
    install that has no llama-cpp-python, as Homebrew's)."""
    repo = Path(__file__).resolve().parents[1]
    if shutil.which("uv") is None:
        raise SystemExit("--setup needs uv: https://docs.astral.sh/uv/")
    work.mkdir(parents=True, exist_ok=True)
    # A fresh folder per build (born empty, nothing to delete first); the
    # wheel it holds is recorded for the checks that install it again.
    dist = Path(tempfile.mkdtemp(prefix="dist-", dir=work))
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist), str(repo)], check=True)
    wheel = next(dist.glob("hfl-*.whl"))
    (work / "wheel.txt").write_text(str(wheel))
    extras = SETUP_EXTRAS + (",mlx" if APPLE_SILICON else "")
    for venv, spec in ((work / "venv", f"{wheel}[{extras}]"), (work / "venv-core", str(wheel))):
        if not (venv / "bin" / "python").exists():
            subprocess.run(["uv", "venv", "-q", "--python", python, str(venv)], check=True)
        python_bin = str(venv / "bin" / "python")
        # --reinstall-package: a rebuilt wheel keeps its version number, and
        # uv would otherwise keep the copy already installed.
        subprocess.run(
            ["uv", "pip", "install", "--reinstall-package", "hfl", "--python", python_bin, spec],
            check=True,
        )
    print(f"installed {wheel.name} into {work / 'venv'} (+ extras) and {work / 'venv-core'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, required=True, help="where everything it makes goes")
    parser.add_argument("--hfl", type=Path, help="the hfl to audit (default: <work>/venv/bin/hfl)")
    parser.add_argument(
        "--setup", action="store_true", help="build + install HFL into <work>, stop"
    )
    parser.add_argument("--python", default="3.12", help="Python for --setup's venvs")
    parser.add_argument("--only", help="sections (A,B,...) or ids (B3,D14), comma-separated")
    parser.add_argument("--doc", type=Path, help="also write statuses into this document's rows")
    parser.add_argument("--list", action="store_true", help="list the checks and stop")
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    if args.setup:
        setup(work, args.python)
        return 0
    load_checks()
    checks = selected(args.only)
    if args.list:
        for item in checks:
            print(item.cid, item.title)
        return 0
    hfl = (args.hfl or work / "venv" / "bin" / "hfl").expanduser().resolve()
    if not hfl.exists():
        raise SystemExit(f"{hfl} does not exist: run with --setup first, or pass --hfl")
    audit = Audit(hfl, work)
    try:
        prepare(audit)
        results = run(audit, checks)
    finally:
        audit.close()
    if args.doc:
        render(args.doc, results)
    print(f"report: {report(work, results)}")
    broken = [c.cid for c in checks if results[c.cid]["status"] == BROKEN]
    print(f"\n{len(checks)} checked · {len(broken)} broken: {', '.join(broken) or 'none'}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

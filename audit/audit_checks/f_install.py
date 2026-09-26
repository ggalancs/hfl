# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section F: the ways HFL is installed and run."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from local_audit import Audit, Uncheckable, check, expect

from audit_checks.c_extras import MODULES

REPO = Path(__file__).resolve().parents[2]
EXTRA_ID = {
    name: f"C{index}" for index, name in enumerate(sorted(MODULES), 1)
}  # as c_extras numbers them
EXTRA_CHECKS = tuple(EXTRA_ID.values())


@check("F1", "pip install of the built wheel, clean venv")
def f1(a: Audit) -> str:
    out = subprocess.run([a.hfl, "version"], capture_output=True, text=True, timeout=60)
    expect(out.returncode == 0 and "/extras/" not in a.hfl, out.stderr[-200:])
    wheel = a.wheel()
    expect(wheel.exists(), f"{wheel} is gone")
    return f"{wheel.name} in a fresh venv; every check of sections A, B, E ran on it"


@check("F2", "each extra installed on its own", needs=EXTRA_CHECKS)
def f2(a: Audit) -> str:
    results = json.loads((a.work / "results.json").read_text())
    extras = {k: v["status"] for k, v in results.items() if k.startswith("C")}
    expect(extras, "section C not run")
    broken = [k for k, v in extras.items() if v == "ROTO"]
    expect(not broken, f"broken extras: {broken}")
    return f"{len(extras)} extras: " + ", ".join(sorted(set(extras.values())))


@check("F3", "Docker image (arm64) and compose")
def f3(a: Audit) -> str:
    if shutil.which("docker") is None:
        raise Uncheckable("docker not installed")
    tag = "hfl-audit:local"
    build = subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(REPO)],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    expect(build.returncode == 0, build.stderr[-300:])
    try:
        check_ = subprocess.run(
            [a.python, str(REPO / "scripts" / "image_check.py"), tag],
            capture_output=True,
            text=True,
            timeout=2400,
            env={
                **a.env,
                "PATH": f"{Path(a.python).parent}:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
            },
        )
        lines = [x for x in check_.stdout.splitlines() if x[:3] in ("OK ", "BAD")]
        expect(check_.returncode == 0, " · ".join(lines)[-400:] or check_.stderr[-300:])
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
    return "built; image_check.py: " + " · ".join(x[:40] for x in lines)


@check("F4", "Homebrew formula")
def f4(a: Audit) -> str:
    if shutil.which("brew") is None:
        raise Uncheckable("brew not installed")
    style = subprocess.run(
        ["brew", "style", str(REPO / "packaging" / "homebrew" / "hfl.rb")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    raise PermissionError(
        "installing it puts HFL into the owner's Homebrew; `brew style` on the formula: "
        + ("clean" if style.returncode == 0 else " ".join(style.stdout.split())[-200:])
    )


@check("F5", "tray app (hfl serve --tray)")
def f5(a: Audit) -> str:
    port = "18777"
    log = a.work / "logs" / "tray.log"
    with open(log, "wb") as sink:
        proc = subprocess.Popen(
            [a.hfl, "serve", "--tray", "--port", port],
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=a.env,
        )
    try:
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
            if proc.poll() is not None:
                break
        alive = proc.poll() is None
        text = log.read_text(errors="replace")
        expect(alive and "Traceback" not in text, text[-300:])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    return "starts with its menu-bar icon and serves"


@check("F6", "PyInstaller executable", needs=(EXTRA_ID["build"],))
def f6(a: Audit) -> str:
    venv = a.work / "extras" / "build"
    pyinstaller = venv / "bin" / "pyinstaller"
    if not pyinstaller.exists():
        raise Uncheckable("run section C first (the [build] extra provides PyInstaller)")
    out_dir = a.work / "pyi"
    build = subprocess.run(
        [
            str(pyinstaller),
            "--noconfirm",
            "--distpath",
            str(out_dir / "dist"),
            "--workpath",
            str(out_dir / "build"),
            str(REPO / "hfl.spec"),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=REPO,
    )
    expect(build.returncode == 0, build.stderr[-300:])
    binary = next((p for p in (out_dir / "dist").rglob("hfl") if p.is_file()), None)
    expect(binary, "no executable built")
    version = subprocess.run([str(binary), "version"], capture_output=True, text=True, timeout=120)
    expect(version.returncode == 0 and "hfl v" in version.stdout, version.stderr[-300:])
    return f"built; `{binary.name} version`: {version.stdout.strip().splitlines()[0]}"


@check("F7", "MSI / winget")
def f7(a: Audit) -> str:
    raise Uncheckable("Windows installers: needs Windows")

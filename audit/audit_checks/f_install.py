# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section F: the ways HFL is installed and run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
from local_audit import Audit, Parts, Uncheckable, check, expect

from audit_checks.c_extras import EXTRA_ID

REPO = Path(__file__).resolve().parents[2]
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


@check("F3", "Docker images (llama and all, this machine's arch)")
def f3(a: Audit) -> str:
    """Both images the Docker workflow publishes, built and checked with the
    workflow's own ``image_check.py``. Only ``llama`` was built here once, and
    the ``all`` image's llama.cpp failed to load every model (dill, which
    vLLM brings, broke the load's stderr silencing): CI caught it at release."""
    if shutil.which("docker") is None:
        raise Uncheckable("docker not installed")
    part = Parts()
    for extras in ("llama", "all"):
        part(f"{extras} image", lambda extras=extras: _image(a, extras))
    return part.verdict()


def _image(a: Audit, extras: str) -> None:
    tag = f"hfl-audit-{extras}:local"
    build = subprocess.run(
        ["docker", "build", "-q", "--build-arg", f"HFL_EXTRAS={extras}", "-t", tag, str(REPO)],
        capture_output=True,
        text=True,
        timeout=5400,
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


@check("F6", "PyInstaller executable runs a model", needs=(EXTRA_ID["llama"],))
def f6(a: Audit) -> str:
    """Built from hfl.spec in the ``[llama]`` venv, as the release workflows
    do, then run for real: ``platform_check.py --hfl <executable>`` pulls,
    serves and answers with llama.cpp in process, without llama-server on
    PATH. ``version`` alone passed for the 0.22.0 executables and DMG, which
    could not run any model (llama.cpp's libraries were not bundled)."""
    venv = a.work / "extras" / "llama"
    python = venv / "bin" / "python"
    if not python.exists():
        raise Uncheckable("run section C first (the [llama] extra's venv)")
    added = subprocess.run(
        ["uv", "pip", "install", "-q", "--python", str(python), "pyinstaller"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    expect(added.returncode == 0, added.stderr[-300:])
    out_dir = a.work / "pyi"
    build = subprocess.run(
        [
            str(venv / "bin" / "pyinstaller"),
            "--noconfirm",
            "--clean",
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
    # PATH without llama-server: the executable must run the model itself.
    path = os.pathsep.join(
        d
        for d in os.environ.get("PATH", "").split(os.pathsep)
        if d and not (Path(d) / "llama-server").exists()
    )
    checked = subprocess.run(
        [
            a.python,
            str(REPO / "scripts" / "platform_check.py"),
            "--hfl",
            str(binary),
            "--expect-backend",
            "llama.cpp",
        ],
        capture_output=True,
        text=True,
        timeout=2400,
        env={**a.env, "PATH": path},
    )
    lines = [x for x in checked.stdout.splitlines() if x[:3] in ("OK ", "BAD")]
    expect(checked.returncode == 0, " · ".join(x for x in lines if x.startswith("BAD"))[-400:]
           or checked.stderr[-300:])  # fmt: skip
    return f"{binary.stat().st_size // 2**20} MB; platform_check.py: {len(lines)} checks passed"


@check("F7", "MSI / winget")
def f7(a: Audit) -> str:
    raise Uncheckable("Windows installers: needs Windows")

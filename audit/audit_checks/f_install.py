# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section F: the ways HFL is installed and run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from local_audit import WINDOWS, Audit, Parts, Uncheckable, check, expect, venv_exe

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
    pruned = _prune_build_cache()
    part = Parts()
    for extras in ("llama", "all"):
        part(f"{extras} image", lambda extras=extras: _image(a, extras))
    return f"{part.verdict()}; {pruned}"


def _prune_build_cache() -> str:
    """Free Docker's build cache first: the ``all`` image needs ~19 GB of it,
    and with the Docker disk full F3 failed with "No space left on device"
    (twice, 2026-09-27 and 2026-09-29). Only the build cache — never images,
    containers or volumes. ``AUDIT_KEEP_BUILD_CACHE=1`` keeps it."""
    import os

    if os.environ.get("AUDIT_KEEP_BUILD_CACHE", "").strip() == "1":
        return "build cache kept (AUDIT_KEEP_BUILD_CACHE=1)"
    done = subprocess.run(
        ["docker", "builder", "prune", "-f"], capture_output=True, text=True, timeout=1800
    )
    if done.returncode != 0:
        return f"build cache NOT pruned: {(done.stderr or done.stdout).strip()[-120:]}"
    total = next(
        (line.split(":", 1)[1].strip() for line in done.stdout.splitlines()
         if line.startswith("Total:")),
        "0B",
    )  # fmt: skip
    return f"build cache pruned first ({total} freed)"


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
                "PATH": os.pathsep.join(
                    [str(Path(a.python).parent)]
                    + (
                        [os.environ.get("PATH", "")]
                        if WINDOWS
                        else ["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/bin"]
                    )
                ),
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
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        # Without a desktop session the icon cannot exist: what can be
        # checked is that HFL says so plainly, not with a traceback.
        done = a.cli("serve", "--tray", "--port", "18777", timeout=60)
        out = done.stdout + done.stderr
        expect(done.returncode == 1 and "Traceback" not in out, out[-300:])
        expect("desktop session" in out, out[-300:])
        raise Uncheckable(f"no desktop session here; it says so: {out.strip()[:120]}")
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
    do — llama-server fetched by ``scripts/fetch_llama_server.py`` and
    bundled — then run for real, without llama-server on PATH:
    ``platform_check.py --hfl <executable>`` pulls, serves and answers twice,
    by the bundled llama-server (the default) and by llama.cpp in process
    (``HFL_LLM_LIBRARY=llama-cpp``). ``version`` alone passed for the 0.22.0
    executables and DMG, which could not run any model (llama.cpp's
    libraries were not bundled)."""
    venv = a.work / "extras" / "llama"
    python = venv_exe(venv, "python")
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
    # The audited install's own Python runs the fetch: the [llama] venv may
    # hold an older wheel.
    fetched = subprocess.run(
        [a.python, str(REPO / "scripts" / "fetch_llama_server.py"), str(out_dir / "llama.cpp")],
        capture_output=True, text=True, timeout=1800,
    )  # fmt: skip
    expect(fetched.returncode == 0, (fetched.stdout + fetched.stderr)[-300:])
    build = subprocess.run(
        [
            str(venv_exe(venv, "pyinstaller")),
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
        env={**os.environ, "HFL_PYI_LLAMA_CPP": str(out_dir / "llama.cpp")},
    )
    expect(build.returncode == 0, build.stderr[-300:])
    name = "hfl.exe" if WINDOWS else "hfl"
    binary = next((p for p in (out_dir / "dist").rglob(name) if p.is_file()), None)
    expect(binary, "no executable built")
    # PATH without llama-server: the executable must run the model itself.
    path = os.pathsep.join(
        d
        for d in os.environ.get("PATH", "").split(os.pathsep)
        if d and not any((Path(d) / n).exists() for n in ("llama-server", "llama-server.exe"))
    )
    passed = []
    for backend, env in (("llama-server", {}), ("llama.cpp", {"HFL_LLM_LIBRARY": "llama-cpp"})):
        checked = subprocess.run(
            [
                a.python, str(REPO / "scripts" / "platform_check.py"), "--hfl", str(binary),
                "--expect-backend", backend,
            ],
            capture_output=True, text=True, timeout=2400, env={**a.env, "PATH": path, **env},
        )  # fmt: skip
        lines = [x for x in checked.stdout.splitlines() if x[:3] in ("OK ", "BAD")]
        expect(
            checked.returncode == 0,
            f"{backend}: " + (" · ".join(x for x in lines if x.startswith("BAD"))[-400:]
                              or checked.stderr[-300:]),
        )  # fmt: skip
        passed.append(f"{backend} {len(lines)}")
    size = binary.stat().st_size // 2**20
    return f"{size} MB, llama-server bundled; platform_check.py passed: {', '.join(passed)}"


def _wix_bin() -> Path | None:
    """WiX v3's bin folder: ``AUDIT_WIX_BIN``, or where its installer puts it
    (the release workflow's). The official binaries zip needs no install."""
    found = [os.environ.get("AUDIT_WIX_BIN", "")]
    found += [str(d / "bin") for d in Path(r"C:\Program Files (x86)").glob("WiX Toolset*")]
    return next((Path(d) for d in found if d and (Path(d) / "candle.exe").exists()), None)


# What the MSI registers, read from its own tables (nothing is installed).
_MSI_TABLES = r"""
$wi = New-Object -ComObject WindowsInstaller.Installer
$db = $wi.GetType().InvokeMember('OpenDatabase', 'InvokeMethod', $null, $wi, @($args[0], 0))
foreach ($q in @("SELECT Name, Value FROM Environment",
                 "SELECT Property, Value FROM Property WHERE Property='ALLUSERS'")) {
  $v = $db.GetType().InvokeMember('OpenView', 'InvokeMethod', $null, $db, @($q))
  $v.GetType().InvokeMember('Execute', 'InvokeMethod', $null, $v, $null)
  while ($r = $v.GetType().InvokeMember('Fetch', 'InvokeMethod', $null, $v, $null)) {
    $r.GetType().InvokeMember('StringData', 'GetProperty', $null, $r, 1) + '=' +
    $r.GetType().InvokeMember('StringData', 'GetProperty', $null, $r, 2)
  }
}
"""


@check("F7", "MSI / winget", needs=("F6",))
def f7(a: Audit) -> str:
    """The MSI built as windows-msi.yml builds it, from F6's executable, then
    opened without installing anything: ``msiexec /a`` extracts it, the
    extracted hfl.exe must be F6's and run, and its tables must put HFL on
    the system PATH for all users. The winget manifest is validated too."""
    if not WINDOWS:
        raise Uncheckable("Windows installers: needs Windows")
    wix = _wix_bin()
    if wix is None:
        raise Uncheckable("WiX v3 not found (AUDIT_WIX_BIN: a folder with candle.exe)")
    import hashlib
    import tempfile

    exe = a.work / "pyi" / "dist" / "hfl.exe"
    expect(exe.exists(), "F6 built no hfl.exe")
    build = Path(tempfile.mkdtemp(prefix="msi-", dir=a.work))
    (build / "dist").mkdir()
    (build / "packaging" / "windows").mkdir(parents=True)
    shutil.copyfile(exe, build / "dist" / "hfl.exe")
    rtf = Path("packaging") / "windows" / "LICENSE.rtf"
    shutil.copyfile(REPO / rtf, build / rtf)
    import re

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "(.+)"', pyproject, re.M).group(1)  # type: ignore[union-attr]
    wxs = (REPO / "packaging" / "windows" / "hfl.wxs").read_text(encoding="utf-8")
    (build / "dist" / "hfl.wxs").write_text(wxs.replace("@HFL_VERSION@", version), encoding="utf-8")
    msi = build / "dist" / f"hfl-{version}.msi"
    obj, src = r"dist\hfl.wixobj", r"dist\hfl.wxs"
    for step in (
        [str(wix / "candle.exe"), "-nologo", "-arch", "x64", "-out", obj, src],
        [str(wix / "light.exe"), "-nologo", "-ext", "WixUIExtension", "-out", str(msi), obj],
    ):
        done = subprocess.run(step, capture_output=True, text=True, cwd=build, timeout=900)
        # light runs the ICE validation: a failing one fails the build.
        expect(done.returncode == 0, f"{Path(step[0]).name}: {(done.stdout + done.stderr)[-300:]}")
    extract = build / "extract"
    admin = subprocess.run(
        ["msiexec", "/a", str(msi), "/qn", f"TARGETDIR={extract}"], capture_output=True, timeout=600
    )
    expect(admin.returncode == 0, f"msiexec /a exit {admin.returncode}")
    inside = extract / "HFL" / "hfl.exe"

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    expect(inside.exists() and digest(inside) == digest(exe), "the MSI's hfl.exe is not F6's")
    runs = subprocess.run(
        [str(inside), "version"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=a.env,  # PYTHONUTF8: the console's own code page cannot print "·"
        timeout=120,
    )
    expect(runs.returncode == 0 and version in runs.stdout, runs.stdout[-200:] + runs.stderr[-200:])
    query = build / "tables.ps1"  # -File passes arguments as $args; -Command does not
    query.write_text(_MSI_TABLES, encoding="utf-8")
    tables = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(query), str(msi)],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.split()
    expect(any("PATH=" in t and "[INSTALLFOLDER]" in t for t in tables), f"no PATH entry: {tables}")
    expect("ALLUSERS=1" in tables, f"not for all users: {tables}")
    manifest = REPO / "packaging" / "winget" / "manifests" / "g" / "ggalancs" / "HFL" / version
    winget = "winget manifest not checked (no winget here)"
    if shutil.which("winget") and manifest.is_dir():
        valid = subprocess.run(
            ["winget", "validate", "--manifest", str(manifest)], capture_output=True, timeout=300
        )
        expect(valid.returncode == 0, f"winget validate exit {valid.returncode}")
        winget = "winget manifest valid"
    size = msi.stat().st_size // 2**20
    return f"{msi.name} {size} MB: ICE passed, extracts F6's hfl.exe, runs, on PATH; {winget}"

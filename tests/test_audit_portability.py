# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The local audit runs on Windows too: programs in a venv live in
``Scripts\\<name>.exe`` there, not ``bin/<name>``, and the watchdog kills a
process tree without Unix process groups."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[1] / "audit"


@pytest.fixture
def local_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(AUDIT))
    import local_audit

    return local_audit


def test_a_venv_program_on_each_platform(local_audit, monkeypatch):
    venv = Path("/w/venv")
    monkeypatch.setattr(local_audit, "WINDOWS", False)
    assert local_audit.venv_exe(venv, "hfl") == venv / "bin" / "hfl"
    monkeypatch.setattr(local_audit, "WINDOWS", True)
    assert local_audit.venv_exe(venv, "python") == venv / "Scripts" / "python.exe"


def test_no_check_builds_a_unix_venv_path_by_hand():
    """Every program in a venv goes through venv_exe."""
    offenders = []
    for path in [*AUDIT.glob("*.py"), *(AUDIT / "audit_checks").glob("*.py")]:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            hand_built = re.search(r'/ "bin" /', line) and "Scripts" not in line
            if hand_built:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, offenders


@pytest.mark.skipif(sys.platform == "win32", reason="the Unix branch; taskkill is Windows'")
def test_the_watchdog_kills_the_whole_tree(tmp_path):
    """A command that starts a child and hangs: both die at the limit."""
    marker = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time; "
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        f"open({str(marker)!r}, 'w').write(str(c.pid)); time.sleep(120)"
    )
    log = tmp_path / "log.txt"
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, str(AUDIT / "watchdog.py"), "2", str(log), "--",
         sys.executable, "-c", script],
        timeout=60,
    )  # fmt: skip
    assert done.returncode == 124 and time.monotonic() - started < 40
    child = int(marker.read_text())
    time.sleep(0.5)
    alive = subprocess.run(["ps", "-p", str(child)], capture_output=True).returncode == 0
    assert not alive, "the child outlived the watchdog"

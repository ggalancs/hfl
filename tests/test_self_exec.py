# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Starting a part of HFL in a new process works from an install and from a
PyInstaller executable, where ``sys.executable`` is ``hfl`` itself and
``-m`` / ``-c`` were "No such option" (llama-server never started)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hfl.utils import self_exec


def test_installed_uses_the_interpreter(monkeypatch) -> None:
    monkeypatch.setattr(self_exec, "is_frozen", lambda: False)
    guard = self_exec.child_guard_argv(42, ["llama-server", "-m", "x.gguf"])
    assert guard[1:5] == ["-m", "hfl.engine._child_guard", "42", "--"]
    assert self_exec.hfl_argv("serve")[1] == "-c"


def test_an_executable_runs_itself_without_python_flags(monkeypatch) -> None:
    monkeypatch.setattr(self_exec, "is_frozen", lambda: True)
    guard = self_exec.child_guard_argv(42, ["llama-server", "-m", "x.gguf"])
    assert guard == [
        sys.executable,
        self_exec.CHILD_GUARD_FLAG,
        "42",
        "--",
        "llama-server",
        "-m",
        "x.gguf",
    ]
    assert self_exec.hfl_argv("serve", "--port", "1") == [sys.executable, "serve", "--port", "1"]


def test_the_cli_entry_point_runs_the_guard(monkeypatch) -> None:
    """``hfl --hfl-internal-child-guard <pid> -- cmd`` runs cmd under the
    guard (what an executable does) and returns its exit code."""
    import os

    from hfl.cli.main import cli_main

    command = [sys.executable, "-c", "import sys; sys.exit(7)"]
    monkeypatch.setattr(
        sys, "argv", ["hfl", self_exec.CHILD_GUARD_FLAG, str(os.getpid()), "--", *command]
    )
    with pytest.raises(SystemExit) as caught:
        cli_main()
    assert caught.value.code == 7


def test_the_guard_really_starts_its_command() -> None:
    """The interpreter form, run for real: the guard starts the command."""
    import os

    out = subprocess.run(
        self_exec.child_guard_argv(os.getpid(), [sys.executable, "-c", "print('started')"]),
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert out.returncode == 0 and "started" in out.stdout


def test_an_install_never_watches_its_parent(monkeypatch) -> None:
    """Outside an executable a parent may exit on purpose (nohup)."""
    monkeypatch.setattr(self_exec, "is_frozen", lambda: False)
    assert self_exec.watch_onefile_launcher() is False


def test_an_executable_not_started_by_its_launcher_does_not_watch(monkeypatch) -> None:
    monkeypatch.setattr(self_exec, "is_frozen", lambda: True)
    monkeypatch.setattr(self_exec, "_executable_of", lambda pid: "/bin/zsh")
    assert self_exec.watch_onefile_launcher() is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher only")
def test_a_onefile_program_stops_when_its_launcher_is_gone(monkeypatch) -> None:
    """The launcher killed with SIGKILL left ``hfl serve`` running forever;
    the program sends itself SIGTERM once its parent changes."""
    import os
    import signal
    import time

    monkeypatch.setattr(self_exec, "is_frozen", lambda: True)
    monkeypatch.setattr(self_exec, "_executable_of", lambda pid: os.path.realpath(sys.executable))
    parents = iter([1000, 1000, 1])  # the launcher, then re-parented to init
    monkeypatch.setattr(os, "getppid", lambda: next(parents, 1))
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert self_exec.watch_onefile_launcher() is True
    deadline = time.monotonic() + 5
    while not sent and time.monotonic() < deadline:
        pass
    assert sent == [(os.getpid(), signal.SIGTERM)]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""llama-server's guard and stop on Windows, found on a real Windows 10.

``os.kill(pid, 0)`` is no liveness check there: signal 0 is
``CTRL_C_EVENT``, so the guard's "is HFL alive?" sent Ctrl+C to the console
group and llama-server shut down a second after every load (``hfl serve``
could not start a GGUF model). And SIGTERM is TerminateProcess, which ended
the guard alone and left llama-server running with the model in memory."""

from __future__ import annotations

import os
import subprocess

from hfl.engine import _child_guard, llama_server


def test_the_guard_never_signals_the_parent_on_windows(monkeypatch) -> None:
    def no_kill(*args: object) -> None:
        raise AssertionError("os.kill(pid, 0) sends Ctrl+C on Windows")

    monkeypatch.setattr(os, "kill", no_kill)
    monkeypatch.setattr(_child_guard, "_windows_parent_gone", lambda handle: handle == 0)
    monkeypatch.setattr(os, "name", "nt")
    alive, gone = _child_guard._parent_gone(42, 7), _child_guard._parent_gone(42, 0)
    monkeypatch.undo()
    assert alive is False and gone is True


def test_stop_ends_the_guard_and_its_llama_server_on_windows(monkeypatch) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: ran.append(argv))

    class _Guard:
        pid = 4242
        waited = False

        def poll(self) -> None:
            return None

        def send_signal(self, sig: int) -> None:
            raise AssertionError("SIGTERM ends the guard alone on Windows")

        def wait(self, timeout: float) -> int:
            self.waited = True
            return 1

    guard = _Guard()
    monkeypatch.setattr(os, "name", "nt")
    llama_server.stop_server(guard)  # type: ignore[arg-type]
    monkeypatch.undo()
    assert ran == [["taskkill", "/T", "/F", "/PID", "4242"]] and guard.waited

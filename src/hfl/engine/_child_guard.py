# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Run a command for as long as the process that asked for it is alive.

    python -m hfl.engine._child_guard <parent-pid> -- <command> [args...]

HFL starts ``llama-server`` through this guard. A clean shutdown stops the
child through the guard (SIGTERM is forwarded); but if HFL dies without one
— SIGKILL, a crash — nothing would stop the child, and it would keep the
model in memory for good. macOS has no parent-death signal, so the guard
watches instead: once the parent is gone it terminates the child (and kills
it if it does not stop within ten seconds), then exits.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys


def _windows_watch(parent: int) -> int:
    """A handle to the parent that signals once it exits (0: already gone).

    Windows has no ``kill(pid, 0)``: signal 0 is ``CTRL_C_EVENT``, so asking
    whether HFL lived sent Ctrl+C to the console group, and llama-server
    shut down a second after every load. A handle also cannot confuse the
    parent with a later process given the same PID."""
    import ctypes

    synchronize = 0x00100000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    return int(kernel32.OpenProcess(synchronize, False, parent) or 0)


def _windows_parent_gone(handle: int) -> bool:
    import ctypes

    if not handle:
        return True
    wait_object_0 = 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    return bool(kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == wait_object_0)


def _parent_gone(parent: int, handle: int = 0) -> bool:
    if os.name == "nt":
        return _windows_parent_gone(handle)
    if os.getppid() != parent:  # re-parented: the parent has exited
        return True
    try:
        os.kill(parent, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _stop(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        print("usage: _child_guard <parent-pid> -- <command> [args...]", file=sys.stderr)
        return 2
    parent = int(argv[1])
    handle = _windows_watch(parent) if os.name == "nt" else 0
    child = subprocess.Popen(argv[3:])

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    while True:
        try:
            child.wait(timeout=1)  # returns as soon as the child exits
            break
        except subprocess.TimeoutExpired:
            pass
        if _parent_gone(parent, handle):
            _stop(child)
            break
    return child.returncode if child.returncode is not None and child.returncode >= 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

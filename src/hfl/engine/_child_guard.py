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


def _parent_gone(parent: int) -> bool:
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
        if _parent_gone(parent):
            _stop(child)
            break
    return child.returncode if child.returncode is not None and child.returncode >= 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

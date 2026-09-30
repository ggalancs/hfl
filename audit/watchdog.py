#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A hard time limit for a long run that does not depend on the run itself.

    python audit/watchdog.py SECONDS LOG -- COMMAND ...

Runs COMMAND in a process group of its own, its output unbuffered into LOG,
and kills the whole group (every server and child it started) once SECONDS
pass. macOS ships no ``timeout``; this knows only a PID and a clock, so a
hung check cannot keep it from firing.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys


def _kill_tree(proc: subprocess.Popen, sig: int) -> None:
    """Every process the command started, not only the command."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(proc.pid, sig)


def main() -> int:
    if "--" not in sys.argv or len(sys.argv) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    seconds, log = float(sys.argv[1]), sys.argv[2]
    command = sys.argv[sys.argv.index("--") + 1 :]
    with open(log, "ab", buffering=0) as out:
        if os.name == "nt":  # no process groups: a group of its own, killed as a tree
            group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            group = {"start_new_session": True}
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, **group)
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            _kill_tree(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill_tree(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
            out.write(f"\nWATCHDOG: killed after {seconds:.0f}s\n".encode())
            code = 124
        out.write(f"\nexit {code}\n".encode())
    return code


if __name__ == "__main__":
    sys.exit(main())

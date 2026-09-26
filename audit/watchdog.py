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


def main() -> int:
    if "--" not in sys.argv or len(sys.argv) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    seconds, log = float(sys.argv[1]), sys.argv[2]
    command = sys.argv[sys.argv.index("--") + 1 :]
    with open(log, "ab", buffering=0) as out:
        proc = subprocess.Popen(
            command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            out.write(f"\nWATCHDOG: killed after {seconds:.0f}s\n".encode())
            code = 124
        out.write(f"\nexit {code}\n".encode())
    return code


if __name__ == "__main__":
    sys.exit(main())

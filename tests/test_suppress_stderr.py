# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``_suppress_stderr`` silences the C library during a llama.cpp load, not
HFL's own logging. It used to redirect fd 2 wholesale, and every log line
written meanwhile — the load's "KV cache quantised", its "unsupported, falling
back to f16" warning, other requests' lines — went to /dev/null with it.

Run in a subprocess: the fd-level redirection is the thing under test, and
pytest's own capture would sit between the handler and fd 2."""

from __future__ import annotations

import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import logging, os, sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    from hfl.engine.llama_cpp import _suppress_stderr
    log = logging.getLogger("hfl.engine.llama_cpp")
    handler = logging.getLogger().handlers[0]
    before = handler.stream
    with _suppress_stderr():
        os.write(2, b"C-LIBRARY-NOISE\\n")
        log.warning("HFL-WARNING-INSIDE")
    log.info("HFL-INFO-AFTER")
    os.write(2, b"C-AFTER\\n")
    print("SAME-STREAM" if handler.stream is before else "STREAM-CHANGED")
    """
)


def _run() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120
    )


def test_hfl_logging_survives_while_the_c_library_is_silenced() -> None:
    out = _run()
    assert out.returncode == 0, out.stderr
    assert "HFL-WARNING-INSIDE" in out.stderr
    assert "C-LIBRARY-NOISE" not in out.stderr


def test_everything_is_restored_afterwards() -> None:
    out = _run()
    assert "HFL-INFO-AFTER" in out.stderr and "C-AFTER" in out.stderr
    assert "SAME-STREAM" in out.stdout

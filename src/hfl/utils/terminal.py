# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Whether someone can answer a question on standard input."""

from __future__ import annotations

import os
import sys


def _windows_console(fd: int) -> bool:
    """True only for a real console. Windows reports ``NUL`` as a character
    device, so ``isatty()`` is True for ``< NUL``, a service or a scheduled
    task: nobody is there, and a prompt reads end-of-file."""
    import ctypes
    import msvcrt

    try:
        handle = msvcrt.get_osfhandle(fd)  # type: ignore[attr-defined]
    except OSError:
        return False
    mode = ctypes.c_uint32()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    return bool(kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)))


def stdin_is_terminal() -> bool:
    """True when a person can type an answer on standard input."""
    stream = sys.stdin
    if stream is None:
        return False
    try:
        if not stream.isatty():
            return False
        return _windows_console(stream.fileno()) if os.name == "nt" else True
    except (AttributeError, OSError, TypeError, ValueError):
        return False

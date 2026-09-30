# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Starting a part of HFL in a new process, installed or as an executable.

From a Python install ``sys.executable`` is the interpreter, so ``-m`` and
``-c`` run HFL's code. In a PyInstaller executable ``sys.executable`` is the
``hfl`` executable itself, which knows neither flag: llama-server's guard
started ``hfl -m hfl.engine._child_guard …``, got "No such option: -m", and
no executable could ever serve a model through llama-server; ``hfl launch``
failed the same way. These build the command that works in both.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

# ``hfl <this> <parent-pid> -- <command>…`` runs the child guard: the one
# entry point an executable has for it. Not a command users see.
CHILD_GUARD_FLAG = "--hfl-internal-child-guard"


def is_frozen() -> bool:
    """True inside a PyInstaller executable."""
    return bool(getattr(sys, "frozen", False))


def child_guard_argv(parent_pid: int, command: list[str]) -> list[str]:
    """Run ``command`` for as long as ``parent_pid`` lives (see
    ``hfl.engine._child_guard``)."""
    if is_frozen():
        return [sys.executable, CHILD_GUARD_FLAG, str(parent_pid), "--", *command]
    return [sys.executable, "-m", "hfl.engine._child_guard", str(parent_pid), "--", *command]


def hfl_argv(*args: str) -> list[str]:
    """``hfl args…`` with this same HFL."""
    if is_frozen():
        return [sys.executable, *args]
    return [sys.executable, "-c", "from hfl.cli.main import cli_main; cli_main()", *args]


def _kernel32() -> Any:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    return kernel32


def _windows_image_of(pid: int) -> str | None:
    import ctypes
    import os

    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        size = ctypes.c_uint32(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        ok = kernel32.QueryFullProcessImageNameW(
            ctypes.c_void_p(handle), 0, buffer, ctypes.byref(size)
        )
        return os.path.realpath(buffer.value) if ok else None
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _windows_waiter(pid: int) -> Callable[[], None] | None:
    """A call that returns once process ``pid`` has exited (None: gone, or
    not ours to watch). The handle is opened now, so a later process given
    the same PID is never mistaken for it."""
    import ctypes

    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return None

    def wait() -> None:
        kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF)  # INFINITE

    return wait


def _executable_of(pid: int) -> str | None:
    """The executable a process runs, or None when it cannot be read."""
    import os
    import subprocess

    if os.name == "nt":
        return _windows_image_of(pid)
    try:
        if sys.platform.startswith("linux"):
            return os.path.realpath(os.readlink(f"/proc/{pid}/exe"))
        out = subprocess.run(
            ["ps", "-o", "comm=", "-p", str(pid)], capture_output=True, text=True, timeout=5
        )
        path = out.stdout.strip()
        return os.path.realpath(path) if path else None
    except (OSError, subprocess.SubprocessError):
        return None


def watch_onefile_launcher() -> bool:
    """In a one-file executable, stop HFL when its launcher is gone.

    A one-file PyInstaller executable is two processes: a launcher, which
    unpacks the program and forwards signals, and the program. SIGKILL
    cannot be forwarded, so killing the launcher (a process manager, the
    tray, ``kill -9``) left ``hfl serve`` running — port bound, models and
    llama-server in memory — with nothing left to stop it. The program
    watches the launcher and, once it is gone, stops as on SIGTERM (models
    unloaded, llama-server stopped).

    Only when the parent at start is this same executable: in any other
    install a parent may exit on purpose (``nohup hfl serve &``). True when
    it watches.

    Windows too: stopping the launcher there (TerminateProcess, Task
    Manager) left ``hfl.exe serve`` serving on its own, holding the port
    and its own file. ``getppid()`` never changes there, so a handle to the
    launcher says when it is gone.
    """
    import os
    import signal
    import threading
    import time

    if not is_frozen() or os.name not in ("posix", "nt"):
        return False
    parent = os.getppid()
    mine = os.path.realpath(sys.executable)
    if parent <= 1 or _executable_of(parent) != mine:
        return False

    if os.name == "nt":
        waiter = _windows_waiter(parent)
        if waiter is None:
            return False
        wait_for_launcher = waiter

        def watch() -> None:
            wait_for_launcher()
            # To this process's own handler (uvicorn's: a clean shutdown);
            # os.kill would be TerminateProcess, with nothing unloaded.
            signal.raise_signal(signal.SIGTERM)

    else:

        def watch() -> None:
            while os.getppid() == parent:
                time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch, name="hfl-launcher-watch", daemon=True).start()
    return True

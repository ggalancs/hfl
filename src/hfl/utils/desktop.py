# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HFL opened from a desktop rather than typed in a terminal: HFL.app
double-clicked in Finder, or the Start-menu shortcut the MSI installs. It
serves under a tray icon and opens the chat page in the browser."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

# What HFL.app runs when Finder opens it (the Start-menu shortcut passes the
# same arguments itself).
APP_ARGS = ["serve", "--tray", "--open"]

# Where HFL.app writes its output: Finder gives an app no terminal, and a
# failure nobody can read is a failure nobody can report.
MAC_APP_LOG = Path.home() / "Library" / "Logs" / "HFL" / "hfl.log"
_LOG_LIMIT = 10 * 1024 * 1024


def opened_as_mac_app(argv: list[str]) -> bool:
    """Whether Finder opened HFL.app: the frozen executable inside an app
    bundle, with no arguments (macOS before 10.9 passed a ``-psn_…`` one)
    and no terminal on stdin. The same file run from a terminal with no
    arguments shows the help, as ``hfl`` does everywhere."""
    if sys.platform != "darwin" or not getattr(sys, "frozen", False):
        return False
    if ".app/Contents/MacOS/" not in sys.executable:
        return False
    if any(not arg.startswith("-psn_") for arg in argv[1:]):
        return False
    try:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    except (OSError, ValueError):
        interactive = False
    return not interactive


def log_to_file(path: Path = MAC_APP_LOG) -> Path:
    """Send this process's stdout and stderr (and its children's) to
    ``path``, started afresh once it passes 10 MB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        if path.stat().st_size > _LOG_LIMIT:
            path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    return path


def _browser_address(host: str) -> str:
    """The address a browser on this machine uses for a server bound to
    ``host``: loopback of the same family for a wildcard bind (``::`` is
    IPv6-only, so not 127.0.0.1)."""
    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def chat_url(host: str, port: int) -> str:
    """The chat page as a browser on this machine reaches it."""
    address = _browser_address(host)
    if ":" in address and not address.startswith("["):
        address = f"[{address}]"
    return f"http://{address}:{port}/ui"


def _listening_here(port: int, address: str) -> bool:
    """Whether this process listens on ``port`` at ``address``, the one the
    browser will use: the server this HFL started, not another program
    (another user's, say) holding that address. A wildcard listener covers
    its own family; for a name (``localhost``) the port decides."""
    import ipaddress

    import psutil

    process = psutil.Process(os.getpid())
    # ``net_connections`` since psutil 6; 5.9 (still allowed) has ``connections``.
    listing = getattr(process, "net_connections", None) or process.connections
    try:
        connections = listing(kind="inet")
    except (psutil.Error, OSError):
        return False
    version: int
    try:
        version = ipaddress.ip_address(address.strip("[]")).version
    except ValueError:
        version = 0  # a name
    wildcard = {4: "0.0.0.0", 6: "::"}.get(version)
    for c in connections:
        if c.status != psutil.CONN_LISTEN or c.laddr.port != port:
            continue
        ip = str(c.laddr.ip).split("%", 1)[0]
        if not version or ip in (address.strip("[]"), wildcard):
            return True
    return False


def open_chat_when_ready(host: str, port: int, timeout: float = 120.0) -> threading.Thread:
    """Open the chat page once this process serves on ``port`` and answers
    ``/healthz``, from a background thread. Anything else answering there
    does not count: it would have the owner's browser open its page. Gives
    up after ``timeout`` seconds: the tray icon shows why the server did not
    start."""
    url = chat_url(host, port)
    address = _browser_address(host)
    health = url[: -len("/ui")] + "/healthz"

    def _wait() -> None:
        import httpx

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                ours = _listening_here(port, address)
                if ours and httpx.get(health, timeout=2).status_code == 200:
                    logger.info("Opening the chat page: %s", url)
                    webbrowser.open(url)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        logger.warning("The server did not answer in %.0f s; the chat page was not opened", timeout)

    thread = threading.Thread(target=_wait, name="hfl-open-chat", daemon=True)
    thread.start()
    return thread


def hide_own_console() -> bool:
    """Hide the console window when nobody else uses it (Windows). The
    Start-menu shortcut starts hfl.exe in a console of its own, a black
    window that would stay open beside the tray icon; the terminal a user
    typed ``hfl serve --tray`` in is shared with its shell and stays. A
    one-file build runs as two processes, its bootloader (the parent, the
    same executable) and this one: the bootloader is not somebody else."""
    if sys.platform != "win32":
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    pids = (ctypes.c_uint32 * 16)()
    count = kernel32.GetConsoleProcessList(pids, 16)
    if not 0 < count <= 16:
        return False
    others = {pids[i] for i in range(count)} - {os.getpid()}
    if any(not _is_own_bootloader(pid) for pid in others):
        return False
    window = kernel32.GetConsoleWindow()
    if not window:
        return False
    ctypes.windll.user32.ShowWindow(window, 0)  # SW_HIDE
    return True


def _is_own_bootloader(pid: int) -> bool:
    if pid != os.getppid():
        return False
    try:
        import psutil

        exe: str = psutil.Process(pid).exe()
    except Exception:
        return False
    return os.path.normcase(exe) == os.path.normcase(sys.executable)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Whether a person can answer on standard input, Windows included: there
``NUL`` is a character device and ``isatty()`` says True, so ``hfl serve
--host 0.0.0.0`` started unattended asked "Continue?" instead of refusing
an exposure nobody had allowed (found on a real Windows 10)."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

from hfl.utils import terminal


def test_not_a_terminal(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", MagicMock(isatty=lambda: False))
    assert terminal.stdin_is_terminal() is False


def test_no_stdin_at_all(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", None)
    assert terminal.stdin_is_terminal() is False


def test_a_posix_terminal(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", MagicMock(isatty=lambda: True, fileno=lambda: 0))
    monkeypatch.setattr(os, "name", "posix")
    assert terminal.stdin_is_terminal() is True


def test_on_windows_a_tty_must_also_be_a_console(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", MagicMock(isatty=lambda: True, fileno=lambda: 0))
    consoles = {"nul": False, "console": True}
    for kind, is_console in consoles.items():
        monkeypatch.setattr(terminal, "_windows_console", lambda fd, c=is_console: c)
        monkeypatch.setattr(os, "name", "nt")
        seen = terminal.stdin_is_terminal()
        monkeypatch.setattr(os, "name", "posix")
        assert seen is is_console, kind

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl serve --tray`` on a Linux with no desktop session (a server, SSH, a
container) was a DisplayNameError traceback from pystray's import; it says
what is missing instead (found by the local audit's F5 in its Linux image)."""

from __future__ import annotations

import sys

from typer.testing import CliRunner


def test_tray_without_a_display_says_so(monkeypatch) -> None:
    from hfl.cli.main import app

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    result = CliRunner().invoke(app, ["serve", "--tray", "--port", "18999"])
    assert result.exit_code == 1
    assert "desktop session" in result.output
    assert "Traceback" not in result.output

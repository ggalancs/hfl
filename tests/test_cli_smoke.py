# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Smoke coverage for *every* registered CLI command.

Ad-hoc per-command tests miss commands as the CLI grows. These
parametrised tests iterate over the whole Typer command registry so a
broken command definition (unresolved i18n key, a ``typer.OptionInfo``
default that leaks, an import error, a missing help string) fails a
specific ``test_command_help_smoke[<cmd>]`` case in CI instead of only
surfacing at runtime for a user.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from hfl.cli.main import app

runner = CliRunner()


def _all_command_names() -> list[str]:
    names: list[str] = []
    for c in app.registered_commands:
        name = c.name or (c.callback.__name__.replace("_", "-") if c.callback else None)
        if name:
            names.append(name)
    return sorted(set(names))


ALL_COMMANDS = _all_command_names()


def test_registry_is_non_trivial() -> None:
    # Guards the parametrisation itself: if the introspection breaks and
    # returns [], the parametrised smoke would silently run zero cases.
    assert len(ALL_COMMANDS) >= 25, ALL_COMMANDS


@pytest.mark.parametrize("cmd", ALL_COMMANDS)
def test_command_help_smoke(cmd: str) -> None:
    """``hfl <cmd> --help`` must render with exit code 0.

    Exercises the full command definition (Typer resolves every option's
    default and help text) without running any side effects.
    """
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0, (
        f"`hfl {cmd} --help` failed (exit {result.exit_code}):\n{result.output}"
    )


def test_top_level_help_and_version() -> None:
    for args in (["--help"], ["--version"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"`hfl {' '.join(args)}` failed:\n{result.output}"

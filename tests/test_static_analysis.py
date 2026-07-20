# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Static-analysis gate: keep ``src/hfl`` clean under mypy and ruff.

This is the regression net for the type/lint cleanup. The project carries
no automatic CI, so these tests are how a type or lint regression gets
caught locally — they run the exact same checks a maintainer would:

* ``mypy src/hfl``           — no type errors (config in ``pyproject.toml``:
  ``warn_return_any``, ``warn_unused_ignores``, the ``disallow_untyped_defs``
  override for ``hfl.api.*`` / ``hfl.cli.commands.*``).
* ``ruff check src/hfl``     — no lint errors. This is what catches the
  ``cast(np.ndarray, …)``-references-a-function-local class of bug
  (pyflakes ``F823``) that mypy alone does **not** see.
* ``ruff format --check``    — formatting is canonical.

Each test shells out with the interpreter running the suite so it uses the
same pinned tool versions, and skips cleanly when the tool isn't installed
(e.g. a runtime-only environment without the dev extras).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TARGET = "src/hfl"


def _run(*tool_args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m <tool_args>`` from the repo root, capturing output."""
    return subprocess.run(
        [sys.executable, "-m", *tool_args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )


@pytest.mark.integration
@pytest.mark.slow
def test_mypy_clean_on_src() -> None:
    """``mypy src/hfl`` must report zero type errors.

    Guards the whole pre-existing-error cleanup: any newly-introduced
    ``no-any-return`` / ``union-attr`` / unused ``type: ignore`` etc. fails here.
    """
    pytest.importorskip("mypy", reason="mypy not installed (dev extra)")
    result = _run("mypy", _TARGET)
    assert result.returncode == 0, (
        "mypy found type errors in src/hfl — the type gate regressed:\n\n"
        + result.stdout
        + result.stderr
    )


@pytest.mark.integration
def test_ruff_check_clean_on_src() -> None:
    """``ruff check src/hfl`` must pass.

    Catches the lint class mypy can't — notably pyflakes ``F823`` (a ``cast``
    target that references a not-yet-assigned function-local), the exact bug
    the type cleanup risked introducing.
    """
    pytest.importorskip("ruff", reason="ruff not installed (dev extra)")
    result = _run("ruff", "check", _TARGET)
    assert result.returncode == 0, (
        "ruff check found lint errors in src/hfl:\n\n" + result.stdout + result.stderr
    )


@pytest.mark.integration
def test_ruff_format_clean_on_src() -> None:
    """``ruff format --check src/hfl`` must report no reformatting needed."""
    pytest.importorskip("ruff", reason="ruff not installed (dev extra)")
    result = _run("ruff", "format", "--check", _TARGET)
    assert result.returncode == 0, (
        "ruff format would reshape files in src/hfl — run `ruff format src/hfl`:\n\n"
        + result.stdout
        + result.stderr
    )

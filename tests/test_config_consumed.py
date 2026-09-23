# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Every configuration field must be read by the code it claims to configure.

Seven ``HFLConfig`` fields were read by nothing, and three were advertised:
the README called ``HFL_QUEUE_ENABLED`` the dispatcher's "master switch",
``docs/env-vars.md`` and both architecture documents listed
``HFL_REGISTRY_SQLITE_TIMEOUT`` for a SQLite registry that does not exist,
and ``hfl config`` printed ``api_request_timeout`` under "Timeouts" as if
something enforced it. An operator who set any of them changed nothing and
was told nothing.

All seven were removed rather than wired, each for a reason:

* ``queue_enabled`` — wiring it would let one variable switch off the
  serialisation that keeps two requests off one non-reentrant model.
* ``download_timeout`` (1 h) and ``conversion_timeout`` (2 h) — totals.
  A total kills a healthy 400 GB download or 405B conversion; stalls are
  what needs bounding, and Hub transfers already carry a 10 s read timeout
  (plus ``hfl.hub.timeouts`` for API calls).
* ``registry_sqlite_busy_timeout`` — there is no SQLite registry.
* ``default_tts_sample_rate`` / ``default_tts_format`` — duplicates of the
  request schema's defaults, with no environment variable to set them.
* ``api_request_timeout`` — displayed, never applied.

The guard below is what should have existed: a field is consumed only if
something outside ``config.py`` reads it, and printing it in ``hfl config``
does not count.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from hfl.config import HFLConfig

SRC = Path(__file__).resolve().parents[1] / "src" / "hfl"

# Functions whose only job is to display configuration. A read there is a
# promise to the operator, not a use.
DISPLAY_ONLY = {("cli/main.py", "config")}


def _reads_outside_config() -> set[str]:
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel == "config.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skipped: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and (rel, node.name) in DISPLAY_ONLY:
                skipped.update(id(n) for n in ast.walk(node))
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)  # getattr(config, "name", ...)
    return names


def test_every_config_field_is_read_somewhere():
    used = _reads_outside_config()
    unread = sorted(f.name for f in dataclasses.fields(HFLConfig) if f.name not in used)
    assert unread == [], (
        f"HFLConfig fields nothing reads: {unread}. Setting them changes "
        "nothing. Wire each to the code it names, or delete it together with "
        "its docs — and printing it in `hfl config` is not wiring."
    )


def test_the_display_exclusion_is_not_blind():
    """The exclusion must find the function it names, or it excludes nothing
    and a display-only field would count as used."""
    tree = ast.parse((SRC / "cli/main.py").read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for rel, func in DISPLAY_ONLY:
        assert rel == "cli/main.py" and func in names, (rel, func)


def test_removed_variables_stay_out_of_the_docs():
    """The advertised-but-inert variables must not creep back into docs."""
    root = SRC.parents[1]
    docs = [root / "README.md", *(root / "docs").glob("*.md"), *(root / "docs").glob("*.html")]
    offenders = [
        f"{doc.name}: {var}"
        for doc in docs
        for var in ("HFL_QUEUE_ENABLED", "HFL_REGISTRY_SQLITE_TIMEOUT")
        if var in doc.read_text(encoding="utf-8")
    ]
    assert offenders == []

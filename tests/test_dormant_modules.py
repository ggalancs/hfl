# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Modules nothing imports — built, tested, and wired to nothing.

A June architecture review recorded six dormant *symbols*. Re-measuring
in September at module granularity found something larger and different
in kind: **twelve modules, ~2 200 lines, that no other module imports.**

The difference matters, because "dormant" turned out to be two unrelated
situations wearing one label:

* Code that was speculative and never found a home — `failover`,
  `circuit_breaker`, `async_wrapper`. Deleting it loses nothing.
* Code the project's own documentation presents as *active*. `CLAUDE.md`
  lists `api/timeout.py` among the cross-cutting concerns, next to rate
  limiting and streaming. If nothing imports it, request timeouts are not
  a subsystem with no callers — they are a feature that does not run. The
  same question hangs over `observability/signing.py` and
  `observability/audit.py`.

So this file deliberately does **not** delete anything. The first reading
of the problem produced a recommendation — "delete failover, circuit
breaker and retry" — that a finer reading showed to be wrong about
`retry`, whose `with_retry` is used by the downloader and whose
`RetryExhausted` is used by the CLI; only the `RetryContext` class inside
it is unused. A recommendation that was wrong once at module granularity
should not be executed at scale without the owner looking.

What this file does instead is make the set impossible to forget: the
inventory below is checked on every run, so a **new** orphan fails
immediately, and resolving an existing one — by wiring it or removing it
— also fails, prompting the entry to be struck off. An inventory that
only ever grows is a list nobody reads.

Note on granularity: a symbol-level version of this check flags 330 of
865 public symbols, because exception classes, public API surface and
dynamically dispatched methods all look unreferenced. That threshold is
useless. Import edges between modules are the signal that actually
distinguishes "nobody calls this subsystem" from "this is a library".
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hfl"

# Reached without an import edge. These are not findings.
REACHED_OTHERWISE: dict[str, str] = {
    "cli.main": "console_scripts entry point: hfl = hfl.cli.main:app (pyproject.toml).",
    "plugins": "Loaded through importlib at runtime, by design.",
}

# Orphans as of 2026-09-22, each with the decision it is waiting on.
# Striking one off is the point of the exercise.
KNOWN_ORPHANS: dict[str, str] = {
    # --- the three that were questions, now answered by measurement -----
    "api.timeout": "VERIFIED DEAD (2026-09-23). Superseded duplicate: "
    "hfl.api.helpers.run_dispatched enforces the same config.generation_timeout "
    "and is what 6 routers import. Measured: limit 0.25/0.75/2.0s cuts at "
    "exactly that, HTTP 504 code=TIMEOUT. This module even exports a "
    "run_with_timeout under the same name as the live one, which is how "
    "CLAUDE.md came to point at the inert copy. Safe to delete. "
    "See tests/test_timeouts_run.py.",
    # --- speculative, never found a home --------------------------------
    "engine.failover": "Assumes a multi-backend world HFL is not. Candidate for removal.",
    "utils.circuit_breaker": "Same assumption. Candidate for removal.",
    "engine.async_wrapper": "Superseded by asyncio.to_thread at the call sites.",
    "engine.observability": "EngineObserver; token accounting moved to run_dispatched "
    "in 0.18.0, which is why nothing calls it.",
    # --- smaller, unclassified ------------------------------------------
}


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def _orphan_modules() -> set[str]:
    """Modules under src/hfl that no other module under src/hfl imports.

    Both absolute (``from hfl.x.y import z``) and relative (``from .y
    import z``) forms are resolved — a scan that missed relative imports
    would invent orphans that are perfectly well connected.
    """
    modules = {_module_name(p): p for p in SRC.rglob("*.py") if p.name != "__init__.py"}
    imported: set[str] = set()

    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover — the suite fails elsewhere
            continue
        me = _module_name(path)
        pkg = me.rsplit(".", 1)[0] if "." in me else ""

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:  # relative: from .x import y / from ..x import y
                    parts = pkg.split(".") if pkg else []
                    base = ".".join(parts[: len(parts) - (node.level - 1)]) if parts else ""
                    target = f"{base}.{node.module}" if node.module else base
                elif node.module and node.module.startswith("hfl"):
                    target = node.module[len("hfl.") :] if node.module != "hfl" else ""
                else:
                    continue
                target = target.strip(".")
                if target and target != me:
                    imported.add(target)
                for alias in node.names:  # from hfl.engine import failover
                    candidate = f"{target}.{alias.name}".strip(".")
                    if candidate != me:
                        imported.add(candidate)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("hfl."):
                        target = alias.name[len("hfl.") :]
                        if target != me:
                            imported.add(target)

    return {m for m in modules if m not in imported and m not in REACHED_OTHERWISE}


def test_no_new_module_falls_out_of_the_import_graph():
    """A module nobody imports is either a bug or dead weight — never nothing."""
    new = sorted(_orphan_modules() - set(KNOWN_ORPHANS))
    assert not new, (
        "These modules are imported by nothing under src/hfl:\n  "
        + "\n  ".join(new)
        + "\n\nEither wire them to a call site, delete them, or — if they are "
        "reached some other way, like an entry point or importlib — add them to "
        "REACHED_OTHERWISE with that reason. Shipping a subsystem that never "
        "runs is how a documented feature turns out to be absent."
    )


def test_the_inventory_shrinks_when_an_orphan_is_resolved():
    """Striking entries off is the point; a list that only grows is ignored."""
    resolved = sorted(set(KNOWN_ORPHANS) - _orphan_modules())
    assert not resolved, (
        f"No longer orphaned: {resolved}. Remove them from KNOWN_ORPHANS — "
        "leaving a resolved entry behind makes the inventory a historical "
        "document instead of a live one."
    )


def test_every_known_orphan_still_exists():
    """A deleted module must leave the inventory with it."""
    missing = sorted(
        name
        for name in KNOWN_ORPHANS
        if not (SRC / Path(*name.split("."))).with_suffix(".py").exists()
    )
    assert not missing, f"KNOWN_ORPHANS names modules that are gone: {missing}"


def test_the_documented_subsystems_carry_their_verdict():
    """The three that looked like they might be absent features.

    All three were measured on 2026-09-23 rather than reasoned about, and
    they did not come out the same: `api/timeout.py` is a superseded
    duplicate (timeouts demonstrably run, from `helpers.py`), while
    `signing` and `audit` are unfinished features with no callers. The
    distinction decides whether deleting them is free or destructive, so
    the verdict travels with the entry.
    """
    # ``observability.audit`` left this list when it was wired into
    # ``require_owner`` — resolved entries are struck off rather than kept
    # as history, which is the same rule the inventory itself follows.
    for name in ("api.timeout",):
        assert name in KNOWN_ORPHANS, f"{name} dropped out of the inventory"
        assert "VERIFIED" in KNOWN_ORPHANS[name], (
            f"{name} was one of the three the docs presented as active. Its entry "
            "must carry the verdict the measurement produced, so nobody has to "
            "redo the investigation to know whether deleting it is safe."
        )

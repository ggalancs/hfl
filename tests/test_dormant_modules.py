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

# The inventory is empty as of 2026-09-23, and that is the point: every
# entry was either wired to a call site or removed. What remains is the
# guard — a new orphan fails immediately instead of sitting here for
# three months the way the last twelve did.
#
# Resolved by wiring: engine.embedding_pooling (the `pooling` field on
# /api/embed reached the engine), core.sessions (`hfl run --session` and
# `hfl sessions`), core.sandbox (`--sandbox` / HFL_SANDBOX),
# observability.audit (emitted from `require_owner`),
# observability.tracing (spans at startup and around every inference),
# observability.signing (a sixth probe in `hfl verify`), api.deprecation
# (RFC 8594 headers on the legacy embeddings endpoint).
#
# Resolved by removal, because they were not unfinished features:
# api.timeout (a superseded duplicate of hfl.api.helpers, which even
# shared a function name and so misled CLAUDE.md), engine.async_wrapper
# (superseded by asyncio.to_thread, and reviving it would let two
# coroutines onto one non-reentrant engine), engine.observability
# (superseded three times over: real timing fields since 0.18.2,
# `_account_generation`, and the benchmark harness), engine.failover
# (assumes a multi-backend world HFL is not) and utils.circuit_breaker
# (the only external service is the Hub, which already has `with_retry`,
# tuned in 0.16.2 precisely not to swallow permanent errors).
KNOWN_ORPHANS: dict[str, str] = {}


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


def _test_only_modules() -> set[str]:
    """Modules that only ``tests/`` imports.

    A sharper signal than the orphan check above, and one it misses. When
    the five dead modules were removed, three test files still imported
    them and the suite failed to collect — so "no module under src/hfl
    imports it" was true while "nothing uses it" was not. A module whose
    only callers are its own tests is dormant in the way that matters: it
    is exercised, so it looks alive, and it serves no request.
    """
    src_imported: set[str] = set()
    test_imported: set[str] = set()
    modules = {_module_name(p) for p in SRC.rglob("*.py") if p.name != "__init__.py"}

    def collect(root: Path, sink: set[str], strip_hfl: bool) -> None:
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            me = _module_name(path) if strip_hfl else None
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("hfl."):
                    target = node.module[len("hfl.") :]
                    if target != me:
                        sink.add(target)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("hfl."):
                            target = alias.name[len("hfl.") :]
                            if target != me:
                                sink.add(target)

    collect(SRC, src_imported, strip_hfl=True)
    collect(SRC.parents[1] / "tests", test_imported, strip_hfl=False)

    return {
        m
        for m in modules
        if m in test_imported and m not in src_imported and m not in REACHED_OTHERWISE
    }


def test_no_module_is_exercised_only_by_its_own_tests():
    """Tested but never called is dormancy wearing a green tick.

    Learned the hard way: the orphan check above passed for five modules
    that three test files still imported, because it only ever looked at
    ``src/hfl``. A module can have full coverage and serve no request.
    """
    test_only = sorted(_test_only_modules())
    assert not test_only, (
        "These modules are imported by tests but by nothing in src/hfl:\n  "
        + "\n  ".join(test_only)
        + "\n\nCoverage is not use. Wire them to a call site or remove them "
        "together with the tests whose only subject they are."
    )


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


def test_the_inventory_is_empty_and_stays_that_way():
    """The state this file was created to reach.

    It began with twelve entries, three of them modules the docs
    presented as active. All twelve are resolved — wired or removed — so
    the dictionary is empty and the tests above are now pure regression
    guards rather than a backlog.

    If an entry ever reappears, it should carry the decision it is
    waiting on, not just a name: the reason a module is unreferenced is
    what decides whether removing it is free or destructive, and
    rediscovering that costs an afternoon.
    """
    assert KNOWN_ORPHANS == {}, (
        f"new dormant modules were accepted into the inventory: "
        f"{sorted(KNOWN_ORPHANS)}. That is allowed, but each needs its "
        "verdict written down, and it should not sit here for months."
    )

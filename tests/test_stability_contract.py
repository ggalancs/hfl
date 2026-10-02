# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The stability contract (``docs/stability.md``), held against the code.

``tests/stability/surface.json`` records HFL's public surface: the HTTP
routes, the CLI commands with their arguments and options, the documented
``HFL_*`` variables and the fields of a ``models.json`` entry. Something
recorded may leave only after a release that deprecated it (listed under
``deprecated``, with that version); something new must be recorded, so it
joins the contract on purpose. ``scripts/stability_surface.py --write``
records the current surface (keeping ``deprecated``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RECORDED = json.loads((REPO / "tests" / "stability" / "surface.json").read_text())


def _fresh(code: str) -> object:
    """Run ``code`` in a new interpreter and return the JSON it prints:
    other tests reload or replace the server and CLI modules, and the
    contract must not depend on the order tests run in."""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)


def _keys(surface: dict) -> set[str]:
    """One string per contract item: ``route GET /x``, ``cli serve``,
    ``cli serve --port``, ``env HFL_X``, ``manifest name``."""
    keys = {f"route {r}" for r in surface["routes"]}
    for command, params in surface["cli"].items():
        keys.add(f"cli {command}")
        keys.update(f"cli {command} {p}" for p in params)
    keys.update(f"env {v}" for v in surface["env"])
    keys.update(f"manifest {f}" for f in surface["manifest_fields"])
    return keys


@pytest.fixture(scope="module")
def current() -> set[str]:
    code = (
        "import json, runpy; "
        "s = runpy.run_path('scripts/stability_surface.py'); "
        "print(json.dumps(s['surface']()))"
    )
    surface = _fresh(code)
    assert isinstance(surface, dict)
    return _keys(surface)


def test_nothing_public_leaves_without_a_deprecation(current) -> None:
    gone = _keys(RECORDED) - current - set(RECORDED["deprecated"])
    assert not gone, (
        f"removed from the public surface without a deprecation: {sorted(gone)}. "
        "Deprecate first (docs/stability.md), or restore it."
    )


def test_new_public_surface_is_recorded(current) -> None:
    new = current - _keys(RECORDED)
    assert not new, (
        f"new public surface not in the contract: {sorted(new)}. "
        "Record it: python scripts/stability_surface.py --write"
    )


def test_each_deprecation_names_its_version_and_replacement() -> None:
    for key, note in RECORDED["deprecated"].items():
        assert note.get("since") and note.get("use"), key
        assert key.split(" ", 1)[0] in {"route", "cli", "env", "manifest"}, key


def test_a_deprecated_route_says_so_on_the_wire() -> None:
    """Recorded deprecated routes are marked so in the OpenAPI schema;
    ``tests/test_deprecation_wired.py`` ties the mark to the headers."""
    code = (
        "import json, runpy; "
        "s = runpy.run_path('scripts/stability_surface.py'); "
        "print(json.dumps(s['deprecated_routes']()))"
    )
    marked = {f"route {r}" for r in _fresh(code)}  # type: ignore[union-attr]
    recorded = {k for k in RECORDED["deprecated"] if k.startswith("route ")}
    assert recorded <= marked, f"recorded as deprecated, not marked in the app: {recorded - marked}"


def test_a_registry_entry_from_another_version_loads() -> None:
    """An older entry (only the fields every version wrote) and a newer one
    (a field this version does not know) both load."""
    from hfl.models.manifest import ModelManifest

    oldest = {"name": "m", "repo_id": "o/m", "local_path": "/x/m.gguf", "format": "gguf"}
    assert ModelManifest.from_dict(oldest).name == "m"
    assert ModelManifest.from_dict({**oldest, "added_in_a_later_version": 1}).repo_id == "o/m"

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HFL's public surface, as ``docs/stability.md`` defines it, read from the code.

    python scripts/stability_surface.py           # print it
    python scripts/stability_surface.py --write   # record it in tests/stability/surface.json

``tests/test_stability_contract.py`` compares the code with the recorded
surface: something new must be recorded (it becomes part of the contract on
purpose), and something recorded can leave only after being listed under
``deprecated`` in that file — with the version that deprecated it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SURFACE = REPO / "tests" / "stability" / "surface.json"


def _schema() -> dict:
    from hfl.api.server import app

    return app.openapi()


def routes() -> list[str]:
    """``METHOD /path`` of every operation in the server's OpenAPI schema —
    what clients see (hidden routes, like ``/ui``, are not in it), and stable
    across FastAPI versions, unlike ``app.routes`` (0.141 wraps included
    routers). Plus the WebSocket routes, which OpenAPI does not describe."""
    methods = {"get", "put", "post", "delete", "patch", "head"}
    found = {
        f"{method.upper()} {path}"
        for path, operations in _schema()["paths"].items()
        for method in operations
        if method in methods
    }
    found.update(f"WS {path}" for path in _websockets())
    return sorted(found)


def deprecated_routes() -> list[str]:
    """The operations the schema marks ``deprecated``."""
    return sorted(
        f"{method.upper()} {path}"
        for path, operations in _schema()["paths"].items()
        for method, operation in operations.items()
        if isinstance(operation, dict) and operation.get("deprecated")
    )


def _websockets() -> set[str]:
    from fastapi import APIRouter
    from fastapi.routing import APIWebSocketRoute

    from hfl.api import server

    found = set()
    for value in server.__dict__.values():
        # A router imported as itself (``ws_router``) or through its module.
        router = value if isinstance(value, APIRouter) else getattr(value, "router", None)
        for route in getattr(router, "routes", []):
            if isinstance(route, APIWebSocketRoute):
                found.add(route.path)
    return found


def cli() -> dict[str, list[str]]:
    """Every command (sub-commands as ``group command``) with its arguments
    and options. Read through what every click (and Typer's own copy of it,
    from 0.27) has: ``commands`` on a group, ``param_type_name`` on a
    parameter."""
    import typer

    from hfl.cli.main import app

    found: dict[str, list[str]] = {}

    def walk(command: object, prefix: str) -> None:
        commands = getattr(command, "commands", None)
        if isinstance(commands, dict):
            for name, sub in commands.items():
                if not getattr(sub, "hidden", False):
                    walk(sub, f"{prefix} {name}".strip())
            return
        params = getattr(command, "params", [])
        options = sorted(
            opt
            for p in params
            if p.param_type_name == "option" and not getattr(p, "hidden", False)
            for opt in p.opts
            if opt.startswith("--")
        )
        arguments = [f"<{p.name}>" for p in params if p.param_type_name == "argument"]
        found[prefix] = arguments + options

    walk(typer.main.get_command(app), "")
    return dict(sorted(found.items()))


def env_vars() -> list[str]:
    """The ``HFL_*`` variables ``docs/env-vars.md`` documents."""
    text = (REPO / "docs" / "env-vars.md").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"`(HFL_[A-Z0-9_]+)`", text)))


def manifest_fields() -> list[str]:
    """The fields of each ``models.json`` entry."""
    from hfl.models.manifest import ModelManifest

    return sorted(f.name for f in dataclasses.fields(ModelManifest))


def surface() -> dict[str, object]:
    return {
        "routes": routes(),
        "cli": cli(),
        "env": env_vars(),
        "manifest_fields": manifest_fields(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help=f"record it in {SURFACE}")
    args = parser.parse_args()
    current = surface()
    if not args.write:
        print(json.dumps(current, indent=2))
        return 0
    recorded = json.loads(SURFACE.read_text()) if SURFACE.exists() else {}
    current["deprecated"] = recorded.get("deprecated", {})
    SURFACE.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
    print(f"recorded in {SURFACE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

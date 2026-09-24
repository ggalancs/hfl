# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Deleting a local model — one rule for ``hfl rm`` and ``/api/delete``.

HFL deletes only what lives inside its own models folder. An entry that
points anywhere else (a GGUF of the user's registered in place, a leftover
pointing at /tmp) loses its registry entry and keeps its file. A file that
another entry still uses (``hfl cp`` is zero-copy) is kept as well.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hfl.models.manifest import ModelManifest


@dataclass
class Removal:
    name: str
    deleted: bool = False
    shared_with: list[str] = field(default_factory=list)
    kept_outside: Path | None = None


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _inside(path: Path, folder: Path) -> bool:
    """``path`` is strictly inside ``folder`` (never the folder itself)."""
    path, folder = _resolved(path), _resolved(folder)
    return path != folder and path.is_relative_to(folder)


def remove_model(registry: Any, manifest: ModelManifest) -> Removal:
    """Remove ``manifest`` from ``registry`` and delete its files if HFL owns them."""
    from hfl.config import config

    result = Removal(name=manifest.name)
    path = Path(manifest.local_path)
    target = _resolved(path)
    result.shared_with = sorted(
        other.name
        for other in registry.list_all()
        if other.name != manifest.name and _resolved(Path(other.local_path)) == target
    )
    if not result.shared_with:
        if not _inside(path, config.models_dir):
            result.kept_outside = path
        elif path.is_dir():
            shutil.rmtree(path)
            result.deleted = True
        elif path.is_file():
            path.unlink()
            result.deleted = True
    registry.remove(manifest.name)
    return result

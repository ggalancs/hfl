# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Where a GGUF vision model's image projector (``mmproj``) is.

Hub repos ship it beside the weights under names such as
``mmproj-F16.gguf`` or ``mmproj-SmolVLM-256M-Instruct-f16.gguf``; ``hfl pull``
fetches it with the model. A split model's parts may sit in a quant folder
(``BF16/...-00001-of-00002.gguf``) with the projector at the repo's root, so
the repo folder is looked in too — only inside HFL's models folder: a GGUF
registered from anywhere else must not pick up a stranger's projector.
"""

from __future__ import annotations

from pathlib import Path

_PREFERRED = ("f16", "bf16", "q8_0", "f32")


def _best(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda p: p.name.lower())
    for quant in _PREFERRED:
        for path in ranked:
            stem = path.name.lower().removesuffix(".gguf")
            if stem.endswith(quant) and not stem.endswith("b" + quant):
                return path
    return ranked[0]


def _in_folder(folder: Path) -> list[Path]:
    return [p for p in folder.glob("*.gguf") if "mmproj" in p.name.lower() and p.is_file()]


def find_projector(model_path: Path) -> Path | None:
    """The projector of the GGUF at ``model_path``, or None for a text model."""
    folder = model_path if model_path.is_dir() else model_path.parent
    found = _best(_in_folder(folder))
    if found is not None:
        return found
    try:
        from hfl.config import config

        models_dir = Path(config.models_dir).resolve()
        repo = folder.parent
        where = repo.resolve()
    except Exception:  # pragma: no cover - config always importable
        return None
    if where != models_dir and models_dir in where.parents:
        return _best(_in_folder(repo))
    return None

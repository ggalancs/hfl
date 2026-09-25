# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl import``: serve a GGUF you already have, where it is.

A model downloaded by LM Studio, llama.cpp or by hand can be registered
without copying it into ``~/.hfl`` and without a running server: the
registry entry points at the file in place. ``hfl rm`` never deletes a file
outside HFL's models folder, so removing the entry leaves the file alone.

A folder may be given instead of a file when it holds one model: its GGUF,
or the first part of a split one. Image projectors (``mmproj``) are not
models; one beside the file is found at load, as for a pulled model.
"""

from __future__ import annotations

import re
from pathlib import Path

from hfl.models.manifest import ModelManifest

_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})$", re.IGNORECASE)


class ImportRefused(ValueError):
    """Why a path cannot be imported, with an i18n key and its fields."""

    def __init__(self, key: str, **fields: str) -> None:
        super().__init__(key)
        self.key, self.fields = key, fields


def _part(path: Path) -> int | None:
    """``...-00002-of-00003.gguf`` → 2; ``None`` for a single file."""
    found = _SPLIT.search(path.stem)
    return int(found.group(1)) if found else None


def _is_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False


def choose_gguf(path: Path) -> Path:
    """The model file ``path`` names: itself, or the one model in a folder."""
    path = path.expanduser()
    if not path.exists():
        raise ImportRefused("import.not_found", path=str(path))
    if path.is_dir():
        # Projectors are not models; a split model is its first part.
        models = sorted(
            p
            for p in path.glob("*.gguf")
            if "mmproj" not in p.name.lower() and _part(p) in (None, 1)
        )
        if not models:
            raise ImportRefused("import.no_gguf", path=str(path))
        if len(models) > 1:
            names = ", ".join(p.name for p in models)
            raise ImportRefused("import.several", path=str(path), files=names)
        path = models[0]
    if "mmproj" in path.name.lower():
        raise ImportRefused("import.projector", path=str(path))
    if (_part(path) or 1) > 1:
        # A later part of a split model: llama.cpp opens the first.
        first = path.with_name(_SPLIT.sub(lambda m: f"-00001-of-{m.group(2)}", path.stem) + ".gguf")
        if not first.exists():
            raise ImportRefused("import.no_first_part", path=str(path))
        path = first
    if path.suffix.lower() != ".gguf" or not _is_gguf(path):
        raise ImportRefused("import.not_gguf", path=str(path))
    return path.resolve()


def default_name(model: Path) -> str:
    """``Qwen3-8B-Q4_K_M-00001-of-00002.gguf`` → ``qwen3-8b-q4_k_m``."""
    stem = _SPLIT.sub("", model.stem)
    name = re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-.")
    return name or "imported-model"


def manifest_for(model: Path, name: str, alias: str | None = None) -> ModelManifest:
    """The registry entry for a GGUF imported in place."""
    from hfl.engine.llama_cpp import _read_gguf_model_info
    from hfl.hub.resolver import _detect_quant

    parts = [model]
    split = _SPLIT.search(model.stem)
    if split:
        prefix = model.stem[: split.start()]
        parts = sorted(model.parent.glob(f"{prefix}-*-of-{split.group(2)}.gguf"))
    info = _read_gguf_model_info(str(model)) or {}
    return ModelManifest(
        name=name,
        repo_id=f"local/{name}",
        local_path=str(model),
        format="gguf",
        alias=alias,
        size_bytes=sum(p.stat().st_size for p in parts),
        quantization=_detect_quant(model.name),
        architecture=info.get("architecture"),
        model_type="llm",
    )

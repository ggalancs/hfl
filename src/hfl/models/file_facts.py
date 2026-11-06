# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""What a model's own files say about it, for ``/api/show`` and the lists.

Ollama copies ``model_info`` from the GGUF header, and clients size their
prompts from it: Copilot Chat's Ollama provider reads
``model_info[general.architecture + ".context_length"]`` and takes 32768
when it is missing. HFL filled ``model_info`` and ``details.family`` from
the manifest, whose ``architecture`` no pull records and whose
``context_length`` is the window the owner pinned (0 = automatic), not the
one the model was trained for. Measured on 2026-11-06: all eleven of the
owner's models reported ``unknown`` and no context at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The facts, and where a transformers-style config.json keeps each one.
_CONFIG_KEYS = {
    "context_length": "max_position_embeddings",
    "embedding_length": "hidden_size",
    "block_count": "num_hidden_layers",
    "head_count": "num_attention_heads",
    "head_count_kv": "num_key_value_heads",
}


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _top_model_type(directory: Path) -> str | None:
    """``model_type`` of the checkpoint itself, not of its nested language
    model (a vision model's ``text_config`` says ``gemma4_text``)."""
    try:
        cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    kind = cfg.get("model_type") if isinstance(cfg, dict) else None
    return kind.strip() if isinstance(kind, str) and kind.strip() else None


def file_facts(model_path: str | Path) -> dict[str, Any]:
    """Architecture and layout from the GGUF header or ``config.json``.

    Keys, each only when the files give it: ``architecture`` and the
    counts ``context_length`` (trained), ``embedding_length``,
    ``block_count``, ``head_count``, ``head_count_kv``. Empty for a path
    that holds neither. Never raises.
    """
    from hfl.engine.footprint import _gguf_files, _text_config
    from hfl.engine.llama_cpp import _read_gguf_model_info

    if not str(model_path or ""):
        return {}  # Path("") is the working directory
    path = Path(model_path)
    try:
        ggufs = _gguf_files(path)
        if ggufs:
            info = _read_gguf_model_info(str(ggufs[0])) or {}
            arch = info.get("architecture")
            counts = {
                "context_length": info.get("max_context"),
                "embedding_length": info.get("embedding_length"),
                "block_count": info.get("block_count"),
                "head_count": info.get("head_count"),
                "head_count_kv": info.get("head_count_kv"),
            }
        elif path.is_dir():
            arch = _top_model_type(path)
            cfg = _text_config(path)
            counts = {fact: cfg.get(key) for fact, key in _CONFIG_KEYS.items()}
        else:
            return {}
    except OSError:
        return {}
    facts: dict[str, Any] = {}
    if isinstance(arch, str) and arch:
        facts["architecture"] = arch
    for fact, value in counts.items():
        count = _count(value)
        if count is not None:
            facts[fact] = count
    return facts


def architecture(manifest: Any) -> str | None:
    """The model's architecture as its files name it, else the manifest's.

    Only the first key of a GGUF header is read, so listing every model
    stays cheap."""
    from hfl.converter.gguf_header import read_fields
    from hfl.engine.footprint import _gguf_files

    local = str(getattr(manifest, "local_path", "") or "")
    found: Any = None
    if local:  # Path("") is the working directory
        path = Path(local)
        try:
            ggufs = _gguf_files(path)
            if ggufs:
                fields = read_fields(ggufs[0], {"general.architecture"})
                found = fields.get("general.architecture")
            elif path.is_dir():
                found = _top_model_type(path)
        except (OSError, ValueError):
            found = None
    if isinstance(found, str) and found.strip():
        return found.strip()
    return getattr(manifest, "architecture", None) or None

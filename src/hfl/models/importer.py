# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl import``: serve a model you already have, where it is.

A model downloaded by LM Studio, llama.cpp, ``huggingface-cli`` or by hand
can be registered without copying it into ``~/.hfl`` and without a running
server: the registry entry points at it in place. ``hfl rm`` never deletes
anything outside HFL's models folder, so removing the entry leaves it alone.

Two kinds: a GGUF (a file, or a folder holding one model: its GGUF, or the
first part of a split one — image projectors, ``mmproj``, are not models;
one beside the file is found at load, as for a pulled model), and a
Hugging Face folder (``config.json`` and ``.safetensors`` weights: MLX
builds as LM Studio keeps them, or a model as the Hub has it), served by the
backend a pulled one would get.
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


# What a folder needs to be run: one of these tokenizer files.
_TOKENIZERS = ("tokenizer.json", "tokenizer.model", "tiktoken.model", "vocab.json")


def _hf_folder(path: Path) -> bool:
    """A Hugging Face model folder: its config and safetensors weights."""
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def choose_model(path: Path) -> tuple[Path, str]:
    """The model ``path`` names and its format: ``("gguf" | "safetensors")``.

    A folder with ``config.json`` and ``.safetensors`` is the model itself (a
    weights file inside it names the folder too); anything else is a GGUF.
    """
    path = path.expanduser()
    folder = path.parent if path.is_file() and path.suffix == ".safetensors" else path
    if folder.is_dir() and _hf_folder(folder):
        if not any((folder / name).is_file() for name in _TOKENIZERS):
            raise ImportRefused("import.no_tokenizer", path=str(folder))
        return folder.resolve(), "safetensors"
    return choose_gguf(path), "gguf"


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
    """``Qwen3-8B-Q4_K_M-00001-of-00002.gguf`` → ``qwen3-8b-q4_k_m``; a
    folder → its name (``Qwen2.5-0.5B-Instruct-4bit`` → lower case)."""
    stem = model.name if model.is_dir() else _SPLIT.sub("", model.stem)
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


def _folder_quantization(config: dict) -> str | None:
    """What the weights are: ``4bit`` for an MLX quantized build, else the
    dtype (``BF16``)."""
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if isinstance(quant, dict) and isinstance(quant.get("bits"), int):
        return f"{quant['bits']}bit"
    dtype = config.get("torch_dtype") or config.get("dtype")
    names = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
    return names.get(dtype) if isinstance(dtype, str) else None


def manifest_for_folder(folder: Path, name: str, alias: str | None = None) -> ModelManifest:
    """The registry entry for a Hugging Face folder imported in place."""
    import json

    from hfl.converter.formats import (
        ModelType,
        detect_model_type,
        get_model_type_display_name,
        is_model_type_supported,
    )

    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ImportRefused("import.bad_config", path=str(folder)) from exc
    if not isinstance(config, dict):
        raise ImportRefused("import.bad_config", path=str(folder))
    kind = detect_model_type(folder)
    if kind == ModelType.UNKNOWN:
        kind = ModelType.LLM  # a config with no telling task: a chat model, as pull takes it
    if not is_model_type_supported(kind):
        raise ImportRefused(
            "import.unsupported", path=str(folder), kind=get_model_type_display_name(kind)
        )
    architectures = config.get("architectures")
    architecture = (
        architectures[0]
        if isinstance(architectures, list) and architectures
        else config.get("model_type")
    )
    return ModelManifest(
        name=name,
        repo_id=f"local/{name}",
        local_path=str(folder),
        format="safetensors",
        alias=alias,
        size_bytes=sum(p.stat().st_size for p in folder.glob("*.safetensors")),
        quantization=_folder_quantization(config),
        architecture=architecture if isinstance(architecture, str) else None,
        model_type=kind.value,
    )

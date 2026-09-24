# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""How much memory a model will take once loaded.

Two parts, because they answer to different knobs:

* **Weights** — the bytes of the weight files. For GGUF, MLX and
  safetensors the loaded size tracks the file size closely: measured on
  this project's hardware, a 17 GB MLX build took 16.8 GB of resident
  memory and an 8.4 GB GGUF took 9.0 GB with a 4096-token context.
* **KV cache** — ``2 (K and V) x layers x kv_heads x head_dim x bytes x
  context``. It depends on the context the model is opened with, which is
  why the same model can fit at 8k tokens and not at 128k.

The estimate is an upper bound on purpose. It feeds a decision that is
cheap to get wrong in one direction (refusing a load that would have
fitted, with the numbers shown) and expensive in the other (a host that
swaps, or a Metal allocation failure mid-generation).

``estimate_footprint`` runs before a load, when the context may still be
unknown; ``footprint_of_loaded`` re-measures with the context the engine
actually opened, which is what the residency accounting keeps.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["Footprint", "estimate_footprint", "footprint_of_loaded", "GIB"]

GIB = 1024**3

# Context assumed for backends that grow their KV cache on demand (MLX,
# transformers) when nobody asked for a size: a long conversation, not the
# model's advertised maximum, which for current models is often 128k+ and
# would make every estimate absurd. llama.cpp is different — it allocates
# the whole window at load — so its real context is always used when known.
DEFAULT_GROWING_CTX = 8192

# Element width of the llama.cpp KV cache per HFL_KV_CACHE_TYPE, relative
# to f16 (which ``_kv_bytes_per_token`` assumes).
_KV_TYPE_SCALE = {"f16": 1.0, "f32": 2.0, "bf16": 1.0, "q8_0": 0.53, "q4_0": 0.28, "q4_1": 0.31}

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".npz", ".pt", ".pth")
_SPLIT_GGUF = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$")


@dataclass(frozen=True)
class Footprint:
    """Estimated resident memory of one loaded model, in bytes."""

    weights_bytes: int
    kv_bytes: int
    n_ctx: int
    """Context the KV part was sized for; 0 when the KV part is unknown."""
    kv_known: bool
    """False when the file gave no layout to size the KV cache from — the
    total is then the weights alone, and says so."""

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_bytes

    @property
    def known(self) -> bool:
        return self.total_bytes > 0

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB


_UNKNOWN = Footprint(0, 0, 0, False)


# ----------------------------------------------------------------------
# GGUF
# ----------------------------------------------------------------------


def _gguf_files(path: Path) -> list[Path]:
    """Every file llama.cpp will map for this model.

    A split GGUF (``model-00001-of-00003.gguf``) loads all its shards; a
    directory holds one model, whose shards are its ``.gguf`` files minus
    any ``mmproj`` vision projector (loaded separately, if at all).
    """
    if path.is_file():
        match = _SPLIT_GGUF.match(path.name)
        if not match:
            return [path]
        stem = match.group("stem")
        return sorted(p for p in path.parent.glob(f"{stem}-*-of-*.gguf") if p.is_file())
    if path.is_dir():
        return sorted(
            p for p in path.glob("*.gguf") if p.is_file() and "mmproj" not in p.name.lower()
        )
    return []


def _gguf_kv_bytes_per_token(first: Path) -> int:
    from hfl.engine.llama_cpp import _kv_bytes_per_token, _read_gguf_model_info

    per_token_f16 = _kv_bytes_per_token(_read_gguf_model_info(str(first)))
    if not per_token_f16:
        return 0
    from hfl.config import config

    scale = _KV_TYPE_SCALE.get(str(getattr(config, "kv_cache_type", "f16")).lower(), 1.0)
    return int(per_token_f16 * scale)


def _gguf_max_ctx(first: Path) -> int:
    from hfl.engine.llama_cpp import _read_gguf_model_info

    info = _read_gguf_model_info(str(first)) or {}
    return int(info.get("max_context") or 0)


def _estimate_gguf(files: list[Path], n_ctx: int) -> Footprint:
    weights = sum(p.stat().st_size for p in files)
    per_token = _gguf_kv_bytes_per_token(files[0])
    if not per_token:
        return Footprint(weights, 0, 0, False)
    if n_ctx <= 0:
        # Not decided yet: llama.cpp will auto-size, bounded by the model's
        # advertised maximum. Size for the default a growing backend would
        # use, capped by that maximum; the post-load re-measure replaces it.
        max_ctx = _gguf_max_ctx(files[0])
        n_ctx = min(DEFAULT_GROWING_CTX, max_ctx) if max_ctx else DEFAULT_GROWING_CTX
    return Footprint(weights, per_token * n_ctx, n_ctx, True)


# ----------------------------------------------------------------------
# safetensors / MLX directories
# ----------------------------------------------------------------------


def _weight_files(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix in _WEIGHT_SUFFIXES]
    # A repo that ships both .safetensors and a legacy .bin copy loads one.
    if any(p.suffix == ".safetensors" for p in files):
        files = [p for p in files if p.suffix == ".safetensors"]
    return files


def _text_config(directory: Path) -> dict[str, Any]:
    try:
        cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(cfg, dict):
        return {}
    # Vision-language models nest the language model's layout.
    nested = cfg.get("text_config")
    if isinstance(nested, dict):
        return {**cfg, **nested}
    return cfg


def _hf_kv_bytes(cfg: dict[str, Any], n_ctx: int) -> int:
    """KV cache bytes for ``n_ctx`` tokens, from a transformers-style config.

    Hybrid-attention models (Gemma 3/4 and others) mark each layer in
    ``layer_types``: a ``sliding_attention`` layer keeps at most
    ``sliding_window`` tokens, and ``full_attention`` layers may use their
    own head count and width (``num_global_key_value_heads``,
    ``global_head_dim``). Treating every layer as full over-estimated
    gemma-4-31b 4.4x — measured 1.24 GiB of KV at 6 001 tokens, against
    1.25 GiB from this formula and 5.5 GiB from the uniform one.
    """
    layers = int(cfg.get("num_hidden_layers") or 0)
    heads = int(cfg.get("num_attention_heads") or 0)
    kv_heads = int(cfg.get("num_key_value_heads") or heads or 0)
    hidden = int(cfg.get("hidden_size") or 0)
    head_dim = int(cfg.get("head_dim") or (hidden // heads if heads else 0))
    if not (layers and kv_heads and head_dim and n_ctx > 0):
        return 0
    per_elem = 2 * 2  # K and V, 2-byte elements

    layer_types = cfg.get("layer_types")
    window = int(cfg.get("sliding_window") or 0)
    if not (isinstance(layer_types, list) and len(layer_types) == layers and window > 0):
        return per_elem * layers * kv_heads * head_dim * n_ctx

    full_heads = int(cfg.get("num_global_key_value_heads") or kv_heads)
    full_dim = int(cfg.get("global_head_dim") or head_dim)
    total = 0
    for kind in layer_types:
        if kind == "sliding_attention":
            total += per_elem * kv_heads * head_dim * min(n_ctx, window)
        else:
            total += per_elem * full_heads * full_dim * n_ctx
    return total


def _estimate_directory(directory: Path, n_ctx: int) -> Footprint:
    files = _weight_files(directory)
    if not files:
        return _UNKNOWN
    weights = sum(p.stat().st_size for p in files)
    cfg = _text_config(directory)
    if n_ctx <= 0:
        max_ctx = int(cfg.get("max_position_embeddings") or 0)
        n_ctx = min(DEFAULT_GROWING_CTX, max_ctx) if max_ctx else DEFAULT_GROWING_CTX
    kv = _hf_kv_bytes(cfg, n_ctx)
    if not kv:
        return Footprint(weights, 0, 0, False)
    return Footprint(weights, kv, n_ctx, True)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def estimate_footprint(model_path: str | Path, n_ctx: int = 0) -> Footprint:
    """Memory a model at ``model_path`` will take when opened at ``n_ctx``.

    ``n_ctx <= 0`` means "not decided yet". Never raises: a path that
    cannot be read yields an unknown footprint (``known`` is False), which
    callers must report as unknown rather than treat as free.
    """
    path = Path(model_path)
    try:
        ggufs = _gguf_files(path)
        if ggufs:
            return _estimate_gguf(ggufs, n_ctx)
        if path.is_dir():
            return _estimate_directory(path, n_ctx)
    except OSError as exc:
        logger.debug("could not size %s: %s", path, exc)
    return _UNKNOWN


def footprint_of_loaded(model_path: str | Path, engine: Any) -> Footprint:
    """Re-measure with the context the engine actually opened.

    llama.cpp reports the context it allocated, so its KV part becomes
    exact. Backends that report ``0`` keep the pre-load assumption.
    """
    ctx = 0
    try:
        ctx = int(getattr(engine, "context_size", 0) or 0)
    except (TypeError, ValueError):
        ctx = 0
    return estimate_footprint(model_path, ctx)

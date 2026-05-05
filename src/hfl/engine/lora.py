# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 F4 — LoRA hot-swap.

Apply / remove LoRA adapters on a running model without reloading
the base weights. Powers ``POST /api/lora/apply`` and
``POST /api/lora/remove`` plus the ``hfl lora`` CLI.

Pattern:

- ``apply_lora(engine, lora_path, scale=1.0)`` → adapter id
- ``remove_lora(engine, adapter_id)`` → bool
- ``list_loras(engine)`` → list[AdapterInfo]
- ``LoraRegistry`` keeps an in-memory map of (engine_id, adapter_id) →
  metadata so the route layer can answer "what's currently applied
  on this model?".

Implementation lives behind a small interface so the engine-specific
wiring (llama-cpp-python's ``Llama.set_lora_adapter``) is hidden;
tests inject a fake engine and exercise the registry / sequencing.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfl.engine.base import InferenceEngine

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterInfo",
    "LoraRegistry",
    "apply_lora",
    "remove_lora",
    "list_loras",
    "get_registry",
    "reset_registry",
]


@dataclass
class AdapterInfo:
    """One applied adapter row."""

    adapter_id: str
    path: str
    name: str | None
    scale: float
    engine_id: str


class LoraRegistry:
    """Thread-safe in-memory map: ``adapter_id`` → :class:`AdapterInfo`.

    The dispatcher serialises engine calls but the LoRA registry is
    consulted from /api/lora/* concurrently with /api/chat. A lock
    keeps the listing stable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, AdapterInfo] = {}

    def add(self, info: AdapterInfo) -> None:
        with self._lock:
            self._by_id[info.adapter_id] = info

    def remove(self, adapter_id: str) -> AdapterInfo | None:
        with self._lock:
            return self._by_id.pop(adapter_id, None)

    def list(self, engine_id: str | None = None) -> list[AdapterInfo]:
        with self._lock:
            entries = list(self._by_id.values())
        if engine_id is not None:
            entries = [e for e in entries if e.engine_id == engine_id]
        return entries

    def clear(self) -> None:
        with self._lock:
            self._by_id.clear()


_GLOBAL: LoraRegistry | None = None
_GLOBAL_LOCK = threading.Lock()


def get_registry() -> LoraRegistry:
    """Lazily-built singleton registry. Process-wide."""
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = LoraRegistry()
    return _GLOBAL


def reset_registry() -> None:
    """Wipe the global registry. Tests use this for isolation."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        _GLOBAL = None


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------


def _engine_id(engine: "InferenceEngine") -> str:
    """Stable identity for an engine instance — used as a partitioning
    key in the registry so multiple loaded models can each carry
    their own adapter set.
    """
    return f"engine-{id(engine)}"


def _set_lora(engine: "InferenceEngine", lora_path: str, scale: float) -> None:
    """Apply a LoRA via the engine's hot-swap entry point.

    llama-cpp-python ≥ 0.3 exposes ``Llama.set_lora_adapter`` which
    takes a ``LlamaLoraAdapter``; older builds shipped
    ``apply_lora_from_file`` returning bool. We try both. If neither
    exists we raise — the caller should surface that as 503.
    """
    setter = getattr(engine, "apply_lora", None)
    if callable(setter):
        setter(lora_path, scale)
        return

    inner = getattr(engine, "_model", None)
    if inner is not None:
        if hasattr(inner, "set_lora_adapter"):
            inner.set_lora_adapter(lora_path, scale)
            return
        if hasattr(inner, "apply_lora_from_file"):
            inner.apply_lora_from_file(lora_path)
            return

    raise RuntimeError(
        "engine does not expose a LoRA hot-swap API; needs llama-cpp-python or an HFL wrapper"
    )


def _unset_lora(engine: "InferenceEngine", adapter_id: str) -> None:
    """Remove an adapter by id. Most engines expose
    ``remove_lora_adapter`` or its older alias ``unload_lora``.
    """
    remover = getattr(engine, "remove_lora", None)
    if callable(remover):
        remover(adapter_id)
        return

    inner = getattr(engine, "_model", None)
    if inner is not None:
        for name in ("remove_lora_adapter", "unload_lora"):
            fn = getattr(inner, name, None)
            if callable(fn):
                fn(adapter_id)
                return

    raise RuntimeError("engine does not expose a LoRA removal API; cannot detach adapter")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_lora(
    engine: "InferenceEngine",
    *,
    lora_path: str,
    scale: float = 1.0,
    name: str | None = None,
) -> AdapterInfo:
    """Hot-apply a LoRA adapter to ``engine``.

    Args:
        engine: A loaded engine that supports LoRA hot-swap.
        lora_path: Filesystem path to the adapter (``adapter_model.safetensors``
            or the GGUF-flavoured ``ggml-adapter-model.bin``).
        scale: Mix weight, default 1.0. Multiple adapters can be
            stacked at fractional scales (e.g. 0.7 + 0.3).
        name: Optional friendly label surfaced by ``list_loras``.

    Raises:
        FileNotFoundError: ``lora_path`` doesn't exist.
        RuntimeError: the engine doesn't support hot-swap or the
            apply call raised.
    """
    p = Path(lora_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")

    if not (0.0 <= scale <= 5.0):
        # llama-cpp clamps internally but we surface a clear error
        # rather than silently accepting nonsense.
        raise ValueError(f"scale must be in [0.0, 5.0]; got {scale}")

    _set_lora(engine, str(p), scale)

    info = AdapterInfo(
        adapter_id=str(uuid.uuid4()),
        path=str(p),
        name=name,
        scale=scale,
        engine_id=_engine_id(engine),
    )
    get_registry().add(info)
    logger.info("LoRA applied: id=%s path=%s scale=%s", info.adapter_id, info.path, scale)
    return info


def remove_lora(engine: "InferenceEngine", adapter_id: str) -> bool:
    """Detach a previously-applied adapter.

    Returns ``False`` when the id is unknown to the registry; raises
    ``RuntimeError`` when the engine refuses the unset call.
    """
    info = get_registry().remove(adapter_id)
    if info is None:
        return False

    _unset_lora(engine, adapter_id)
    logger.info("LoRA removed: id=%s path=%s", info.adapter_id, info.path)
    return True


def list_loras(engine: "InferenceEngine | None" = None) -> list[AdapterInfo]:
    """Enumerate active adapters.

    When ``engine`` is provided, only adapters bound to that engine
    are returned; otherwise everything in the registry is listed.
    """
    eid = _engine_id(engine) if engine is not None else None
    return get_registry().list(eid)

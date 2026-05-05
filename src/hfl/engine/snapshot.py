# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 F6 — KV cache snapshot save/restore.

Most LLM servers re-process the system prompt and any few-shot
context every time a new conversation starts. ``llama-cpp-python``
exposes ``Llama.save_state`` / ``Llama.load_state`` which capture
the entire KV cache (tokens already processed + the cache tensors)
in a single binary blob. HFL exposes that as a server feature so
operators can:

- Save a "ready-to-go" snapshot after loading a long system prompt
  or expensive few-shot prefix.
- Restore that snapshot at the start of a new server process for
  warm-start (zero TTFT on the cached prefix).
- Persist conversation context across server restarts.

Snapshot format: a Python pickle of ``Llama.save_state()`` plus a
small JSON sidecar with metadata (model_name, tokens, created_at).
The pickle is HFL-internal — we do not document it as a stable
format; the sidecar is consulted before loading to verify
``model_name`` matches.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hfl.engine.base import InferenceEngine

logger = logging.getLogger(__name__)

__all__ = [
    "SnapshotMeta",
    "save_snapshot",
    "load_snapshot",
    "list_snapshots",
    "delete_snapshot",
]


@dataclass(frozen=True)
class SnapshotMeta:
    """Sidecar metadata. Stored as ``<name>.meta.json`` next to the
    binary state file."""

    name: str
    model: str
    tokens: int
    created_at: float
    bytes: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _snapshot_dir() -> Path:
    from hfl.config import config

    out = config.home_dir / "snapshots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _state_path(name: str) -> Path:
    return _snapshot_dir() / f"{name}.state"


def _meta_path(name: str) -> Path:
    return _snapshot_dir() / f"{name}.meta.json"


def _validate_name(name: str) -> None:
    """Reject path separators / parent traversal / odd unicode.

    Keeps snapshot files inside the dedicated dir even when an
    operator passes a malicious name through the API.
    """
    if not name or not name.strip():
        raise ValueError("snapshot name cannot be empty")
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"invalid snapshot name: {name!r}")
    if not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"snapshot name must be alphanumeric / -._: {name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_snapshot(engine: "InferenceEngine", *, name: str, model_name: str) -> SnapshotMeta:
    """Persist the engine's KV cache state to disk.

    Args:
        engine: A loaded engine. Must expose either
            ``save_state()`` (llama-cpp-python) or ``_model.save_state()``.
        name: Snapshot identifier — alphanumeric + ``-_.``. The state
            file lives under ``HFL_HOME/snapshots/<name>.state``;
            metadata under ``<name>.meta.json``.
        model_name: Logical model name used for the load-time check.

    Raises:
        ValueError: ``name`` is malformed.
        RuntimeError: the engine has no save_state path or it raised.
    """
    _validate_name(name)

    # llama-cpp-python's high-level Llama exposes save_state() at
    # the top level; some HFL wrappers proxy ``_model.save_state``.
    save_fn = getattr(engine, "save_state", None)
    if save_fn is None:
        save_fn = getattr(getattr(engine, "_model", None), "save_state", None)
    if save_fn is None:
        raise RuntimeError(
            "engine does not expose save_state(); KV snapshots require llama-cpp-python"
        )

    try:
        state = save_fn()
    except Exception as exc:
        raise RuntimeError(f"engine.save_state() failed: {exc}")

    state_path = _state_path(name)
    with state_path.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    tokens = int(getattr(state, "n_tokens", 0) or 0)
    meta = SnapshotMeta(
        name=name,
        model=model_name,
        tokens=tokens,
        created_at=time.time(),
        bytes=state_path.stat().st_size,
    )
    with _meta_path(name).open("w") as f:
        json.dump(meta.to_json(), f, indent=2)

    return meta


def load_snapshot(engine: "InferenceEngine", *, name: str, model_name: str) -> SnapshotMeta:
    """Restore a KV cache state into ``engine``.

    Validates that the sidecar's ``model`` matches ``model_name``
    so a snapshot taken from a different model can't be force-loaded
    (the binary state contains model-specific tensor shapes — loading
    into the wrong model corrupts memory).
    """
    _validate_name(name)

    meta_p = _meta_path(name)
    state_p = _state_path(name)
    if not meta_p.exists() or not state_p.exists():
        raise FileNotFoundError(f"snapshot {name!r} not found")

    with meta_p.open() as f:
        meta_data = json.load(f)
    if meta_data.get("model") != model_name:
        raise ValueError(
            f"snapshot {name!r} was taken from model {meta_data.get('model')!r}, not {model_name!r}"
        )

    load_fn = getattr(engine, "load_state", None)
    if load_fn is None:
        load_fn = getattr(getattr(engine, "_model", None), "load_state", None)
    if load_fn is None:
        raise RuntimeError(
            "engine does not expose load_state(); KV snapshots require llama-cpp-python"
        )

    with state_p.open("rb") as f:
        state = pickle.load(f)

    try:
        load_fn(state)
    except Exception as exc:
        raise RuntimeError(f"engine.load_state() failed: {exc}")

    return SnapshotMeta(**meta_data)


def list_snapshots() -> list[SnapshotMeta]:
    """Return all registered snapshots ordered by creation time desc."""
    out: list[SnapshotMeta] = []
    for meta_p in _snapshot_dir().glob("*.meta.json"):
        try:
            with meta_p.open() as f:
                out.append(SnapshotMeta(**json.load(f)))
        except Exception:
            logger.warning("ignoring corrupt snapshot metadata at %s", meta_p)
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


def delete_snapshot(name: str) -> bool:
    """Remove a snapshot's state + sidecar. ``True`` when something
    was deleted, ``False`` when no matching files existed."""
    _validate_name(name)
    state_p = _state_path(name)
    meta_p = _meta_path(name)
    deleted = False
    if state_p.exists():
        state_p.unlink()
        deleted = True
    if meta_p.exists():
        meta_p.unlink()
        deleted = True
    return deleted

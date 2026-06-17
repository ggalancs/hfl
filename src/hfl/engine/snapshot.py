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

Security: ``pickle.loads`` executes arbitrary code during
deserialisation (CWE-502), so a tampered or attacker-planted
``.state`` file would be a code-execution primitive. Every state
blob is therefore authenticated with an HMAC-SHA256 keyed by a
per-installation secret (``HFL_HOME/snapshot.key``) and verified
*before* it is unpickled — only blobs this installation wrote will
load. Snapshots are a same-machine warm-start cache, not a portable
interchange format; copy one between machines and it will be
rejected (re-save it locally).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pickle
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hfl.engine.base import InferenceEngine

logger = logging.getLogger(__name__)

__all__ = [
    "SnapshotMeta",
    "SnapshotVersionMismatch",
    "save_snapshot",
    "load_snapshot",
    "list_snapshots",
    "delete_snapshot",
    "SNAPSHOT_FORMAT_VERSION",
    "SnapshotIntegrityError",
]


# Snapshot format version. Bumped whenever the on-disk pickle layout
# (the bytes ``Llama.save_state()`` returns) changes — usually driven
# by a llama-cpp-python major release that re-shapes the KV tensors.
# Loading a snapshot whose version doesn't match this constant raises
# ``SnapshotVersionMismatch`` rather than corrupting memory by feeding
# the wrong shape into ``Llama.load_state``.
SNAPSHOT_FORMAT_VERSION = 1


class SnapshotVersionMismatch(ValueError):
    """Raised when a snapshot file was produced by an older / newer
    HFL build whose on-disk format the current process can't safely
    consume.
    """


class SnapshotIntegrityError(ValueError):
    """Raised when a snapshot's state blob fails its HMAC integrity
    check (tampered, planted, or written by a different installation),
    so it is refused before :func:`pickle.loads` can run.
    """


@dataclass(frozen=True)
class SnapshotMeta:
    """Sidecar metadata. Stored as ``<name>.meta.json`` next to the
    binary state file."""

    name: str
    model: str
    tokens: int
    created_at: float
    bytes: int
    version: int = SNAPSHOT_FORMAT_VERSION
    """Format-version stamp written at save time. ``load_snapshot``
    rejects values that don't match
    :data:`SNAPSHOT_FORMAT_VERSION`."""
    mac: str = ""
    """HMAC-SHA256 of the pickled state blob, keyed by the
    per-installation snapshot key. Verified before unpickling to stop a
    tampered / planted ``.state`` file from executing code on load."""

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


def _snapshot_key() -> bytes:
    """Per-installation secret authenticating snapshot state blobs.

    Generated once (32 random bytes) and stored ``0600`` in the HFL home
    dir. This is a *local integrity* key, not a shared secret: its only
    job is to guarantee that a ``.state`` file unpickled by this
    installation was written by it, so a tampered or planted pickle
    cannot execute code via :func:`pickle.loads` (CWE-502).
    """
    from hfl.config import config

    key_path = config.home_dir / "snapshot.key"
    try:
        data = key_path.read_bytes()
        if len(data) >= 32:
            return data
    except FileNotFoundError:
        pass
    key = secrets.token_bytes(32)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    try:
        key_path.chmod(0o600)
    except OSError:  # pragma: no cover — non-POSIX filesystem
        pass
    return key


def _state_mac(raw: bytes) -> str:
    """Hex HMAC-SHA256 of ``raw`` under the per-installation key."""
    return hmac.new(_snapshot_key(), raw, hashlib.sha256).hexdigest()


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
        raise RuntimeError(f"engine.save_state() failed: {exc}") from exc

    state_path = _state_path(name)
    raw = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    # Atomic write: a crash / OOM mid-write must not leave a truncated ``.state``
    # that a later load would feed into ``Llama.load_state`` and corrupt the
    # model's memory. Write to a sibling temp then atomically rename.
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_bytes(raw)
    os.replace(tmp_path, state_path)

    tokens = int(getattr(state, "n_tokens", 0) or 0)
    meta = SnapshotMeta(
        name=name,
        model=model_name,
        tokens=tokens,
        created_at=time.time(),
        bytes=state_path.stat().st_size,
        mac=_state_mac(raw),
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
    # Reject foreign format versions before we feed bytes into
    # ``Llama.load_state``. Older snapshots stored without the
    # ``version`` field default to ``1`` for backward compatibility
    # with the initial release.
    found_version = int(meta_data.get("version", 1))
    if found_version != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotVersionMismatch(
            f"snapshot {name!r} was written with format version "
            f"{found_version}, this process expects {SNAPSHOT_FORMAT_VERSION}"
        )

    load_fn = getattr(engine, "load_state", None)
    if load_fn is None:
        load_fn = getattr(getattr(engine, "_model", None), "load_state", None)
    if load_fn is None:
        raise RuntimeError(
            "engine does not expose load_state(); KV snapshots require llama-cpp-python"
        )

    # Authenticate the state blob BEFORE unpickling: pickle.loads executes
    # arbitrary code during deserialisation (CWE-502), so a tampered or
    # attacker-planted .state file is an RCE primitive. The per-install
    # HMAC ensures we only unpickle blobs this installation wrote.
    raw = state_p.read_bytes()
    expected_mac = meta_data.get("mac")
    if not expected_mac:
        raise SnapshotIntegrityError(
            f"snapshot {name!r} has no integrity tag (created before integrity "
            "protection, or hand-edited) — refusing to unpickle. Re-save it."
        )
    if not hmac.compare_digest(_state_mac(raw), str(expected_mac)):
        raise SnapshotIntegrityError(
            f"snapshot {name!r} failed its integrity check — refusing to load a "
            "possibly-tampered state file."
        )
    state = pickle.loads(raw)

    try:
        load_fn(state)
    except Exception as exc:
        raise RuntimeError(f"engine.load_state() failed: {exc}") from exc

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

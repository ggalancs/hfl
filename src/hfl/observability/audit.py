# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Structured audit log for privileged events (Phase 14 P2 — V2 row 27).

Emits one JSON object per line to the file referenced by
``HFL_AUDIT_LOG_PATH`` (or a configured default). Events are
append-only with a size-based rotation (default 100 MB, keep the
last 5 rotations).

Event envelope::

    {
      "ts": "2026-04-17T11:22:33.456789Z",
      "event": "model.create",
      "actor": "api-key:abcd1234…",   # SHA-256 prefix, never the key
      "resource": "my-model",
      "metadata": {"parent": "llama3.3"},
      "outcome": "ok"
    }

Every privileged route / CLI command that mutates server state
records exactly one event. Read-only operations (``/api/tags``,
``/api/ps``) are NOT audited — they would dwarf the useful
signal.

The emitter is lazy and idempotent: calling ``audit_event`` when
auditing is disabled (no config, no env var) is a cheap no-op so
every callsite can assume the helper exists.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "audit_event",
    "configure_audit_log",
    "reset_audit_log",
    "AUDIT_EVENTS",
]


# Every event name we emit lives here — makes it easy to grep for
# the full catalogue and keeps the call sites consistent.
AUDIT_EVENTS = frozenset(
    {
        "model.create",
        "model.delete",
        "model.copy",
        "model.pull",
        "model.stop",
        "model.unload",
        "model.load",
        "blob.upload",
        "mcp.connect",
        "mcp.disconnect",
        "api_key.mint",
        "api_key.revoke",
    }
)


_lock = threading.Lock()
_audit_logger: logging.Logger | None = None
_audit_path: Path | None = None


def _default_path() -> Path | None:
    """Resolve the audit log path from env or config.

    Returns ``None`` when auditing is disabled, i.e. neither the
    env var nor a sensible default is present.
    """
    env = os.environ.get("HFL_AUDIT_LOG_PATH")
    if env:
        return Path(env)
    # Not enabled by default — opt-in keeps the footprint zero on
    # small installations that don't need the trail.
    return None


def configure_audit_log(
    path: str | os.PathLike | None = None,
    *,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> None:
    """Install the audit handler (idempotent).

    Called automatically on first ``audit_event`` call; tests can
    invoke it explicitly to redirect the sink.
    """
    global _audit_logger, _audit_path
    with _lock:
        if _audit_logger is not None and path is None:
            return
        if path is not None:
            _audit_path = Path(path)
        else:
            _audit_path = _default_path()
        if _audit_path is None:
            _audit_logger = None
            return

        _audit_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            str(_audit_path),
            maxBytes=max_bytes
            or int(os.environ.get("HFL_AUDIT_LOG_MAX_BYTES", str(100 * 1024 * 1024))),
            backupCount=backup_count
            if backup_count is not None
            else int(os.environ.get("HFL_AUDIT_LOG_BACKUPS", "5")),
            encoding="utf-8",
        )
        # Raw-passthrough formatter: ``audit_event`` already renders
        # JSON; the handler should just write the message.
        handler.setFormatter(logging.Formatter("%(message)s"))
        new_logger = logging.getLogger("hfl.audit")
        new_logger.setLevel(logging.INFO)
        # Drop any stale handlers from a previous configure() call.
        for old in list(new_logger.handlers):
            new_logger.removeHandler(old)
        new_logger.addHandler(handler)
        new_logger.propagate = False
        _audit_logger = new_logger


def reset_audit_log() -> None:
    """Test hook — forget the singleton and close handlers."""
    global _audit_logger, _audit_path
    with _lock:
        if _audit_logger is not None:
            for handler in list(_audit_logger.handlers):
                handler.close()
                _audit_logger.removeHandler(handler)
        _audit_logger = None
        _audit_path = None


def audit_event(
    event: str,
    *,
    actor: str | None = None,
    resource: str | None = None,
    metadata: dict[str, Any] | None = None,
    outcome: str = "ok",
) -> None:
    """Emit a single audit event.

    Zero-cost no-op when auditing isn't configured. Invalid event
    names log a warning (developer-facing) but never raise — the
    audit path must never abort a privileged operation.
    """
    if event not in AUDIT_EVENTS:
        logger.warning("unknown audit event: %s", event)
    if _audit_logger is None:
        configure_audit_log()
    if _audit_logger is None:
        return
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "event": event,
        "outcome": outcome,
    }
    if actor:
        payload["actor"] = actor
    if resource:
        payload["resource"] = resource
    if metadata:
        # Copy so callers can mutate their own dict afterwards.
        payload["metadata"] = dict(metadata)
    try:
        _audit_logger.info(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        # Never propagate audit-path failures to the caller.
        logger.exception("audit emission failed")

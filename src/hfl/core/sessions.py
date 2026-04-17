# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Session persistence (Phase 18 P3 — V2 row 36).

Captures a chat session — model name, options, and the conversation
so far — to a JSON file under ``~/.hfl/sessions/<name>.json`` so
users can resume long multi-turn exchanges across restarts.

The format is human-readable by design: users inspect / redact /
merge sessions by hand when they need to. No binary blobs, no
pickle.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "ChatSession",
    "save_session",
    "load_session",
    "list_sessions",
    "delete_session",
    "sessions_dir",
    "InvalidSessionNameError",
    "SessionNotFoundError",
]


_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class InvalidSessionNameError(ValueError):
    """Raised for session names that could escape the sessions dir."""


class SessionNotFoundError(FileNotFoundError):
    """Raised when ``load_session`` can't find the requested file."""


@dataclass
class ChatSession:
    """Serialisable snapshot of a chat session."""

    name: str
    model: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    options: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    system: str | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


def sessions_dir() -> Path:
    """Return ``~/.hfl/sessions`` creating it if missing."""
    from hfl.config import config

    path = config.home_dir / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _SESSION_NAME_RE.match(name):
        raise InvalidSessionNameError(
            f"session name {name!r} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}"
        )
    return name


def _path_for(name: str) -> Path:
    validated = _validate_name(name)
    return sessions_dir() / f"{validated}.json"


def save_session(session: ChatSession) -> Path:
    """Persist ``session`` to disk, returning the written path.

    Refreshes the ``updated_at`` stamp before write so subsequent
    ``list_sessions`` calls reflect the latest activity.
    """
    session.touch()
    path = _path_for(session.name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(asdict(session), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_session(name: str) -> ChatSession:
    """Load a previously-saved session by name."""
    path = _path_for(name)
    if not path.exists():
        raise SessionNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = ChatSession.__dataclass_fields__.keys()
    filtered = {k: v for k, v in data.items() if k in allowed}
    return ChatSession(**filtered)


def list_sessions() -> list[ChatSession]:
    """Return every saved session, newest first.

    Files with malformed JSON are skipped (the directory may have
    orphaned ``.tmp`` entries from an interrupted write).
    """
    out: list[ChatSession] = []
    for entry in sorted(sessions_dir().glob("*.json")):
        try:
            out.append(load_session(entry.stem))
        except (json.JSONDecodeError, InvalidSessionNameError, SessionNotFoundError):
            continue
    out.sort(key=lambda s: s.updated_at, reverse=True)
    return out


def delete_session(name: str) -> bool:
    """Delete ``name`` if present. Returns True on a real deletion."""
    path = _path_for(name)
    if not path.exists():
        return False
    path.unlink()
    return True

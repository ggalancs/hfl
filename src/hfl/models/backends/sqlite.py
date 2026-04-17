# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""SQLite-based registry backend for concurrent local access."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from hfl.config import config as _hfl_config
from hfl.models.backends.base import RegistryBackend

if TYPE_CHECKING:
    from hfl.models.manifest import ModelManifest


class SQLiteBackend(RegistryBackend):
    """SQLite-based storage backend.

    Suitable for concurrent local access by multiple processes.
    Uses SQLite's built-in locking for safety.

    Schema:
        models(name TEXT PRIMARY KEY, alias TEXT, data JSON)
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize the SQLite backend.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=_hfl_config.registry_sqlite_busy_timeout,
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    name TEXT PRIMARY KEY,
                    alias TEXT,
                    data TEXT NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_alias ON models(alias)
            """)
            self._conn.commit()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for atomic transactions."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except (sqlite3.Error, json.JSONDecodeError, ValueError, TypeError):
            self._conn.rollback()
            raise

    def load(self) -> list["ModelManifest"]:
        """Load all models from the database."""
        from hfl.models.manifest import ModelManifest

        with self._lock:
            cursor = self._conn.execute("SELECT data FROM models")
            rows = cursor.fetchall()
            return [ModelManifest.from_dict(json.loads(row["data"])) for row in rows]

    def save(self, models: list["ModelManifest"]) -> None:
        """Save all models to the database (replace all)."""
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute("DELETE FROM models")
                for model in models:
                    cursor.execute(
                        "INSERT INTO models (name, alias, data) VALUES (?, ?, ?)",
                        (model.name, model.alias, json.dumps(model.to_dict())),
                    )

    def add(self, manifest: "ModelManifest") -> None:
        """Add or update a model."""
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO models (name, alias, data)
                    VALUES (?, ?, ?)
                    """,
                    (manifest.name, manifest.alias, json.dumps(manifest.to_dict())),
                )

    def remove(self, name: str) -> bool:
        """Remove a model by name."""
        with self._lock:
            with self._transaction() as cursor:
                cursor.execute("DELETE FROM models WHERE name = ?", (name,))
                return cursor.rowcount > 0

    def get(self, name: str) -> "ModelManifest | None":
        """Get a model by name."""
        from hfl.models.manifest import ModelManifest

        with self._lock:
            cursor = self._conn.execute(
                "SELECT data FROM models WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                return ModelManifest.from_dict(json.loads(row["data"]))
            return None

    def get_by_alias(self, alias: str) -> "ModelManifest | None":
        """Get a model by alias.

        Args:
            alias: Alias to search for

        Returns:
            ModelManifest if found, None otherwise
        """
        from hfl.models.manifest import ModelManifest

        with self._lock:
            cursor = self._conn.execute(
                "SELECT data FROM models WHERE alias = ?",
                (alias,),
            )
            row = cursor.fetchone()
            if row:
                return ModelManifest.from_dict(json.loads(row["data"]))
            return None

    def update_alias(self, name: str, alias: str) -> bool:
        """Update the alias for a model."""
        with self._lock:
            # Check if alias already in use
            cursor = self._conn.execute(
                "SELECT name FROM models WHERE alias = ? OR name = ?",
                (alias, alias),
            )
            if cursor.fetchone():
                return False

            with self._transaction() as cursor:
                # Get current model data
                cursor.execute("SELECT data FROM models WHERE name = ?", (name,))
                row = cursor.fetchone()
                if not row:
                    return False

                # Update the alias in the JSON data as well
                from hfl.models.manifest import ModelManifest

                model = ModelManifest.from_dict(json.loads(row["data"]))
                model.alias = alias

                cursor.execute(
                    "UPDATE models SET alias = ?, data = ? WHERE name = ?",
                    (alias, json.dumps(model.to_dict()), name),
                )
                return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if hasattr(self._local, "conn") and self._local.conn is not None:
                self._local.conn.close()
                self._local.conn = None

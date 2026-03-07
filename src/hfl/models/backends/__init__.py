# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Pluggable registry backends.

This module provides different storage backends for the model registry:
- FileBackend: JSON file storage (default, suitable for single-node)
- SQLiteBackend: SQLite database storage (suitable for concurrent local access)

Usage:
    from hfl.models.backends import FileBackend, SQLiteBackend

    # Use file backend (default)
    backend = FileBackend(path)

    # Use SQLite for concurrent access
    backend = SQLiteBackend(db_path)
"""

from hfl.models.backends.base import RegistryBackend
from hfl.models.backends.file import FileBackend
from hfl.models.backends.sqlite import SQLiteBackend

__all__ = ["RegistryBackend", "FileBackend", "SQLiteBackend"]

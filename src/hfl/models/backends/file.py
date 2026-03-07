# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""File-based registry backend using JSON storage."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from hfl.models.backends.base import RegistryBackend

if TYPE_CHECKING:
    from hfl.models.manifest import ModelManifest

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt

    def _lock_file(fd: int, exclusive: bool) -> None:
        """Lock file on Windows."""
        msvcrt.locking(fd, msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)

    def _unlock_file(fd: int) -> None:
        """Unlock file on Windows."""
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # File might already be unlocked

else:
    import fcntl

    def _lock_file(fd: int, exclusive: bool) -> None:
        """Lock file on POSIX."""
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def _unlock_file(fd: int) -> None:
        """Unlock file on POSIX."""
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileBackend(RegistryBackend):
    """JSON file-based storage backend.

    Suitable for single-node deployments. Uses file locking for
    multi-process safety and an RLock for thread safety.

    Storage format: JSON array of model manifests.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the file backend.

        Args:
            path: Path to the JSON file for storage
        """
        self.path = path
        self._lock = threading.RLock()

    @contextmanager
    def _file_lock(self, exclusive: bool = False) -> Iterator[None]:
        """Acquire file lock for atomic operations (cross-platform).

        Args:
            exclusive: If True, acquire exclusive (write) lock.
                      If False, acquire shared (read) lock.
        """
        lock_path = self.path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        max_retries = 10
        retry_delay = 0.1

        try:
            for attempt in range(max_retries):
                try:
                    _lock_file(fd, exclusive)
                    break
                except (OSError, BlockingIOError):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)
            yield
        finally:
            _unlock_file(fd)
            os.close(fd)

    def load(self) -> list["ModelManifest"]:
        """Load all models from the JSON file."""
        from hfl.models.manifest import ModelManifest

        with self._lock:
            try:
                data = json.loads(self.path.read_text())
                return [ModelManifest.from_dict(m) for m in data]
            except (json.JSONDecodeError, FileNotFoundError):
                return []

    def save(self, models: list["ModelManifest"]) -> None:
        """Save all models to the JSON file."""
        with self._lock:
            with self._file_lock(exclusive=True):
                data = [m.to_dict() for m in models]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(data, indent=2))

    def add(self, manifest: "ModelManifest") -> None:
        """Add or update a model."""
        with self._lock:
            with self._file_lock(exclusive=True):
                models = self.load()
                # Remove existing with same name
                models = [m for m in models if m.name != manifest.name]
                models.append(manifest)
                # Save atomically
                data = [m.to_dict() for m in models]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(data, indent=2))

    def remove(self, name: str) -> bool:
        """Remove a model by name."""
        with self._lock:
            with self._file_lock(exclusive=True):
                models = self.load()
                initial_count = len(models)
                models = [m for m in models if m.name != name]

                if len(models) == initial_count:
                    return False

                data = [m.to_dict() for m in models]
                self.path.write_text(json.dumps(data, indent=2))
                return True

    def get(self, name: str) -> "ModelManifest | None":
        """Get a model by name."""
        with self._lock:
            models = self.load()
            for model in models:
                if model.name == name:
                    return model
            return None

    def update_alias(self, name: str, alias: str) -> bool:
        """Update the alias for a model."""
        with self._lock:
            with self._file_lock(exclusive=True):
                models = self.load()

                # Check if alias already in use
                for model in models:
                    if model.alias == alias or model.name == alias:
                        return False

                # Find and update the model
                for model in models:
                    if model.name == name:
                        model.alias = alias
                        data = [m.to_dict() for m in models]
                        self.path.write_text(json.dumps(data, indent=2))
                        return True

                return False

    def close(self) -> None:
        """No resources to close for file backend."""
        pass

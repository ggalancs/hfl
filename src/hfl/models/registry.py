# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Local registry for downloaded models.
Persists to ~/.hfl/models.json.

Features:
- O(1) lookup by name, alias, and repo_id using dict indexes
- Thread-safe operations with RLock
- File locking for multi-process safety
- Atomic save operations
- Corruption recovery with automatic backups
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from hfl.config import config
from hfl.models.manifest import ModelManifest

logger = logging.getLogger(__name__)

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


class ModelRegistry:
    """Manages the local model inventory with thread-safe operations.

    Uses dict indexes for O(1) lookups by name, alias, and repo_id.
    All public methods are thread-safe using an RLock.
    File operations use cross-platform file locking for multi-process safety.
    """

    def __init__(self) -> None:
        self.path = config.registry_path
        self._models: list[ModelManifest] = []
        # Indexes for O(1) lookup
        self._by_name: dict[str, ModelManifest] = {}
        self._by_alias: dict[str, ModelManifest] = {}
        self._by_repo_id: dict[str, ModelManifest] = {}
        # Thread safety
        self._lock = threading.RLock()
        self._load()

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

    def _load(self) -> None:
        """Load models from disk and build indexes.

        Includes corruption recovery: if the main file is corrupt,
        attempts to recover from backup.
        """
        if not self.path.exists():
            self._models = []
            self._rebuild_indexes()
            return

        try:
            data = json.loads(self.path.read_text())
            self._models = self._parse_manifests(data)
            logger.debug(f"Loaded {len(self._models)} models from registry")
        except json.JSONDecodeError as e:
            logger.warning(f"Registry file corrupt: {e}")
            self._models = self._recover_from_backup()
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._models = self._recover_from_backup()

        self._rebuild_indexes()

    def _parse_manifests(self, data: list) -> list[ModelManifest]:
        """Parse manifest data with validation.

        Invalid entries are logged and skipped rather than failing the entire load.
        """
        if not isinstance(data, list):
            raise ValueError("Registry data must be a list")

        manifests = []
        for i, entry in enumerate(data):
            try:
                manifest = ModelManifest.from_dict(entry)
                manifests.append(manifest)
            except Exception as e:
                logger.warning(f"Skipping invalid manifest entry {i}: {e}")
        return manifests

    def _recover_from_backup(self) -> list[ModelManifest]:
        """Attempt to recover registry from backup.

        Returns:
            List of recovered models, or empty list if recovery fails.
        """
        backup_path = self._backup_path
        if not backup_path.exists():
            logger.warning("No backup found for registry recovery")
            self._emit_corruption_event("no_backup")
            return []

        try:
            data = json.loads(backup_path.read_text())
            models = self._parse_manifests(data)
            logger.info(f"Recovered {len(models)} models from backup")
            self._emit_corruption_event("recovered", len(models))

            # Restore backup as main file
            shutil.copy2(backup_path, self.path)
            return models
        except Exception as e:
            logger.error(f"Backup recovery failed: {e}")
            self._emit_corruption_event("recovery_failed", error=str(e))
            return []

    @property
    def _backup_path(self) -> Path:
        """Path to the backup file."""
        return self.path.with_suffix(".json.bak")

    def _create_backup(self) -> None:
        """Create a backup of the current registry file."""
        if self.path.exists():
            try:
                shutil.copy2(self.path, self._backup_path)
                logger.debug("Created registry backup")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

    def _emit_corruption_event(
        self,
        recovery_status: str,
        models_recovered: int = 0,
        error: str | None = None,
    ) -> None:
        """Emit event for registry corruption."""
        try:
            from hfl.events import EventType, emit

            emit(
                EventType.ERROR,
                source="registry",
                error="registry_corruption",
                recovery_status=recovery_status,
                models_recovered=models_recovered,
                error_message=error,
            )
        except ImportError:
            pass

    def _rebuild_indexes(self) -> None:
        """Rebuild all lookup indexes from the model list."""
        self._by_name = {m.name: m for m in self._models}
        self._by_alias = {m.alias: m for m in self._models if m.alias}
        self._by_repo_id = {m.repo_id: m for m in self._models}

    def _save(self) -> None:
        """Persist models to disk with atomic write and backup.

        Creates a backup before saving and uses atomic write
        to prevent corruption.
        """
        # Create backup of current file
        self._create_backup()

        # Prepare data
        data = [m.to_dict() for m in self._models]

        # Atomic write: write to temp file then rename
        temp_path = self.path.with_suffix(".json.tmp")
        try:
            temp_path.write_text(json.dumps(data, indent=2))
            temp_path.replace(self.path)  # Atomic on POSIX
            logger.debug(f"Saved registry with {len(self._models)} models")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            raise

    def validate_integrity(self) -> tuple[bool, list[str]]:
        """Validate registry integrity.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check for duplicate names
        names = [m.name for m in self._models]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            errors.append(f"Duplicate model names: {set(duplicates)}")

        # Check for duplicate aliases
        aliases = [m.alias for m in self._models if m.alias]
        if len(aliases) != len(set(aliases)):
            duplicates = [a for a in aliases if aliases.count(a) > 1]
            errors.append(f"Duplicate aliases: {set(duplicates)}")

        # Check for missing local paths
        for model in self._models:
            if not model.local_path:
                errors.append(f"Model '{model.name}' has no local_path")
            elif not Path(model.local_path).exists():
                errors.append(f"Model '{model.name}' path does not exist: {model.local_path}")

        return len(errors) == 0, errors

    def repair(self) -> int:
        """Repair registry by removing invalid entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            original_count = len(self._models)

            # Remove models with non-existent paths
            valid_models = []
            for model in self._models:
                if model.local_path and Path(model.local_path).exists():
                    valid_models.append(model)
                else:
                    logger.warning(f"Removing invalid model: {model.name}")

            # Remove duplicates (keep first occurrence)
            seen_names = set()
            unique_models = []
            for model in valid_models:
                if model.name not in seen_names:
                    seen_names.add(model.name)
                    unique_models.append(model)
                else:
                    logger.warning(f"Removing duplicate model: {model.name}")

            self._models = unique_models
            self._rebuild_indexes()

            removed = original_count - len(self._models)
            if removed > 0:
                with self._file_lock(exclusive=True):
                    self._save()
                logger.info(f"Registry repair: removed {removed} invalid entries")

            return removed

    def add(self, manifest: ModelManifest) -> None:
        """Registers a new model (thread-safe).

        If a model with the same name exists, it is replaced.
        Uses file locking for multi-process safety.
        """
        with self._lock:
            with self._file_lock(exclusive=True):
                # Reload from disk to avoid stale data
                self._load()

                # Remove existing model with same name (if any)
                if manifest.name in self._by_name:
                    self._models = [m for m in self._models if m.name != manifest.name]
                self._models.append(manifest)
                self._rebuild_indexes()
                self._save()

    def get(self, name: str) -> ModelManifest | None:
        """Finds a model by name, alias, or repo_id (thread-safe).

        Lookup is O(1) for all three identifiers.
        """
        with self._lock:
            # Try name first (most common)
            if name in self._by_name:
                return self._by_name[name]
            # Try alias
            if name in self._by_alias:
                return self._by_alias[name]
            # Try repo_id
            if name in self._by_repo_id:
                return self._by_repo_id[name]
            return None

    def set_alias(self, name: str, alias: str) -> bool:
        """Sets an alias for an existing model (thread-safe).

        Returns False if:
        - The alias is invalid format
        - The alias is already in use
        - The model doesn't exist

        Raises:
            ValidationError: If alias format is invalid
        """
        from hfl.validators import validate_alias

        # Validate alias format first (raises ValidationError if invalid)
        validate_alias(alias)

        with self._lock:
            with self._file_lock(exclusive=True):
                # Reload from disk to avoid stale data
                self._load()

                # Verify the alias is not already in use (O(1) check)
                if alias in self._by_alias or alias in self._by_name:
                    return False

                model = self._by_name.get(name)
                if model is None:
                    return False

                model.alias = alias
                self._rebuild_indexes()
                self._save()
                return True

    def list_all(self) -> list[ModelManifest]:
        """Lists all registered models, sorted by creation date (thread-safe)."""
        with self._lock:
            return sorted(self._models, key=lambda m: m.created_at, reverse=True)

    def remove(self, name: str) -> bool:
        """Removes a model from the registry (thread-safe).

        Does not delete files, only registry entry.
        Returns True if a model was removed, False otherwise.
        """
        with self._lock:
            with self._file_lock(exclusive=True):
                # Reload from disk to avoid stale data
                self._load()

                if name not in self._by_name:
                    return False

                self._models = [m for m in self._models if m.name != name]
                self._rebuild_indexes()
                self._save()
                return True

    def __len__(self) -> int:
        """Return the number of registered models (thread-safe)."""
        with self._lock:
            return len(self._models)

    def __contains__(self, name: str) -> bool:
        """Check if a model exists by name, alias, or repo_id (thread-safe)."""
        return self.get(name) is not None

    def refresh(self) -> None:
        """Reload the registry from disk (thread-safe).

        Useful when models may have been added/removed by external processes.
        """
        with self._lock:
            with self._file_lock(exclusive=False):
                self._load()


# Singleton access delegated to container for unified management


def get_registry() -> ModelRegistry:
    """Get the singleton ModelRegistry instance.

    This avoids reloading from disk on every API call.
    """
    from hfl.core.container import get_registry as _get_registry

    return _get_registry()


def reset_registry() -> None:
    """Reset the registry cache (for testing)."""
    from hfl.core.container import get_container

    get_container().registry.reset()

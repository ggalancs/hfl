# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Local registry for downloaded models.
Persists to ~/.hfl/models.json.

Optimizations:
- O(1) lookup by name, alias, and repo_id using dict indexes
- Lazy index rebuilding only when needed
"""

import json

from hfl.config import config
from hfl.models.manifest import ModelManifest


class ModelRegistry:
    """Manages the local model inventory.

    Uses dict indexes for O(1) lookups by name, alias, and repo_id.
    """

    def __init__(self) -> None:
        self.path = config.registry_path
        self._models: list[ModelManifest] = []
        # Indexes for O(1) lookup
        self._by_name: dict[str, ModelManifest] = {}
        self._by_alias: dict[str, ModelManifest] = {}
        self._by_repo_id: dict[str, ModelManifest] = {}
        self._load()

    def _load(self) -> None:
        """Load models from disk and build indexes."""
        try:
            data = json.loads(self.path.read_text())
            self._models = [ModelManifest.from_dict(m) for m in data]
        except (json.JSONDecodeError, FileNotFoundError):
            self._models = []
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        """Rebuild all lookup indexes from the model list."""
        self._by_name = {m.name: m for m in self._models}
        self._by_alias = {m.alias: m for m in self._models if m.alias}
        self._by_repo_id = {m.repo_id: m for m in self._models}

    def _save(self) -> None:
        """Persist models to disk."""
        data = [m.to_dict() for m in self._models]
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, manifest: ModelManifest) -> None:
        """Registers a new model.

        If a model with the same name exists, it is replaced.
        """
        # Remove existing model with same name (if any)
        if manifest.name in self._by_name:
            self._models = [m for m in self._models if m.name != manifest.name]
        self._models.append(manifest)
        self._rebuild_indexes()
        self._save()

    def get(self, name: str) -> ModelManifest | None:
        """Finds a model by name, alias, or repo_id.

        Lookup is O(1) for all three identifiers.
        """
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
        """Sets an alias for an existing model.

        Returns False if:
        - The alias is already in use
        - The model doesn't exist
        """
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
        """Lists all registered models, sorted by creation date (newest first)."""
        return sorted(self._models, key=lambda m: m.created_at, reverse=True)

    def remove(self, name: str) -> bool:
        """Removes a model from the registry (does not delete files).

        Returns True if a model was removed, False otherwise.
        """
        if name not in self._by_name:
            return False

        self._models = [m for m in self._models if m.name != name]
        self._rebuild_indexes()
        self._save()
        return True

    def __len__(self) -> int:
        """Return the number of registered models."""
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        """Check if a model exists (by name, alias, or repo_id)."""
        return self.get(name) is not None

    def refresh(self) -> None:
        """Reload the registry from disk.

        Useful when models may have been added/removed by external processes.
        """
        self._load()


# Singleton instance cache
_registry_instance: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get the singleton ModelRegistry instance.

    This avoids reloading from disk on every API call.
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


def reset_registry() -> None:
    """Reset the registry cache (for testing)."""
    global _registry_instance
    _registry_instance = None

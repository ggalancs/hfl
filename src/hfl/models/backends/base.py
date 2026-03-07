# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Abstract base class for registry backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfl.models.manifest import ModelManifest


class RegistryBackend(ABC):
    """Abstract interface for registry storage backends.

    This allows the ModelRegistry to use different storage implementations
    while maintaining the same interface.

    Implementations must be thread-safe for single-process concurrent access.
    """

    @abstractmethod
    def load(self) -> list["ModelManifest"]:
        """Load all models from storage.

        Returns:
            List of ModelManifest objects
        """

    @abstractmethod
    def save(self, models: list["ModelManifest"]) -> None:
        """Save all models to storage.

        Args:
            models: List of ModelManifest objects to persist
        """

    @abstractmethod
    def add(self, manifest: "ModelManifest") -> None:
        """Add or update a model in storage.

        If a model with the same name exists, it should be replaced.

        Args:
            manifest: Model manifest to add/update
        """

    @abstractmethod
    def remove(self, name: str) -> bool:
        """Remove a model from storage by name.

        Args:
            name: Name of the model to remove

        Returns:
            True if a model was removed, False if not found
        """

    @abstractmethod
    def get(self, name: str) -> "ModelManifest | None":
        """Get a model by name.

        Args:
            name: Name of the model to retrieve

        Returns:
            ModelManifest if found, None otherwise
        """

    @abstractmethod
    def update_alias(self, name: str, alias: str) -> bool:
        """Update the alias for a model.

        Args:
            name: Name of the model
            alias: New alias to set

        Returns:
            True if updated, False if model not found or alias in use
        """

    @abstractmethod
    def close(self) -> None:
        """Close any open resources.

        Called when the registry is being shut down.
        """

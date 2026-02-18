# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 hfl Contributors
"""
Registro local de modelos descargados.
Persiste en ~/.hfl/models.json.
"""

import json
from hfl.config import config
from hfl.models.manifest import ModelManifest


class ModelRegistry:
    """Gestiona el inventario local de modelos."""

    def __init__(self):
        self.path = config.registry_path
        self._models: list[ModelManifest] = self._load()

    def _load(self) -> list[ModelManifest]:
        try:
            data = json.loads(self.path.read_text())
            return [ModelManifest.from_dict(m) for m in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self):
        data = [m.to_dict() for m in self._models]
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, manifest: ModelManifest):
        """Registra un nuevo modelo."""
        # Evitar duplicados
        self._models = [m for m in self._models if m.name != manifest.name]
        self._models.append(manifest)
        self._save()

    def get(self, name: str) -> ModelManifest | None:
        """Busca un modelo por nombre, alias o repo_id."""
        for m in self._models:
            if m.name == name or m.repo_id == name:
                return m
            if m.alias and m.alias == name:
                return m
        return None

    def set_alias(self, name: str, alias: str) -> bool:
        """Establece un alias para un modelo existente."""
        # Verificar que el alias no esté en uso
        for m in self._models:
            if m.alias == alias or m.name == alias:
                return False

        for m in self._models:
            if m.name == name:
                m.alias = alias
                self._save()
                return True
        return False

    def list_all(self) -> list[ModelManifest]:
        """Lista todos los modelos registrados."""
        return sorted(self._models, key=lambda m: m.created_at, reverse=True)

    def remove(self, name: str) -> bool:
        """Elimina un modelo del registro (no borra archivos)."""
        before = len(self._models)
        self._models = [m for m in self._models if m.name != name]
        if len(self._models) < before:
            self._save()
            return True
        return False

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests para el módulo models (manifest, registry)."""

import pytest
import json
from pathlib import Path
from datetime import datetime


class TestModelManifest:
    """Tests para ModelManifest dataclass."""

    def test_minimal_creation(self):
        """Creación con campos mínimos."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="test-model",
            repo_id="org/model",
            local_path="/path/to/model",
            format="gguf",
        )

        assert manifest.name == "test-model"
        assert manifest.repo_id == "org/model"
        assert manifest.local_path == "/path/to/model"
        assert manifest.format == "gguf"

    def test_full_creation(self):
        """Creación con todos los campos."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="test-model",
            repo_id="org/model",
            local_path="/path/to/model",
            format="gguf",
            size_bytes=5 * 1024**3,
            quantization="Q4_K_M",
            original_format="safetensors",
            architecture="llama",
            parameters="7B",
            context_length=8192,
            chat_template="{{ messages }}",
        )

        assert manifest.size_bytes == 5 * 1024**3
        assert manifest.quantization == "Q4_K_M"
        assert manifest.original_format == "safetensors"
        assert manifest.architecture == "llama"
        assert manifest.parameters == "7B"
        assert manifest.context_length == 8192
        assert manifest.chat_template == "{{ messages }}"

    def test_default_values(self):
        """Verifica valores por defecto."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="test",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
        )

        assert manifest.size_bytes == 0
        assert manifest.quantization is None
        assert manifest.original_format is None
        assert manifest.architecture is None
        assert manifest.parameters is None
        assert manifest.context_length == 4096
        assert manifest.chat_template is None
        assert manifest.last_used is None
        assert manifest.created_at is not None

    def test_created_at_auto_generated(self):
        """Verifica que created_at se genera automáticamente."""
        from hfl.models.manifest import ModelManifest

        before = datetime.now().isoformat()
        manifest = ModelManifest(
            name="test",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
        )
        after = datetime.now().isoformat()

        assert before <= manifest.created_at <= after

    def test_display_size_gigabytes(self):
        """Formateo de tamaño en GB."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="test",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
            size_bytes=5 * 1024**3,
        )

        assert manifest.display_size == "5.0 GB"

    def test_display_size_megabytes(self):
        """Formateo de tamaño en MB."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="test",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
            size_bytes=500 * 1024**2,
        )

        assert manifest.display_size == "500 MB"

    def test_display_size_decimal_gb(self):
        """Formateo de tamaño decimal en GB."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="test",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
            size_bytes=int(2.5 * 1024**3),
        )

        assert "2.5 GB" in manifest.display_size

    def test_to_dict(self):
        """Conversión a diccionario."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="test-model",
            repo_id="org/model",
            local_path="/path/to/model",
            format="gguf",
            size_bytes=1000,
            quantization="Q4_K_M",
        )

        d = manifest.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "test-model"
        assert d["repo_id"] == "org/model"
        assert d["size_bytes"] == 1000
        assert d["quantization"] == "Q4_K_M"
        assert "created_at" in d

    def test_from_dict(self):
        """Creación desde diccionario."""
        from hfl.models.manifest import ModelManifest

        data = {
            "name": "test-model",
            "repo_id": "org/model",
            "local_path": "/path",
            "format": "gguf",
            "size_bytes": 2000,
            "quantization": "Q5_K_M",
            "architecture": "mistral",
        }

        manifest = ModelManifest.from_dict(data)

        assert manifest.name == "test-model"
        assert manifest.repo_id == "org/model"
        assert manifest.size_bytes == 2000
        assert manifest.quantization == "Q5_K_M"
        assert manifest.architecture == "mistral"

    def test_from_dict_ignores_extra_fields(self):
        """from_dict ignora campos extra."""
        from hfl.models.manifest import ModelManifest

        data = {
            "name": "test",
            "repo_id": "org/model",
            "local_path": "/path",
            "format": "gguf",
            "extra_field": "ignored",
            "another_extra": 123,
        }

        manifest = ModelManifest.from_dict(data)

        assert manifest.name == "test"
        assert not hasattr(manifest, "extra_field")

    def test_roundtrip(self):
        """Verifica roundtrip to_dict -> from_dict."""
        from hfl.models.manifest import ModelManifest

        original = ModelManifest(
            name="roundtrip-test",
            repo_id="org/model",
            local_path="/path/to/model",
            format="gguf",
            size_bytes=12345,
            quantization="Q4_K_M",
            architecture="llama",
            parameters="7B",
            context_length=4096,
        )

        d = original.to_dict()
        restored = ModelManifest.from_dict(d)

        assert restored.name == original.name
        assert restored.repo_id == original.repo_id
        assert restored.size_bytes == original.size_bytes
        assert restored.quantization == original.quantization
        assert restored.architecture == original.architecture


class TestModelRegistry:
    """Tests para ModelRegistry."""

    def test_empty_registry(self, temp_config):
        """Registry vacío inicial."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()

        assert len(registry.list_all()) == 0

    def test_add_model(self, temp_config, sample_manifest):
        """Añadir modelo al registry."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        registry.add(sample_manifest)

        assert len(registry.list_all()) == 1

    def test_get_by_name(self, temp_config, sample_manifest):
        """Buscar modelo por nombre."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        registry.add(sample_manifest)

        result = registry.get("test-model-q4_k_m")

        assert result is not None
        assert result.name == "test-model-q4_k_m"

    def test_get_by_repo_id(self, temp_config, sample_manifest):
        """Buscar modelo por repo_id."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        registry.add(sample_manifest)

        result = registry.get("test-org/test-model")

        assert result is not None
        assert result.repo_id == "test-org/test-model"

    def test_get_not_found(self, temp_config):
        """Buscar modelo inexistente."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_remove_model(self, temp_config, sample_manifest):
        """Eliminar modelo del registry."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        registry.add(sample_manifest)

        result = registry.remove("test-model-q4_k_m")

        assert result is True
        assert registry.get("test-model-q4_k_m") is None
        assert len(registry.list_all()) == 0

    def test_remove_not_found(self, temp_config):
        """Eliminar modelo inexistente."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()

        result = registry.remove("nonexistent")

        assert result is False

    def test_list_all_sorted_by_date(self, temp_config):
        """Lista ordenada por fecha (más reciente primero)."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest
        import time

        registry = ModelRegistry()

        # Añadir modelos con diferentes fechas
        manifest1 = ModelManifest(
            name="model-1",
            repo_id="org/model-1",
            local_path="/path/1",
            format="gguf",
        )
        time.sleep(0.01)  # Pequeña pausa para diferente timestamp

        manifest2 = ModelManifest(
            name="model-2",
            repo_id="org/model-2",
            local_path="/path/2",
            format="gguf",
        )

        registry.add(manifest1)
        registry.add(manifest2)

        models = registry.list_all()

        # El más reciente primero
        assert models[0].name == "model-2"
        assert models[1].name == "model-1"

    def test_avoid_duplicates(self, temp_config):
        """Evitar duplicados (mismo nombre)."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        registry = ModelRegistry()

        manifest1 = ModelManifest(
            name="same-name",
            repo_id="org/model-1",
            local_path="/path/1",
            format="gguf",
        )

        manifest2 = ModelManifest(
            name="same-name",
            repo_id="org/model-2",
            local_path="/path/2",
            format="gguf",
        )

        registry.add(manifest1)
        registry.add(manifest2)

        assert len(registry.list_all()) == 1
        # El segundo reemplaza al primero
        assert registry.get("same-name").local_path == "/path/2"

    def test_persistence_save(self, temp_config, sample_manifest):
        """Verifica persistencia al guardar."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        registry.add(sample_manifest)

        # Verificar archivo
        content = temp_config.registry_path.read_text()
        data = json.loads(content)

        assert len(data) == 1
        assert data[0]["name"] == "test-model-q4_k_m"

    def test_persistence_load(self, temp_config, sample_manifest):
        """Verifica persistencia al cargar."""
        from hfl.models.registry import ModelRegistry

        # Crear registry y guardar
        registry1 = ModelRegistry()
        registry1.add(sample_manifest)

        # Nueva instancia carga datos
        registry2 = ModelRegistry()

        assert len(registry2.list_all()) == 1
        assert registry2.get("test-model-q4_k_m") is not None

    def test_load_corrupted_json(self, temp_config):
        """Manejo de JSON corrupto."""
        from hfl.models.registry import ModelRegistry

        # Escribir JSON inválido
        temp_config.registry_path.write_text("{ invalid json")

        # No debería fallar
        registry = ModelRegistry()

        assert len(registry.list_all()) == 0

    def test_load_missing_file(self, temp_dir):
        """Manejo de archivo faltante."""
        from hfl.config import HFLConfig
        from hfl.models.registry import ModelRegistry
        import hfl.config

        # Config sin ensure_dirs
        test_config = HFLConfig(home_dir=temp_dir)
        test_config.models_dir.mkdir(parents=True, exist_ok=True)
        test_config.cache_dir.mkdir(parents=True, exist_ok=True)
        # NO crear registry_path

        # Temporarily patch
        original_config = hfl.config.config
        hfl.config.config = test_config

        try:
            registry = ModelRegistry()
            assert len(registry.list_all()) == 0
        finally:
            hfl.config.config = original_config

    def test_multiple_models(self, temp_config):
        """Múltiples modelos en registry."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        registry = ModelRegistry()

        for i in range(5):
            manifest = ModelManifest(
                name=f"model-{i}",
                repo_id=f"org/model-{i}",
                local_path=f"/path/{i}",
                format="gguf",
            )
            registry.add(manifest)

        assert len(registry.list_all()) == 5

        # Todos accesibles
        for i in range(5):
            assert registry.get(f"model-{i}") is not None

    def test_concurrent_modifications(self, temp_config):
        """Modificaciones desde múltiples instancias."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        # Primera instancia añade modelo
        registry1 = ModelRegistry()
        manifest1 = ModelManifest(
            name="model-1",
            repo_id="org/model-1",
            local_path="/path/1",
            format="gguf",
        )
        registry1.add(manifest1)

        # Segunda instancia (nuevo load) añade otro
        registry2 = ModelRegistry()
        manifest2 = ModelManifest(
            name="model-2",
            repo_id="org/model-2",
            local_path="/path/2",
            format="gguf",
        )
        registry2.add(manifest2)

        # La segunda instancia debería tener ambos
        assert len(registry2.list_all()) == 2


class TestModelAlias:
    """Tests para funcionalidad de alias de modelos."""

    def test_manifest_with_alias(self):
        """Creación de manifest con alias."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="very-long-model-name-q4_k_m",
            repo_id="org/very-long-model-name",
            local_path="/path/to/model",
            format="gguf",
            alias="short",
        )

        assert manifest.alias == "short"

    def test_manifest_alias_default_none(self):
        """Alias es None por defecto."""
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="test-model",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
        )

        assert manifest.alias is None

    def test_get_by_alias(self, temp_config):
        """Buscar modelo por alias."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        registry = ModelRegistry()
        manifest = ModelManifest(
            name="very-long-model-name-q4_k_m",
            repo_id="org/model",
            local_path="/path/to/model",
            format="gguf",
            alias="coder",
        )
        registry.add(manifest)

        result = registry.get("coder")

        assert result is not None
        assert result.name == "very-long-model-name-q4_k_m"
        assert result.alias == "coder"

    def test_get_by_name_still_works_with_alias(self, temp_config):
        """Buscar por nombre sigue funcionando cuando hay alias."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        registry = ModelRegistry()
        manifest = ModelManifest(
            name="model-name",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
            alias="short",
        )
        registry.add(manifest)

        # Buscar por nombre completo
        result = registry.get("model-name")
        assert result is not None
        assert result.alias == "short"

        # Buscar por alias
        result2 = registry.get("short")
        assert result2 is not None
        assert result2.name == "model-name"

    def test_set_alias_on_existing_model(self, temp_config):
        """Asignar alias a modelo existente."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        registry = ModelRegistry()
        manifest = ModelManifest(
            name="model-without-alias",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
        )
        registry.add(manifest)

        # Asignar alias
        result = registry.set_alias("model-without-alias", "myalias")
        assert result is True

        # Verificar que funciona
        model = registry.get("myalias")
        assert model is not None
        assert model.alias == "myalias"

    def test_set_alias_fails_if_in_use(self, temp_config):
        """No permite alias duplicados."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        registry = ModelRegistry()

        manifest1 = ModelManifest(
            name="model-1",
            repo_id="org/model-1",
            local_path="/path/1",
            format="gguf",
            alias="taken",
        )
        manifest2 = ModelManifest(
            name="model-2",
            repo_id="org/model-2",
            local_path="/path/2",
            format="gguf",
        )
        registry.add(manifest1)
        registry.add(manifest2)

        # Intentar usar alias ya tomado
        result = registry.set_alias("model-2", "taken")
        assert result is False

        # model-2 no debería tener alias
        model = registry.get("model-2")
        assert model.alias is None

    def test_set_alias_fails_if_name_conflict(self, temp_config):
        """No permite alias que coincida con nombre de otro modelo."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        registry = ModelRegistry()

        manifest1 = ModelManifest(
            name="model-1",
            repo_id="org/model-1",
            local_path="/path/1",
            format="gguf",
        )
        manifest2 = ModelManifest(
            name="model-2",
            repo_id="org/model-2",
            local_path="/path/2",
            format="gguf",
        )
        registry.add(manifest1)
        registry.add(manifest2)

        # Intentar usar nombre de otro modelo como alias
        result = registry.set_alias("model-2", "model-1")
        assert result is False

    def test_set_alias_model_not_found(self, temp_config):
        """set_alias falla si el modelo no existe."""
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()

        result = registry.set_alias("nonexistent", "alias")
        assert result is False

    def test_alias_persistence(self, temp_config):
        """Alias se persiste correctamente."""
        from hfl.models.registry import ModelRegistry
        from hfl.models.manifest import ModelManifest

        # Crear y guardar
        registry1 = ModelRegistry()
        manifest = ModelManifest(
            name="model",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
            alias="persistent",
        )
        registry1.add(manifest)

        # Cargar en nueva instancia
        registry2 = ModelRegistry()
        model = registry2.get("persistent")

        assert model is not None
        assert model.alias == "persistent"

    def test_alias_roundtrip(self):
        """Alias sobrevive to_dict/from_dict."""
        from hfl.models.manifest import ModelManifest

        original = ModelManifest(
            name="model",
            repo_id="org/model",
            local_path="/path",
            format="gguf",
            alias="myalias",
        )

        d = original.to_dict()
        restored = ModelManifest.from_dict(d)

        assert restored.alias == "myalias"

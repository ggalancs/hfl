# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for registry backend implementations."""

import pytest

from hfl.models.backends import FileBackend, RegistryBackend, SQLiteBackend
from hfl.models.manifest import ModelManifest


@pytest.fixture
def sample_manifest():
    """Create a sample model manifest."""
    return ModelManifest(
        name="test-model",
        repo_id="org/test-model",
        format="gguf",
        size_bytes=1000,
        local_path="/tmp/test-model",
    )


@pytest.fixture
def another_manifest():
    """Create another sample model manifest."""
    return ModelManifest(
        name="another-model",
        repo_id="org/another-model",
        format="safetensors",
        size_bytes=2000,
        local_path="/tmp/another-model",
    )


class TestFileBackend:
    """Tests for FileBackend."""

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a file backend for testing."""
        path = tmp_path / "models.json"
        return FileBackend(path)

    def test_load_empty(self, backend):
        """Should return empty list when file doesn't exist."""
        result = backend.load()
        assert result == []

    def test_save_and_load(self, backend, sample_manifest):
        """Should persist and retrieve models."""
        backend.save([sample_manifest])

        result = backend.load()

        assert len(result) == 1
        assert result[0].name == "test-model"
        assert result[0].repo_id == "org/test-model"

    def test_add(self, backend, sample_manifest):
        """Should add a model."""
        backend.add(sample_manifest)

        result = backend.load()
        assert len(result) == 1
        assert result[0].name == "test-model"

    def test_add_replaces_existing(self, backend, sample_manifest):
        """Should replace model with same name."""
        backend.add(sample_manifest)

        updated = ModelManifest(
            name="test-model",
            repo_id="org/updated-model",
            format="gguf",
            size_bytes=2000,
            local_path="/tmp/updated",
        )
        backend.add(updated)

        result = backend.load()
        assert len(result) == 1
        assert result[0].repo_id == "org/updated-model"

    def test_remove(self, backend, sample_manifest):
        """Should remove a model by name."""
        backend.add(sample_manifest)

        result = backend.remove("test-model")

        assert result is True
        assert backend.load() == []

    def test_remove_nonexistent(self, backend):
        """Should return False when removing nonexistent model."""
        result = backend.remove("nonexistent")
        assert result is False

    def test_get(self, backend, sample_manifest):
        """Should retrieve a model by name."""
        backend.add(sample_manifest)

        result = backend.get("test-model")

        assert result is not None
        assert result.name == "test-model"

    def test_get_nonexistent(self, backend):
        """Should return None for nonexistent model."""
        result = backend.get("nonexistent")
        assert result is None

    def test_update_alias(self, backend, sample_manifest):
        """Should update model alias."""
        backend.add(sample_manifest)

        result = backend.update_alias("test-model", "my-alias")

        assert result is True
        model = backend.get("test-model")
        assert model.alias == "my-alias"

    def test_update_alias_nonexistent(self, backend):
        """Should return False for nonexistent model."""
        result = backend.update_alias("nonexistent", "alias")
        assert result is False

    def test_update_alias_duplicate(self, backend, sample_manifest, another_manifest):
        """Should reject duplicate alias."""
        sample_manifest.alias = "existing-alias"
        backend.add(sample_manifest)
        backend.add(another_manifest)

        result = backend.update_alias("another-model", "existing-alias")

        assert result is False

    def test_close(self, backend):
        """Close should not raise."""
        backend.close()  # Should not raise


class TestSQLiteBackend:
    """Tests for SQLiteBackend."""

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a SQLite backend for testing."""
        db_path = tmp_path / "models.db"
        backend = SQLiteBackend(db_path)
        yield backend
        backend.close()

    def test_load_empty(self, backend):
        """Should return empty list for fresh database."""
        result = backend.load()
        assert result == []

    def test_save_and_load(self, backend, sample_manifest):
        """Should persist and retrieve models."""
        backend.save([sample_manifest])

        result = backend.load()

        assert len(result) == 1
        assert result[0].name == "test-model"

    def test_add(self, backend, sample_manifest):
        """Should add a model."""
        backend.add(sample_manifest)

        result = backend.load()
        assert len(result) == 1
        assert result[0].name == "test-model"

    def test_add_replaces_existing(self, backend, sample_manifest):
        """Should replace model with same name using INSERT OR REPLACE."""
        backend.add(sample_manifest)

        updated = ModelManifest(
            name="test-model",
            repo_id="org/updated-model",
            format="gguf",
            size_bytes=2000,
            local_path="/tmp/updated",
        )
        backend.add(updated)

        result = backend.load()
        assert len(result) == 1
        assert result[0].repo_id == "org/updated-model"

    def test_remove(self, backend, sample_manifest):
        """Should remove a model by name."""
        backend.add(sample_manifest)

        result = backend.remove("test-model")

        assert result is True
        assert backend.load() == []

    def test_remove_nonexistent(self, backend):
        """Should return False when removing nonexistent model."""
        result = backend.remove("nonexistent")
        assert result is False

    def test_get(self, backend, sample_manifest):
        """Should retrieve a model by name."""
        backend.add(sample_manifest)

        result = backend.get("test-model")

        assert result is not None
        assert result.name == "test-model"

    def test_get_nonexistent(self, backend):
        """Should return None for nonexistent model."""
        result = backend.get("nonexistent")
        assert result is None

    def test_get_by_alias(self, backend, sample_manifest):
        """Should retrieve model by alias."""
        sample_manifest.alias = "my-alias"
        backend.add(sample_manifest)

        result = backend.get_by_alias("my-alias")

        assert result is not None
        assert result.name == "test-model"

    def test_update_alias(self, backend, sample_manifest):
        """Should update model alias."""
        backend.add(sample_manifest)

        result = backend.update_alias("test-model", "my-alias")

        assert result is True
        model = backend.get("test-model")
        assert model.alias == "my-alias"

    def test_update_alias_nonexistent(self, backend):
        """Should return False for nonexistent model."""
        result = backend.update_alias("nonexistent", "alias")
        assert result is False

    def test_update_alias_duplicate(self, backend, sample_manifest, another_manifest):
        """Should reject duplicate alias."""
        sample_manifest.alias = "existing-alias"
        backend.add(sample_manifest)
        backend.add(another_manifest)

        result = backend.update_alias("another-model", "existing-alias")

        assert result is False

    def test_multiple_models(self, backend, sample_manifest, another_manifest):
        """Should handle multiple models."""
        backend.add(sample_manifest)
        backend.add(another_manifest)

        result = backend.load()

        assert len(result) == 2
        names = {m.name for m in result}
        assert names == {"test-model", "another-model"}


class TestBackendInterface:
    """Tests to verify both backends implement the same interface."""

    @pytest.fixture(params=["file", "sqlite"])
    def backend(self, request, tmp_path) -> RegistryBackend:
        """Create either file or sqlite backend."""
        if request.param == "file":
            return FileBackend(tmp_path / "models.json")
        else:
            backend = SQLiteBackend(tmp_path / "models.db")
            request.addfinalizer(backend.close)
            return backend

    def test_interface_load(self, backend):
        """Both backends should support load()."""
        result = backend.load()
        assert isinstance(result, list)

    def test_interface_add_get_remove(self, backend, sample_manifest):
        """Both backends should support add/get/remove cycle."""
        backend.add(sample_manifest)
        assert backend.get("test-model") is not None
        assert backend.remove("test-model") is True
        assert backend.get("test-model") is None

    def test_interface_alias(self, backend, sample_manifest):
        """Both backends should support alias operations."""
        backend.add(sample_manifest)
        assert backend.update_alias("test-model", "alias") is True
        model = backend.get("test-model")
        assert model.alias == "alias"

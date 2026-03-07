# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for registry corruption recovery."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hfl.models.manifest import ModelManifest
from hfl.models.registry import ModelRegistry


@pytest.fixture
def temp_registry_dir(tmp_path):
    """Create a temporary directory for registry tests."""
    return tmp_path


@pytest.fixture
def mock_config(temp_registry_dir):
    """Mock config to use temp directory."""
    mock_cfg = MagicMock()
    mock_cfg.registry_path = temp_registry_dir / "models.json"
    return mock_cfg


def create_manifest(name: str, path: str = "/tmp/model") -> dict:
    """Create a valid manifest dict."""
    return {
        "name": name,
        "repo_id": f"test/{name}",
        "local_path": path,
        "format": "gguf",
        "size_bytes": 1000,
        "created_at": "2024-01-01T00:00:00Z",
        "license_type": "permissive",
    }


class TestRegistryLoad:
    """Tests for registry loading with corruption handling."""

    def test_load_valid_file(self, mock_config):
        """Should load valid registry file."""
        mock_config.registry_path.write_text(
            json.dumps([create_manifest("model1")])
        )

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        assert len(registry) == 1
        assert "model1" in registry

    def test_load_missing_file(self, mock_config):
        """Should handle missing file gracefully."""
        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        assert len(registry) == 0

    def test_load_corrupt_json(self, mock_config):
        """Should handle corrupt JSON gracefully."""
        mock_config.registry_path.write_text("{ invalid json }")

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        assert len(registry) == 0

    def test_load_corrupt_recovers_from_backup(self, mock_config):
        """Should recover from backup when main file is corrupt."""
        # Create backup with valid data
        backup_path = mock_config.registry_path.with_suffix(".json.bak")
        backup_path.write_text(json.dumps([create_manifest("recovered_model")]))

        # Create corrupt main file
        mock_config.registry_path.write_text("corrupted!")

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        assert len(registry) == 1
        assert "recovered_model" in registry

    def test_load_skips_invalid_entries(self, mock_config):
        """Should skip invalid manifest entries during load."""
        data = [
            create_manifest("valid"),
            {"invalid": "entry"},  # Missing required fields
            create_manifest("also_valid"),
        ]
        mock_config.registry_path.write_text(json.dumps(data))

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        # Should have 2 valid models, skipped 1 invalid
        assert len(registry) == 2
        assert "valid" in registry
        assert "also_valid" in registry


class TestRegistrySave:
    """Tests for registry saving with backup."""

    def test_save_creates_backup(self, mock_config):
        """Save should create backup of existing file."""
        mock_config.registry_path.write_text(
            json.dumps([create_manifest("original")])
        )

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()
            manifest = ModelManifest(
                name="new_model",
                repo_id="test/new",
                local_path="/tmp/model",
                format="gguf",
                size_bytes=1000,
            )
            registry.add(manifest)

        backup_path = mock_config.registry_path.with_suffix(".json.bak")
        assert backup_path.exists()

        # Backup should contain original data
        backup_data = json.loads(backup_path.read_text())
        assert len(backup_data) == 1
        assert backup_data[0]["name"] == "original"

    def test_atomic_save_prevents_corruption(self, mock_config):
        """Save should use atomic write."""
        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()
            manifest = ModelManifest(
                name="test_model",
                repo_id="test/model",
                local_path="/tmp/model",
                format="gguf",
                size_bytes=1000,
            )
            registry.add(manifest)

        # Main file should exist and be valid
        assert mock_config.registry_path.exists()
        data = json.loads(mock_config.registry_path.read_text())
        assert len(data) == 1

        # Temp file should not exist
        temp_path = mock_config.registry_path.with_suffix(".json.tmp")
        assert not temp_path.exists()


class TestValidateIntegrity:
    """Tests for registry integrity validation."""

    def test_validate_valid_registry(self, mock_config, tmp_path):
        """Should return True for valid registry."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        mock_config.registry_path.write_text(
            json.dumps([create_manifest("model1", str(model_path))])
        )

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()
            is_valid, errors = registry.validate_integrity()

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_duplicate_names(self, mock_config):
        """Should detect duplicate model names."""
        data = [
            create_manifest("dupe"),
            create_manifest("dupe"),
        ]
        mock_config.registry_path.write_text(json.dumps(data))

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()
            # Force duplicate by modifying internal list
            registry._models.append(registry._models[0])
            is_valid, errors = registry.validate_integrity()

        assert is_valid is False
        assert any("Duplicate model names" in e for e in errors)

    def test_validate_missing_path(self, mock_config):
        """Should detect missing local paths."""
        mock_config.registry_path.write_text(
            json.dumps([create_manifest("model1", "/nonexistent/path")])
        )

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()
            is_valid, errors = registry.validate_integrity()

        assert is_valid is False
        assert any("does not exist" in e for e in errors)


class TestRepair:
    """Tests for registry repair functionality."""

    def test_repair_removes_invalid_paths(self, mock_config, tmp_path):
        """Repair should remove models with non-existent paths."""
        valid_path = tmp_path / "valid.gguf"
        valid_path.touch()

        data = [
            create_manifest("valid", str(valid_path)),
            create_manifest("invalid", "/nonexistent/path"),
        ]
        mock_config.registry_path.write_text(json.dumps(data))

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()
            removed = registry.repair()

        assert removed == 1
        assert len(registry) == 1
        assert "valid" in registry
        assert "invalid" not in registry

    def test_repair_removes_duplicates(self, mock_config, tmp_path):
        """Repair should remove duplicate entries."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()
            manifest = ModelManifest(
                name="dupe",
                repo_id="test/dupe",
                local_path=str(model_path),
                format="gguf",
                size_bytes=1000,
            )
            # Add same model twice by manipulating internal list
            registry._models = [manifest, manifest]
            registry._rebuild_indexes()

            removed = registry.repair()

        assert removed == 1
        assert len(registry) == 1

    def test_repair_empty_registry(self, mock_config):
        """Repair on empty registry should work."""
        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()
            removed = registry.repair()

        assert removed == 0
        assert len(registry) == 0


class TestCorruptionEvent:
    """Tests for corruption event emission."""

    @patch("hfl.events.emit")
    def test_emits_event_on_corruption(self, mock_emit, mock_config):
        """Should emit event when corruption is detected."""
        mock_config.registry_path.write_text("not valid json")

        with patch("hfl.models.registry.config", mock_config):
            ModelRegistry()

        mock_emit.assert_called()
        # Check that ERROR event was emitted
        calls = [str(c) for c in mock_emit.call_args_list]
        assert any("registry_corruption" in c for c in calls)

    @patch("hfl.events.emit")
    def test_emits_event_on_recovery(self, mock_emit, mock_config):
        """Should emit event when recovery succeeds."""
        # Create backup
        backup_path = mock_config.registry_path.with_suffix(".json.bak")
        backup_path.write_text(json.dumps([create_manifest("recovered")]))

        # Create corrupt main file
        mock_config.registry_path.write_text("corrupt")

        with patch("hfl.models.registry.config", mock_config):
            ModelRegistry()

        mock_emit.assert_called()
        calls = [str(c) for c in mock_emit.call_args_list]
        assert any("recovered" in c for c in calls)


class TestBackupPath:
    """Tests for backup path property."""

    def test_backup_path_suffix(self, mock_config):
        """Backup path should have .json.bak suffix."""
        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        assert registry._backup_path.suffix == ".bak"
        assert ".json" in str(registry._backup_path)


class TestParseManifests:
    """Tests for manifest parsing."""

    def test_parse_valid_list(self, mock_config):
        """Should parse valid manifest list."""
        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        data = [create_manifest("m1"), create_manifest("m2")]
        result = registry._parse_manifests(data)

        assert len(result) == 2

    def test_parse_not_list_raises(self, mock_config):
        """Should raise for non-list data."""
        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        with pytest.raises(ValueError, match="must be a list"):
            registry._parse_manifests({"not": "a list"})

    def test_parse_skips_invalid(self, mock_config):
        """Should skip invalid entries."""
        with patch("hfl.models.registry.config", mock_config):
            registry = ModelRegistry()

        data = [
            create_manifest("valid"),
            {"missing": "fields"},
            create_manifest("also_valid"),
        ]
        result = registry._parse_manifests(data)

        assert len(result) == 2

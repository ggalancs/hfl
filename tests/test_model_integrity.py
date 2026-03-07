# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for model integrity verification."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from hfl.models.manifest import ModelManifest


@pytest.fixture
def temp_model_file(tmp_path):
    """Create a temporary model file."""
    model_path = tmp_path / "test_model.gguf"
    model_path.write_bytes(b"test model content for hashing")
    return model_path


@pytest.fixture
def temp_model_hash(temp_model_file):
    """Compute the expected hash of the test model."""
    hash_obj = hashlib.sha256()
    hash_obj.update(temp_model_file.read_bytes())
    return hash_obj.hexdigest()


@pytest.fixture
def manifest_with_file(temp_model_file, temp_model_hash):
    """Create a manifest with a real file."""
    return ModelManifest(
        name="test-model",
        repo_id="test/model",
        local_path=str(temp_model_file),
        format="gguf",
        size_bytes=temp_model_file.stat().st_size,
        file_hash=temp_model_hash,
    )


class TestModelManifestPath:
    """Tests for path property."""

    def test_path_returns_path_object(self, temp_model_file):
        """path property should return Path object."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
        )
        assert isinstance(manifest.path, Path)
        assert manifest.path == temp_model_file


class TestFileExists:
    """Tests for file_exists method."""

    def test_file_exists_true(self, temp_model_file):
        """Should return True when file exists."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
        )
        assert manifest.file_exists() is True

    def test_file_exists_false(self):
        """Should return False when file doesn't exist."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path="/nonexistent/path/model.gguf",
            format="gguf",
        )
        assert manifest.file_exists() is False


class TestComputeHash:
    """Tests for compute_hash method."""

    def test_compute_hash_success(self, temp_model_file, temp_model_hash):
        """Should compute correct hash for existing file."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
        )
        computed = manifest.compute_hash()
        assert computed == temp_model_hash

    def test_compute_hash_missing_file(self):
        """Should return None for missing file."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path="/nonexistent/model.gguf",
            format="gguf",
        )
        assert manifest.compute_hash() is None

    def test_compute_hash_custom_algorithm(self, temp_model_file):
        """Should use custom hash algorithm."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
            hash_algorithm="md5",
        )

        computed = manifest.compute_hash()

        # Verify with direct computation
        expected = hashlib.md5(temp_model_file.read_bytes()).hexdigest()
        assert computed == expected


class TestVerifyIntegrity:
    """Tests for verify_integrity method."""

    def test_verify_success(self, manifest_with_file):
        """Should verify successfully when file and hash match."""
        is_valid, message = manifest_with_file.verify_integrity()
        assert is_valid is True
        assert "verified" in message.lower()
        assert manifest_with_file.verified_at is not None

    def test_verify_missing_file(self):
        """Should fail verification for missing file."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path="/nonexistent/model.gguf",
            format="gguf",
        )
        is_valid, message = manifest.verify_integrity()
        assert is_valid is False
        assert "not found" in message.lower()

    def test_verify_size_mismatch(self, temp_model_file, temp_model_hash):
        """Should fail verification for size mismatch."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
            size_bytes=9999999,  # Wrong size
            file_hash=temp_model_hash,
        )
        is_valid, message = manifest.verify_integrity()
        assert is_valid is False
        assert "size mismatch" in message.lower()

    def test_verify_hash_mismatch(self, temp_model_file):
        """Should fail verification for hash mismatch."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
            size_bytes=temp_model_file.stat().st_size,
            file_hash="0" * 64,  # Wrong hash
        )
        is_valid, message = manifest.verify_integrity()
        assert is_valid is False
        assert "hash mismatch" in message.lower()

    def test_verify_no_hash_stored(self, temp_model_file):
        """Should pass if no hash stored but file exists."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
            size_bytes=temp_model_file.stat().st_size,
            file_hash=None,  # No hash
        )
        is_valid, message = manifest.verify_integrity()
        assert is_valid is True
        assert "no hash to verify" in message.lower()

    def test_verify_zero_size_skips_check(self, temp_model_file, temp_model_hash):
        """Should skip size check when size_bytes is 0."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
            size_bytes=0,  # Size not set
            file_hash=temp_model_hash,
        )
        is_valid, message = manifest.verify_integrity()
        assert is_valid is True


class TestUpdateHash:
    """Tests for update_hash method."""

    def test_update_hash_success(self, temp_model_file, temp_model_hash):
        """Should update hash successfully."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
        )
        assert manifest.file_hash is None

        result = manifest.update_hash()

        assert result is True
        assert manifest.file_hash == temp_model_hash
        assert manifest.verified_at is not None

    def test_update_hash_missing_file(self):
        """Should return False for missing file."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path="/nonexistent/model.gguf",
            format="gguf",
        )
        result = manifest.update_hash()
        assert result is False
        assert manifest.file_hash is None


class TestManifestSerialization:
    """Tests for manifest serialization with integrity fields."""

    def test_to_dict_includes_integrity_fields(self, manifest_with_file):
        """to_dict should include integrity fields."""
        data = manifest_with_file.to_dict()

        assert "file_hash" in data
        assert "hash_algorithm" in data
        assert "verified_at" in data
        assert data["hash_algorithm"] == "sha256"

    def test_from_dict_restores_integrity_fields(self, temp_model_file, temp_model_hash):
        """from_dict should restore integrity fields."""
        data = {
            "name": "test",
            "repo_id": "test/repo",
            "local_path": str(temp_model_file),
            "format": "gguf",
            "file_hash": temp_model_hash,
            "hash_algorithm": "sha256",
            "verified_at": "2024-01-01T00:00:00",
        }

        manifest = ModelManifest.from_dict(data)

        assert manifest.file_hash == temp_model_hash
        assert manifest.hash_algorithm == "sha256"
        assert manifest.verified_at == "2024-01-01T00:00:00"

    def test_from_dict_defaults_integrity_fields(self):
        """from_dict should use defaults for missing integrity fields."""
        data = {
            "name": "test",
            "repo_id": "test/repo",
            "local_path": "/some/path",
            "format": "gguf",
        }

        manifest = ModelManifest.from_dict(data)

        assert manifest.file_hash is None
        assert manifest.hash_algorithm == "sha256"
        assert manifest.verified_at is None


class TestIntegrityEdgeCases:
    """Edge case tests for integrity verification."""

    def test_large_file_hash(self, tmp_path):
        """Should handle larger files correctly."""
        model_path = tmp_path / "large_model.gguf"
        # Create a ~1MB file
        content = b"x" * (1024 * 1024)
        model_path.write_bytes(content)

        expected_hash = hashlib.sha256(content).hexdigest()

        manifest = ModelManifest(
            name="large",
            repo_id="test/large",
            local_path=str(model_path),
            format="gguf",
            size_bytes=len(content),
            file_hash=expected_hash,
        )

        is_valid, message = manifest.verify_integrity()
        assert is_valid is True

    def test_empty_file(self, tmp_path):
        """Should handle empty files."""
        model_path = tmp_path / "empty.gguf"
        model_path.write_bytes(b"")

        expected_hash = hashlib.sha256(b"").hexdigest()

        manifest = ModelManifest(
            name="empty",
            repo_id="test/empty",
            local_path=str(model_path),
            format="gguf",
            size_bytes=0,
            file_hash=expected_hash,
        )

        is_valid, message = manifest.verify_integrity()
        assert is_valid is True

    def test_case_insensitive_hash_comparison(self, temp_model_file, temp_model_hash):
        """Hash comparison should be case-insensitive."""
        manifest = ModelManifest(
            name="test",
            repo_id="test/repo",
            local_path=str(temp_model_file),
            format="gguf",
            size_bytes=temp_model_file.stat().st_size,
            file_hash=temp_model_hash.upper(),  # Uppercase hash
        )

        is_valid, message = manifest.verify_integrity()
        assert is_valid is True

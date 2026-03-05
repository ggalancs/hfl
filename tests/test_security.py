# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for security utilities."""

import pytest

from hfl.security import (
    PathTraversalError,
    compute_file_hash,
    sanitize_model_name,
    sanitize_path,
    verify_file_hash,
)


class TestSanitizePath:
    """Tests for sanitize_path function."""

    def test_simple_relative_path(self, temp_dir):
        """Test simple relative path stays within base."""
        result = sanitize_path(temp_dir, "models/test.gguf")
        # Compare resolved paths to handle symlinks (e.g., /var -> /private/var on macOS)
        expected = (temp_dir / "models" / "test.gguf").resolve()
        assert result == expected

    def test_absolute_path_within_base(self, temp_dir):
        """Test absolute path within base is allowed."""
        target = temp_dir / "models" / "test.gguf"
        result = sanitize_path(temp_dir, str(target))
        assert result == target.resolve()

    def test_path_traversal_attack(self, temp_dir):
        """Test path traversal is blocked."""
        with pytest.raises(PathTraversalError):
            sanitize_path(temp_dir, "../../../etc/passwd")

    def test_hidden_path_traversal(self, temp_dir):
        """Test hidden path traversal via encoding."""
        with pytest.raises(PathTraversalError):
            sanitize_path(temp_dir, "models/../../../etc/passwd")

    def test_absolute_path_outside_base(self, temp_dir):
        """Test absolute path outside base is blocked."""
        with pytest.raises(PathTraversalError):
            sanitize_path(temp_dir, "/etc/passwd")

    def test_empty_path(self, temp_dir):
        """Test empty path resolves to base."""
        result = sanitize_path(temp_dir, "")
        # Compare resolved paths
        assert result == temp_dir.resolve()

    def test_dot_path(self, temp_dir):
        """Test . path resolves to base."""
        result = sanitize_path(temp_dir, ".")
        assert result == temp_dir.resolve()


class TestSanitizeModelName:
    """Tests for sanitize_model_name function."""

    def test_simple_name(self):
        """Test simple model name passes through."""
        assert sanitize_model_name("llama-7b") == "llama-7b"

    def test_org_slash_model(self):
        """Test org/model format is sanitized."""
        assert sanitize_model_name("meta-llama/Llama-3") == "meta-llama--Llama-3"

    def test_path_traversal_blocked(self):
        """Test path traversal in name is blocked."""
        # .. becomes __ and / becomes --
        assert sanitize_model_name("../model") == "__--model"

    def test_backslash_blocked(self):
        """Test backslash is sanitized."""
        assert sanitize_model_name("model\\test") == "model--test"

    def test_empty_name_raises(self):
        """Test empty name raises ValueError."""
        with pytest.raises(ValueError):
            sanitize_model_name("")

    def test_only_dots_becomes_underscores(self):
        """Test name with only dots becomes underscores (after .. replacement)."""
        # .... becomes ____ (two .. patterns)
        result = sanitize_model_name("....")
        assert result == "____"

    def test_whitespace_stripped(self):
        """Test whitespace is stripped."""
        assert sanitize_model_name("  model  ") == "model"


class TestChecksumFunctions:
    """Tests for checksum functions."""

    def test_compute_sha256(self, temp_dir):
        """Test SHA256 computation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("hello world")

        hash_value = compute_file_hash(test_file)
        # Known SHA256 of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert hash_value == expected

    def test_compute_md5(self, temp_dir):
        """Test MD5 computation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("hello world")

        hash_value = compute_file_hash(test_file, algorithm="md5")
        # Known MD5 of "hello world"
        expected = "5eb63bbbe01eeed093cb22bb8f5acdc3"
        assert hash_value == expected

    def test_verify_correct_hash(self, temp_dir):
        """Test verification with correct hash."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("hello world")

        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert verify_file_hash(test_file, expected) is True

    def test_verify_incorrect_hash(self, temp_dir):
        """Test verification with incorrect hash."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("hello world")

        wrong_hash = "deadbeef" * 8
        assert verify_file_hash(test_file, wrong_hash) is False

    def test_verify_case_insensitive(self, temp_dir):
        """Test hash verification is case-insensitive."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("hello world")

        expected_upper = "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9"
        assert verify_file_hash(test_file, expected_upper) is True

    def test_large_file_hash(self, temp_dir):
        """Test hashing of larger files."""
        test_file = temp_dir / "large.bin"
        # Create a 1MB file
        test_file.write_bytes(b"x" * (1024 * 1024))

        # Just verify it completes without error
        hash_value = compute_file_hash(test_file)
        assert len(hash_value) == 64  # SHA256 hex length

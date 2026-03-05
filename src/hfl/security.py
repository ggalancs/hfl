# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Security utilities for HFL.

Provides:
- Path sanitization to prevent path traversal attacks
- File checksum validation
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""


def sanitize_path(base_dir: Path, user_path: str) -> Path:
    """
    Sanitize a user-provided path to prevent path traversal attacks.

    Args:
        base_dir: The base directory that paths must stay within
        user_path: User-provided path (may be relative or absolute)

    Returns:
        Sanitized absolute path within base_dir

    Raises:
        PathTraversalError: If path would escape base_dir
    """
    # Normalize the base directory
    base_dir = base_dir.resolve()

    # Handle user path
    if Path(user_path).is_absolute():
        # For absolute paths, ensure they're within base_dir
        target = Path(user_path).resolve()
    else:
        # For relative paths, join with base and resolve
        target = (base_dir / user_path).resolve()

    # Verify the target is within base_dir
    try:
        target.relative_to(base_dir)
    except ValueError:
        raise PathTraversalError(
            f"Path '{user_path}' would escape base directory '{base_dir}'"
        )

    return target


def sanitize_model_name(name: str) -> str:
    """
    Sanitize a model name to prevent path injection.

    Args:
        name: User-provided model name

    Returns:
        Sanitized model name safe for use in paths

    Raises:
        ValueError: If name contains invalid characters
    """
    # Remove any path separators
    sanitized = name.replace("/", "--").replace("\\", "--")

    # Remove any parent directory references
    sanitized = sanitized.replace("..", "__")

    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip().strip(".")

    if not sanitized:
        raise ValueError(f"Invalid model name: '{name}'")

    return sanitized


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute the hash of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hex-encoded hash digest
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def verify_file_hash(
    file_path: Path, expected_hash: str, algorithm: str = "sha256"
) -> bool:
    """
    Verify a file matches an expected hash.

    Args:
        file_path: Path to the file
        expected_hash: Expected hash value (hex-encoded)
        algorithm: Hash algorithm used (default: sha256)

    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = compute_file_hash(file_path, algorithm)
    return actual_hash.lower() == expected_hash.lower()

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Data model for local model metadata."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelManifest:
    """Complete metadata for a downloaded model."""

    # Identification (required fields first)
    name: str  # Short name (e.g., "llama3.3-70b-q4")
    repo_id: str  # HuggingFace repo (e.g., "meta-llama/Llama-3.3-70B")

    # Storage
    local_path: str  # Absolute path to the model
    format: str  # "gguf", "safetensors", "pytorch"

    # Optional fields (with default)
    alias: str | None = None  # User-defined alias (e.g., "coder")
    size_bytes: int = 0  # Size on disk

    # Quantization
    quantization: str | None = None  # Q4_K_M, Q5_K_M, etc.
    original_format: str | None = None  # Original format if converted

    # Model
    architecture: str | None = None  # llama, mistral, gemma, etc.
    parameters: str | None = None  # "7B", "70B", etc.
    context_length: int = 4096  # Context length
    model_type: str | None = None  # "llm", "tts", "stt", etc. (from ModelType enum)

    # Chat template
    chat_template: str | None = None  # Jinja2 template for chat

    # License (R1 - Legal Audit)
    license: str | None = None  # SPDX license identifier
    license_name: str | None = None  # Human-readable full name
    license_url: str | None = None  # URL to full terms
    license_restrictions: list[str] = field(default_factory=list)
    # e.g., ["non-commercial", "no-derivative", "attribution-required"]
    gated: bool = False  # Required acceptance of terms
    license_accepted_at: str | None = None  # Acceptance timestamp

    # EU AI Act (R4 - Legal Audit)
    gpai_classification: str | None = None  # "gpai", "gpai-systemic", "exempt"
    training_flops: str | None = None  # If available in model card

    # Integrity verification
    file_hash: str | None = None  # SHA-256 hash of the model file
    hash_algorithm: str = "sha256"  # Algorithm used for hash
    verified_at: str | None = None  # Last verification timestamp

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def display_size(self) -> str:
        """Human-readable size (e.g., '4.2 GB')."""
        gb = self.size_bytes / (1024**3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        mb = self.size_bytes / (1024**2)
        return f"{mb:.0f} MB"

    @property
    def path(self) -> Path:
        """Return local_path as Path object."""
        return Path(self.local_path)

    def file_exists(self) -> bool:
        """Check if the model file exists on disk."""
        return self.path.exists()

    def compute_hash(self) -> str | None:
        """Compute the hash of the model file.

        Returns:
            The hex-encoded hash, or None if file doesn't exist.
        """
        if not self.file_exists():
            logger.warning(f"Cannot compute hash: file not found at {self.local_path}")
            return None

        from hfl.security import compute_file_hash

        return compute_file_hash(self.path, self.hash_algorithm)

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify the integrity of the model file.

        Returns:
            Tuple of (is_valid, message).
            is_valid is True if the file exists and hash matches (or no hash stored).
        """
        # Check if file exists
        if not self.file_exists():
            return False, f"Model file not found: {self.local_path}"

        # Check file size
        actual_size = self.path.stat().st_size
        if self.size_bytes > 0 and actual_size != self.size_bytes:
            return False, (
                f"Size mismatch: expected {self.size_bytes} bytes, "
                f"found {actual_size} bytes"
            )

        # If no hash stored, file exists with correct size is good enough
        if not self.file_hash:
            return True, "File exists (no hash to verify)"

        # Compute and compare hash
        actual_hash = self.compute_hash()
        if actual_hash is None:
            return False, "Failed to compute file hash"

        if actual_hash.lower() != self.file_hash.lower():
            return False, (
                f"Hash mismatch: expected {self.file_hash[:16]}..., "
                f"got {actual_hash[:16]}..."
            )

        # Update verification timestamp
        self.verified_at = datetime.now().isoformat()
        return True, "Integrity verified"

    def update_hash(self) -> bool:
        """Compute and store the hash of the model file.

        Returns:
            True if hash was computed successfully, False otherwise.
        """
        computed = self.compute_hash()
        if computed is None:
            return False

        self.file_hash = computed
        self.verified_at = datetime.now().isoformat()
        logger.info(f"Updated hash for {self.name}: {computed[:16]}...")
        return True

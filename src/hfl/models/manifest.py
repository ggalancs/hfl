# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Data model for local model metadata."""

from dataclasses import asdict, dataclass, field
from datetime import datetime


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

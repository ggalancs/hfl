# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Central configuration for hfl."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HFLConfig:
    """Global application configuration."""

    # Root directory (~/.hfl by default)
    home_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("HFL_HOME", Path.home() / ".hfl"))
    )

    # Subdirectories
    @property
    def models_dir(self) -> Path:
        return self.home_dir / "models"

    @property
    def cache_dir(self) -> Path:
        return self.home_dir / "cache"

    @property
    def registry_path(self) -> Path:
        return self.home_dir / "models.json"

    @property
    def llama_cpp_dir(self) -> Path:
        """Directory where llama.cpp is cloned/compiled for conversion."""
        return self.home_dir / "tools" / "llama.cpp"

    # Server
    host: str = "127.0.0.1"
    port: int = 11434  # Same port as Ollama for drop-in compatibility

    # Inference
    default_ctx_size: int = 4096
    default_n_gpu_layers: int = -1  # -1 = all layers to GPU
    default_threads: int = 0  # 0 = auto-detect

    # HuggingFace
    # PRIVACY (R6 - Legal Audit): hf_token is read ONLY from environment variable.
    # It is NEVER persisted to disk, NEVER stored in models.json or any config file.
    # Token is held in memory only for the duration of the process.
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    def ensure_dirs(self):
        """Creates the necessary directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Initialize registry if it doesn't exist
        if not self.registry_path.exists():
            self.registry_path.write_text("[]")


# Global instance
config = HFLConfig()
config.ensure_dirs()

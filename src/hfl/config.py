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

    # Security - CORS
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = field(default_factory=lambda: ["*"])

    # Security - Rate Limiting (requests per minute)
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 60
    rate_limit_window: int = 60  # seconds

    # LLM Inference
    default_ctx_size: int = 4096
    default_n_gpu_layers: int = -1  # -1 = all layers to GPU
    default_threads: int = 0  # 0 = auto-detect

    # TTS defaults
    default_tts_sample_rate: int = 22050
    default_tts_format: str = "wav"  # wav, mp3, ogg

    # Timeouts (seconds)
    model_load_timeout: float = 300.0      # 5 minutes
    generation_timeout: float = 600.0       # 10 minutes
    download_timeout: float = 3600.0        # 1 hour
    conversion_timeout: float = 7200.0      # 2 hours
    api_request_timeout: float = 120.0      # 2 minutes

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0

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

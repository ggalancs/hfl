# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Configuración central de hfl."""

from pathlib import Path
from dataclasses import dataclass, field
import os


@dataclass
class HFLConfig:
    """Configuración global de la aplicación."""

    # Directorio raíz (~/.hfl por defecto)
    home_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("HFL_HOME", Path.home() / ".hfl")
        )
    )

    # Subdirectorios
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
        """Directorio donde se clona/compila llama.cpp para conversión."""
        return self.home_dir / "tools" / "llama.cpp"

    # Servidor
    host: str = "127.0.0.1"
    port: int = 11434  # Mismo puerto que Ollama para drop-in compatibility

    # Inferencia
    default_ctx_size: int = 4096
    default_n_gpu_layers: int = -1  # -1 = todas las capas a GPU
    default_threads: int = 0  # 0 = auto-detectar

    # HuggingFace
    # PRIVACY (R6 - Legal Audit): hf_token is read ONLY from environment variable.
    # It is NEVER persisted to disk, NEVER stored in models.json or any config file.
    # Token is held in memory only for the duration of the process.
    hf_token: str | None = field(
        default_factory=lambda: os.environ.get("HF_TOKEN")
    )

    def ensure_dirs(self):
        """Crea los directorios necesarios."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Inicializar registro si no existe
        if not self.registry_path.exists():
            self.registry_path.write_text("[]")


# Instancia global
config = HFLConfig()
config.ensure_dirs()

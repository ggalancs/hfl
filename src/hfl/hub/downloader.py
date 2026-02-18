# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Descarga de modelos desde HuggingFace Hub con progreso visual.

Soporta:
- Descarga de archivos GGUF individuales
- Descarga de repos completos (safetensors)
- Reanudación de descargas interrumpidas
- Cache inteligente (no re-descarga si ya existe)

Cumplimiento con ToS de HuggingFace (R8 - Auditoría Legal):
- Rate limiting entre llamadas API
- User-Agent identificativo
"""

import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console

from hfl.config import config
from hfl.hub.resolver import ResolvedModel
from hfl.hub.auth import ensure_auth

console = Console()

# Rate limiting para cumplir con ToS de HuggingFace (R8)
_last_api_call: float = 0
_MIN_INTERVAL: float = 0.5  # Mínimo 0.5 segundos entre llamadas API

# Configurar User-Agent para identificar el tool (R8)
# Esto permite a HuggingFace identificar el origen de las solicitudes
try:
    from hfl import __version__
except ImportError:
    __version__ = "0.1.0"

os.environ.setdefault("HF_HUB_USER_AGENT", f"hfl/{__version__}")


def _rate_limit() -> None:
    """Aplica rate limiting entre llamadas API."""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_api_call = time.time()


def pull_model(resolved: ResolvedModel) -> Path:
    """
    Descarga un modelo y devuelve la ruta local.

    Para GGUF: descarga el archivo individual.
    Para safetensors: descarga el snapshot completo del repo.
    """
    # Rate limiting antes de llamadas API (R8 - ToS compliance)
    _rate_limit()
    token = ensure_auth(resolved.repo_id)

    # Directorio destino: ~/.hfl/models/<org>/<model>/
    model_dir = config.models_dir / resolved.repo_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold cyan]Descargando[/] {resolved.repo_id}"
        + (f" ({resolved.filename})" if resolved.filename else "")
    )

    if resolved.format == "gguf" and resolved.filename:
        # Descarga de archivo GGUF individual
        # (resume_download is deprecated - downloads always resume automatically)
        local_path = hf_hub_download(
            repo_id=resolved.repo_id,
            filename=resolved.filename,
            revision=resolved.revision,
            local_dir=model_dir,
            token=token,
        )
        return Path(local_path)
    else:
        # Descarga del snapshot completo
        # Filtrar solo los archivos necesarios
        allow_patterns = []
        if resolved.format == "safetensors":
            allow_patterns = [
                "*.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",       # SentencePiece
                "generation_config.json",
            ]

        # (resume_download is deprecated - downloads always resume automatically)
        local_dir = snapshot_download(
            repo_id=resolved.repo_id,
            revision=resolved.revision,
            local_dir=model_dir,
            token=token,
            allow_patterns=allow_patterns or None,
        )
        return Path(local_dir)

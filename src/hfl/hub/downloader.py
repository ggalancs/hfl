# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Model download from HuggingFace Hub with visual progress.

Supports:
- Individual GGUF file download
- Complete repo download (safetensors)
- Resume of interrupted downloads
- Smart cache (no re-download if already exists)

Compliance with HuggingFace ToS (R8 - Legal Audit):
- Rate limiting between API calls
- Identifying User-Agent
"""

import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console

from hfl.config import config
from hfl.hub.auth import ensure_auth
from hfl.hub.resolver import ResolvedModel

console = Console()

# Rate limiting to comply with HuggingFace ToS (R8)
_last_api_call: float = 0
_MIN_INTERVAL: float = 0.5  # Minimum 0.5 seconds between API calls

# Configure User-Agent to identify the tool (R8)
# This allows HuggingFace to identify the origin of requests
try:
    from hfl import __version__
except ImportError:
    __version__ = "0.1.0"

os.environ.setdefault("HF_HUB_USER_AGENT", f"hfl/{__version__}")


def _rate_limit() -> None:
    """Apply rate limiting between API calls."""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_api_call = time.time()


def pull_model(resolved: ResolvedModel) -> Path:
    """
    Download a model and return the local path.

    For GGUF: downloads the individual file.
    For safetensors: downloads the complete repo snapshot.
    """
    # Rate limiting before API calls (R8 - ToS compliance)
    _rate_limit()
    token = ensure_auth(resolved.repo_id)

    # Destination directory: ~/.hfl/models/<org>/<model>/
    model_dir = config.models_dir / resolved.repo_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold cyan]Downloading[/] {resolved.repo_id}"
        + (f" ({resolved.filename})" if resolved.filename else "")
    )

    if resolved.format == "gguf" and resolved.filename:
        # Individual GGUF file download
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
        # Complete snapshot download
        # Filter only the necessary files
        allow_patterns = []
        if resolved.format == "safetensors":
            allow_patterns = [
                "*.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",  # SentencePiece
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

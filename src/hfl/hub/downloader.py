# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Model download from HuggingFace Hub with visual progress.

Supports:
- Individual GGUF file download
- Complete repo download (safetensors)
- Resume of interrupted downloads
- Smart cache (no re-download if already exists)
- Automatic retry on network errors

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
from hfl.logging_config import get_logger
from hfl.utils.retry import with_retry

console = Console()
logger = get_logger()

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


# Network exceptions that should trigger retry
# Use stdlib/httpx exceptions for compatibility with huggingface-hub >=1.0 (uses httpx)
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (ConnectionError, TimeoutError, OSError)
try:
    from httpx import ConnectError, TimeoutException

    _RETRYABLE_EXCEPTIONS = (ConnectError, TimeoutException, OSError)
except ImportError:
    pass


def _on_download_retry(exception: Exception, attempt: int) -> None:
    """Log retry attempts for downloads."""
    logger.warning("Download attempt %s failed: %s. Retrying...", attempt, exception)
    console.print(f"[yellow]Retry {attempt}:[/] {type(exception).__name__} - Retrying...")


@with_retry(
    max_retries=config.max_retries,
    base_delay=config.retry_base_delay,
    max_delay=config.retry_max_delay,
    exceptions=_RETRYABLE_EXCEPTIONS,
    on_retry=_on_download_retry,
)
def _download_file(
    repo_id: str,
    filename: str,
    revision: str | None,
    local_dir: Path,
    token: str | None,
) -> Path:
    """Download a single file with retry logic."""
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=local_dir,
        token=token,
    )
    return Path(local_path)


@with_retry(
    max_retries=config.max_retries,
    base_delay=config.retry_base_delay,
    max_delay=config.retry_max_delay,
    exceptions=_RETRYABLE_EXCEPTIONS,
    on_retry=_on_download_retry,
)
def _download_snapshot(
    repo_id: str,
    revision: str | None,
    local_dir: Path,
    token: str | None,
    allow_patterns: list[str] | None,
) -> Path:
    """Download a repo snapshot with retry logic."""
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        token=token,
        allow_patterns=allow_patterns,
    )
    return Path(local_path)


def pull_model(resolved: ResolvedModel) -> Path:
    """
    Download a model and return the local path.

    For GGUF: downloads the individual file.
    For safetensors: downloads the complete repo snapshot.

    Automatically retries on network errors with exponential backoff.
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
        # Individual GGUF file download with retry
        return _download_file(
            repo_id=resolved.repo_id,
            filename=resolved.filename,
            revision=resolved.revision,
            local_dir=model_dir,
            token=token,
        )
    else:
        # Complete snapshot download with retry
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

        return _download_snapshot(
            repo_id=resolved.repo_id,
            revision=resolved.revision,
            local_dir=model_dir,
            token=token,
            allow_patterns=allow_patterns or None,
        )

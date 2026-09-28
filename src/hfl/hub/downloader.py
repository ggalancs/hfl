# SPDX-License-Identifier: Apache-2.0
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

import importlib
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


# Only genuine transport/network failures should be retried — NOT HTTP
# status errors. In huggingface_hub 1.x, ``HfHubHTTPError`` (and thus
# ``GatedRepoError`` / ``RepositoryNotFoundError`` / 401 / 403 / 404)
# inherits from ``OSError``, so the old blanket ``OSError`` here silently
# retried *permanent* failures 4× and masked them as ``RetryExhausted``:
# a gated model ("accept the license on huggingface.co") looked like a
# network timeout. ``httpx.TransportError`` is the base for
# connection/timeout/protocol errors only — it excludes ``HTTPStatusError``
# / ``HfHubHTTPError``, so gated/auth/not-found now propagate immediately
# with their real message. (``ConnectionError``/``TimeoutError`` in the
# stdlib fallback are the *specific* OSError subclasses, not the broad
# base, so they likewise don't swallow HTTP errors.)
#
# huggingface_hub 2.x raises httpx2's errors, 1.x httpx's: separate classes,
# and catching only httpx's would stop retrying every failure under 2.x.
def _retryable() -> tuple[type[Exception], ...]:
    found: list[type[Exception]] = []
    for module in ("httpx", "httpx2"):
        try:
            found.append(importlib.import_module(module).TransportError)
        except (ImportError, AttributeError):
            continue
    return tuple(found) or (ConnectionError, TimeoutError)


_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = _retryable()


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


_SAFETENSORS_FILES = [
    "*.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",  # SentencePiece
    "generation_config.json",
]


def model_dir_for(resolved: ResolvedModel) -> Path:
    """Where ``pull_model`` puts ``resolved``: ~/.hfl/models/<org>--<model>/."""
    return config.models_dir / resolved.repo_id.replace("/", "--")


def _planned(resolved: ResolvedModel, names: list[str]) -> list[str]:
    """Of a repo's files, the ones ``pull_model`` fetches for ``resolved``."""
    import fnmatch

    if resolved.format == "gguf" and resolved.filename:
        wanted = {resolved.filename, *resolved.parts, *filter(None, [resolved.projector])}
        return [n for n in names if n in wanted]
    if resolved.format == "safetensors":
        return [n for n in names if any(fnmatch.fnmatch(n, p) for p in _SAFETENSORS_FILES)]
    return list(names)


def expected_files(resolved: ResolvedModel) -> dict[str, int]:
    """The files a pull of ``resolved`` downloads and their sizes, from the
    Hub; empty when it cannot say (a progress total is then unknown)."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(
            resolved.repo_id,
            revision=resolved.revision,
            files_metadata=True,
            token=_token_quietly(resolved.repo_id),
        )
    except Exception:
        return {}
    sizes = {s.rfilename: int(s.size or 0) for s in (info.siblings or [])}
    return {name: sizes[name] for name in _planned(resolved, list(sizes))}


def _token_quietly(repo_id: str) -> str | None:
    try:
        return ensure_auth(repo_id)
    except Exception:
        return None


def bytes_done(resolved: ResolvedModel, planned: dict[str, int]) -> int:
    """Bytes of ``planned`` on disk now: files already in place plus the
    ones ``huggingface_hub`` is still writing (``*.incomplete``)."""
    folder = model_dir_for(resolved)
    done = 0
    for name in planned:
        try:
            done += (folder / name).stat().st_size
        except OSError:
            pass
    partial = folder / ".cache" / "huggingface" / "download"
    try:
        done += sum(p.stat().st_size for p in partial.rglob("*.incomplete"))
    except OSError:
        pass
    return done


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

    # Destination directory: ~/.hfl/models/<org>--<model>/
    model_dir = model_dir_for(resolved)
    model_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold cyan]Downloading[/] {resolved.repo_id}"
        + (f" ({resolved.filename})" if resolved.filename else "")
    )

    if resolved.format == "gguf" and resolved.filename:
        # Individual GGUF file download with retry — with the rest of a
        # split model and a vision model's projector, which llama.cpp reads
        # from beside it (without them the model does not load, or cannot
        # see images).
        path = _download_file(
            repo_id=resolved.repo_id,
            filename=resolved.filename,
            revision=resolved.revision,
            local_dir=model_dir,
            token=token,
        )
        extras = [*getattr(resolved, "parts", []), getattr(resolved, "projector", None)]
        for filename in filter(None, extras):
            console.print(f"[bold cyan]Downloading[/] {filename}")
            _download_file(
                repo_id=resolved.repo_id,
                filename=filename,
                revision=resolved.revision,
                local_dir=model_dir,
                token=token,
            )
        return path
    # Complete snapshot download with retry
    # Filter only the necessary files
    allow_patterns = list(_SAFETENSORS_FILES) if resolved.format == "safetensors" else []

    return _download_snapshot(
        repo_id=resolved.repo_id,
        revision=resolved.revision,
        local_dir=model_dir,
        token=token,
        allow_patterns=allow_patterns or None,
    )

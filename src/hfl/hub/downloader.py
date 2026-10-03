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
    ignore_patterns: list[str] | None = None,
) -> Path:
    """Download a repo snapshot with retry logic."""
    local_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return Path(local_path)


def _ref(resolved: ResolvedModel) -> str | None:
    """The revision every request of a pull uses: the commit the resolver
    saw when it has one. A branch name ("main") moves: the files, the parts
    of a split GGUF and the sha256 they are checked against could each come
    from a different commit than the one recorded as pulled."""
    return getattr(resolved, "commit_sha", None) or resolved.revision


# Python in the repo runs only through trust_remote_code (or llama.cpp's
# converter, which trusts it for some architectures). Without the operator's
# opt-in it is not downloaded at all: a repo with neither safetensors nor
# GGUF is fetched whole, and "whole" included its .py files.
_CODE_FILES = ["*.py"]


def _code_excluded() -> list[str] | None:
    from hfl.security import remote_code_allowed

    return None if remote_code_allowed() else list(_CODE_FILES)


def _hub_sha256(resolved: ResolvedModel, token: str | None) -> dict[str, str]:
    """The sha256 the Hub publishes for each of the repo's LFS files (the
    weights), by name; empty when the Hub could not say."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(
            resolved.repo_id, revision=_ref(resolved), files_metadata=True, token=token
        )
    except Exception as exc:
        logger.warning("Could not read the Hub's checksums for %s: %s", resolved.repo_id, exc)
        return {}
    out: dict[str, str] = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        digest = getattr(lfs, "sha256", None) if lfs is not None else None
        if isinstance(digest, str) and digest:
            out[sibling.rfilename] = digest.lower()
    return out


def _sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_downloads(
    resolved: ResolvedModel, model_dir: Path, token: str | None, names: list[str]
) -> None:
    """Compare the downloaded weights with the sha256 the Hub publishes.

    huggingface_hub checks a download's size, not its content: a file
    damaged in transit, on disk, or resumed from a broken partial passed.
    A mismatch is deleted and downloaded once more; a second mismatch is a
    ``DownloadIntegrityError``. Files the Hub gives no sha256 for (small
    git files) are not checked, and saying "checked" when nothing could be
    compared would be false: that case is reported as not checked.
    """
    from hfl.exceptions import DownloadIntegrityError

    if not config.verify_downloads:
        return
    expected = _hub_sha256(resolved, token)
    present = [n for n in names if n in expected and (model_dir / n).is_file()]
    if not present:
        console.print("[yellow]Integrity not checked: the Hub gave no sha256 for these files[/]")
        return
    for name in present:
        path = model_dir / name
        actual = _sha256_of(path)
        if actual == expected[name]:
            continue
        logger.warning("%s: sha256 mismatch (%s…); downloading it again", name, actual[:16])
        console.print(f"[yellow]{name} does not match the Hub's sha256; downloading it again[/]")
        _discard(model_dir, name)
        _download_file(resolved.repo_id, name, _ref(resolved), model_dir, token)
        actual = _sha256_of(path)
        if actual != expected[name]:
            _discard(model_dir, name)
            raise DownloadIntegrityError(resolved.repo_id, name, expected[name], actual)
    console.print(f"[dim]Checked {len(present)} file(s) against the Hub's sha256[/]")


def _discard(model_dir: Path, name: str) -> None:
    """Delete a downloaded file and huggingface_hub's record of it, so the
    next download fetches it again instead of trusting the copy on disk."""
    (model_dir / name).unlink(missing_ok=True)
    (model_dir / ".cache" / "huggingface" / "download" / f"{name}.metadata").unlink(missing_ok=True)


# What a safetensors model needs beside its weights. The index of a sharded
# model ends in .json, not .safetensors: without it every model split in
# shards (nearly every one of 7B or more) downloaded but could not load in
# Transformers ("no file named model.safetensors", measured on an L4).
_SAFETENSORS_FILES = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",  # SentencePiece
    "vocab.json",  # byte-level BPE (Qwen, GPT-2 family)
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",  # the chat template, in newer repos
    "chat_template.json",
    "preprocessor_config.json",  # vision-language models
    "processor_config.json",
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
    excluded = _code_excluded() or []
    return [n for n in names if not any(fnmatch.fnmatch(n, p) for p in excluded)]


def expected_files(resolved: ResolvedModel) -> dict[str, int]:
    """The files a pull of ``resolved`` downloads and their sizes, from the
    Hub; empty when it cannot say (a progress total is then unknown)."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(
            resolved.repo_id,
            revision=_ref(resolved),
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
            revision=_ref(resolved),
            local_dir=model_dir,
            token=token,
        )
        extras = [*getattr(resolved, "parts", []), getattr(resolved, "projector", None)]
        for filename in filter(None, extras):
            console.print(f"[bold cyan]Downloading[/] {filename}")
            _download_file(
                repo_id=resolved.repo_id,
                filename=filename,
                revision=_ref(resolved),
                local_dir=model_dir,
                token=token,
            )
        _verify_downloads(resolved, model_dir, token, [resolved.filename, *filter(None, extras)])
        return path
    # Complete snapshot download with retry
    # Filter only the necessary files
    allow_patterns = list(_SAFETENSORS_FILES) if resolved.format == "safetensors" else []

    snapshot = _download_snapshot(
        repo_id=resolved.repo_id,
        revision=_ref(resolved),
        local_dir=model_dir,
        token=token,
        allow_patterns=allow_patterns or None,
        ignore_patterns=_code_excluded(),
    )
    downloaded = [str(p.relative_to(model_dir)) for p in model_dir.rglob("*") if p.is_file()]
    _verify_downloads(resolved, model_dir, token, [n for n in downloaded if ".cache" not in n])
    return snapshot

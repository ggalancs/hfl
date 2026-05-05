# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HuggingFace Hub upload helpers used by ``POST /api/push``.

Pushing a locally-registered model means: create the target repo if
needed, upload every artefact in its on-disk directory (weights,
tokenizer, manifest, modelfile), and stream progress back to the
caller in NDJSON. The bulk of the work lives in ``huggingface_hub``;
this module wraps it with HFL-specific resolution (registry lookup,
manifest path discovery) and an Iterator-friendly progress shape.

The upload function is split into two pieces so tests can patch the
network layer without setting up a real HF account:

- :func:`build_upload_plan` — pure: takes a manifest, returns an
  ``UploadPlan`` (target repo, list of files, total size).
- :func:`stream_push` — async generator that drives ``HfApi`` and
  yields progress dicts; tests inject a fake ``api`` to assert on
  the events without hitting the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable

if TYPE_CHECKING:
    from huggingface_hub import HfApi

    from hfl.models.manifest import ModelManifest

logger = logging.getLogger(__name__)

__all__ = ["UploadPlan", "build_upload_plan", "stream_push"]


@dataclass(frozen=True)
class UploadPlan:
    """Resolved push target + the files that will go up.

    Frozen so we can pass it across awaits without worrying about the
    caller mutating it mid-upload.
    """

    repo_id: str
    """``namespace/model`` — what the Hub will call the repo."""

    revision: str | None
    """Branch / tag to push to. ``None`` defaults to ``"main"``."""

    local_dir: Path
    """Directory whose contents will be uploaded."""

    files: tuple[Path, ...]
    """Resolved file list (already filtered). Order is deterministic."""

    total_bytes: int
    """Sum of file sizes — used by progress events."""


def _iter_uploadable_files(local_dir: Path) -> Iterable[Path]:
    """Yield files inside ``local_dir`` that should be pushed.

    Skips the obvious non-artefact directories (cache, lock files,
    Python bytecode). The caller is expected to have validated that
    ``local_dir`` exists and is a directory.
    """
    skip_dirs = {"__pycache__", ".cache", ".lock", ".tmp"}
    skip_suffixes = {".pyc", ".pyo", ".tmp", ".lock"}

    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix in skip_suffixes:
            continue
        if path.name.startswith("."):
            continue
        yield path


def build_upload_plan(
    manifest: "ModelManifest",
    *,
    target_repo_id: str,
    revision: str | None = None,
) -> UploadPlan:
    """Resolve a push target into a concrete ``UploadPlan``.

    Args:
        manifest: The local model registry entry.
        target_repo_id: ``namespace/model`` on the Hub.
        revision: Branch / tag — defaults to ``main`` at upload time.

    Raises:
        FileNotFoundError: ``manifest.local_path`` is missing or empty.
        ValueError: ``target_repo_id`` doesn't have the
            ``namespace/model`` shape.
    """
    if "/" not in target_repo_id or target_repo_id.startswith("/"):
        raise ValueError(f"target_repo_id must be ``namespace/model``: got {target_repo_id!r}")

    local = Path(manifest.local_path)
    if not local.exists():
        raise FileNotFoundError(f"manifest.local_path does not exist: {local}")

    # ``local_path`` may point at a file (a single .gguf) or a
    # directory (transformers-style snapshot). Both are valid push
    # sources; for a single file we just pass its parent + filter.
    files: tuple[Path, ...]
    if local.is_file():
        local_dir = local.parent
        files = (local,)
    else:
        local_dir = local
        files = tuple(_iter_uploadable_files(local_dir))

    if not files:
        raise FileNotFoundError(f"no uploadable files under {local_dir}")

    total_bytes = sum(p.stat().st_size for p in files)

    return UploadPlan(
        repo_id=target_repo_id,
        revision=revision,
        local_dir=local_dir,
        files=files,
        total_bytes=total_bytes,
    )


async def stream_push(
    plan: UploadPlan,
    *,
    api: "HfApi",
    private: bool = False,
    token: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Drive the upload, yielding NDJSON-shaped progress events.

    Event grammar (one dict per yield):

    - ``{"status": "preparing", "total": <bytes>}`` — first event
    - ``{"status": "ensuring repository", "repo": "...", "private": ...}``
    - ``{"status": "uploading", "current": <bytes>, "total": <bytes>}``
      — emitted once before the upload starts (granular progress
      lives inside ``upload_folder`` and is not exposed)
    - ``{"status": "success", "repo": "...", "revision": "..."}``
    - ``{"error": "...", "status": "failed"}`` on any failure
    """
    import asyncio

    yield {"status": "preparing", "total": plan.total_bytes}

    # Step 1: ensure the repo exists. ``create_repo(exist_ok=True)`` is
    # idempotent so a re-push to an existing repo is fine.
    try:
        await asyncio.to_thread(
            api.create_repo,
            repo_id=plan.repo_id,
            private=private,
            token=token,
            exist_ok=True,
        )
    except Exception as exc:
        logger.exception("create_repo failed for %s", plan.repo_id)
        yield {"status": "failed", "error": f"create_repo: {exc}"}
        return

    yield {
        "status": "ensuring repository",
        "repo": plan.repo_id,
        "private": private,
    }

    # Step 2: upload the folder. ``upload_folder`` handles resume,
    # parallelism and checksumming internally; we just publish the
    # bracketing progress events.
    yield {
        "status": "uploading",
        "current": 0,
        "total": plan.total_bytes,
    }

    try:
        commit = await asyncio.to_thread(
            api.upload_folder,
            folder_path=str(plan.local_dir),
            repo_id=plan.repo_id,
            revision=plan.revision,
            token=token,
            commit_message="hfl push",
        )
    except Exception as exc:
        logger.exception("upload_folder failed for %s", plan.repo_id)
        yield {"status": "failed", "error": f"upload_folder: {exc}"}
        return

    revision = plan.revision or "main"
    commit_url = getattr(commit, "commit_url", None)

    yield {
        "status": "success",
        "repo": plan.repo_id,
        "revision": revision,
        "commit_url": commit_url,
    }

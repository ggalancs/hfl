# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl outdated``: which pulled models have a newer version on the Hub.

A pull records the commit it came from. The Hub is asked for the commit its
branch points at now and, when that moved, for each file the model uses —
by their git blob ids at both commits, so nothing is downloaded and a README
edit is not an update. What cannot be checked says so, and why: an
unreachable Hub is never reported as "up to date".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hfl.models.manifest import ModelManifest

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class Check:
    """What was found for one model.

    ``status``: ``current`` (same commit), ``files_current`` (the repo moved,
    the model's files did not), ``update`` (``changed`` lists the files),
    ``update_repo`` (the repo moved and the files cannot be compared: a GGUF
    converted here), ``gone`` (a file is no longer on the branch), ``pinned``
    (pulled at a commit), ``local`` (imported, not from the Hub),
    ``unrecorded`` (pulled before commits were recorded), ``unchecked``
    (``detail`` says why: ``gated``, ``not_found``, ``no_revision`` or
    ``unreachable``; ``error`` names the exception).
    """

    name: str
    status: str
    changed: list[str] = field(default_factory=list)
    hub_sha: str | None = None
    detail: str = ""
    error: str = ""
    command: str | None = None


def _repo_root(manifest: ModelManifest, models_dir: Path) -> Path:
    return models_dir / manifest.repo_id.replace("/", "--")


def local_files(manifest: ModelManifest, models_dir: Path) -> list[str]:
    """The model's files, as paths inside its Hub repo: a GGUF with its other
    parts and its projector, or every file of a folder. Empty when they do
    not sit in the repo's layout (a model converted to GGUF here)."""
    root = _repo_root(manifest, models_dir)
    path = Path(manifest.local_path)
    try:
        path.relative_to(root)
    except ValueError:
        return []
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
    else:
        from hfl.engine.projector import find_projector

        split = re.search(r"-\d{5}-of-(\d{5})$", path.stem)
        files = (
            sorted(path.parent.glob(f"{path.stem[: split.start()]}-*-of-{split.group(1)}.gguf"))
            if split
            else [path]
        )
        projector = find_projector(path)
        if projector is not None:
            files.append(projector)
    out = []
    for file in files:
        rel = file.relative_to(root)
        if not any(part.startswith(".") for part in rel.parts):  # .cache/ and the like
            out.append(rel.as_posix())
    return sorted(out)


def update_command(manifest: ModelManifest) -> str:
    """The ``hfl pull`` that fetches the newer files into the same entry."""
    ref = manifest.revision if manifest.revision not in (None, "main") else None
    revision = f" --revision {ref}" if ref else ""
    if manifest.format == "gguf" and manifest.quantization:
        return f"hfl pull {manifest.repo_id} -q {manifest.quantization}{revision}"
    return f"hfl pull {manifest.repo_id} --format safetensors{revision}"


def _blobs(api: Any, repo_id: str, revision: str) -> tuple[str, dict[str, str]]:
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    blobs = {s.rfilename: (s.blob_id or "") for s in (info.siblings or [])}
    return str(info.sha), blobs


def _why(exc: Exception) -> str:
    """Why the Hub could not answer, as a key the command words."""
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(exc, GatedRepoError):  # before its parent, RepositoryNotFoundError
        return "gated"
    if isinstance(exc, RepositoryNotFoundError):
        return "not_found"
    if isinstance(exc, RevisionNotFoundError):
        return "no_revision"
    return "unreachable"


def _asks_the_hub(manifest: ModelManifest) -> bool:
    """Whether ``check`` needs the Hub for this model (see its first cases)."""
    local = manifest.repo_id.startswith("local/")
    pinned = bool(manifest.revision and _COMMIT.match(manifest.revision))
    return not local and not pinned and bool(manifest.commit_sha)


def check(manifest: ModelManifest, api: Any, models_dir: Path) -> Check:
    """Compare one model with the Hub."""
    result = Check(name=manifest.name, status="unchecked")
    if manifest.repo_id.startswith("local/"):
        result.status = "local"
        return result
    if manifest.revision and _COMMIT.match(manifest.revision):
        result.status = "pinned"
        return result
    if not manifest.commit_sha:
        # No command: pulling again to find out can mean re-downloading (and
        # converting) tens of GB for nothing.
        result.status = "unrecorded"
        return result
    ref = manifest.revision or "main"
    try:
        hub_sha, now = _blobs(api, manifest.repo_id, ref)
        result.hub_sha = hub_sha
        if hub_sha == manifest.commit_sha:
            result.status = "current"
            return result
        mine = local_files(manifest, models_dir)
        if not mine:
            # A GGUF converted here: its source files are not kept to compare.
            result.status = "update_repo"
            result.command = update_command(manifest)
            return result
        _, then = _blobs(api, manifest.repo_id, manifest.commit_sha)
    except Exception as exc:  # the network, the Hub, a token: all "cannot tell"
        result.detail, result.error = _why(exc), type(exc).__name__
        return result
    gone = [f for f in mine if f in then and f not in now]
    result.changed = [f for f in mine if f in then and f in now and then[f] != now[f]]
    if gone:
        result.status, result.changed = "gone", gone
    elif result.changed:
        result.status = "update"
        result.command = update_command(manifest)
    else:
        result.status = "files_current"
    return result


def check_all(manifests: list[ModelManifest], api: Any, models_dir: Path) -> list[Check]:
    """``check`` for each model — but once the Hub cannot be reached, the
    rest are not tried: with no network each would wait out its timeout."""
    results: list[Check] = []
    down: Check | None = None
    for manifest in manifests:
        if down is not None and _asks_the_hub(manifest):
            results.append(Check(manifest.name, "unchecked", detail=down.detail, error=down.error))
            continue
        result = check(manifest, api, models_dir)
        if result.detail == "unreachable":
            down = result
        results.append(result)
    return results

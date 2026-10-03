# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Is a model already on disk? — without touching the network.

Routes that serve models straight from a Hub id (speech-to-text, images)
answer a caller who may not download with the models already here, and
use these checks to say "not on this server" instead of failing deep in
a library. The engines also get ``local_files_only``: these checks shape
the message; the flag is the guard.

Those callers may name models, not places: a filesystem path is the
owner's (:func:`is_model_id` tells them apart), and the checks below look
only in the caches. Answering "that path exists" made them a file-existence
oracle for anybody (404 vs 500 for ``~/.ssh/known_hosts``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# A Hub repo id (``org/name`` or a legacy bare ``name``) or a Whisper size
# (``small``, ``large-v3``, ``tiny.en``). Every segment starts with a letter
# or digit, so no absolute, ``~``, ``.``/``..`` or drive-letter path gets
# through.
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?")


def is_model_id(model: str) -> bool:
    """Whether ``model`` names a model (repo id or size), not a path."""
    return _MODEL_ID.fullmatch(model) is not None


def hub_model_available_locally(model: str) -> bool:
    """A Hub repo already in the Hugging Face cache."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model, local_files_only=True)
    except Exception:
        return False
    return True


def whisper_available_locally(model: str) -> bool:
    """A Whisper size (``small``, ``large-v3``...) or repo id already on disk."""
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        return openai_whisper_cached(model)
    try:
        download_model(model, local_files_only=True)
    except Exception:
        return False
    return True


def openai_whisper_cached(model: str) -> bool:
    """openai-whisper keeps its checkpoints as ``~/.cache/whisper/<name>.pt``."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "whisper"
    return (cache / f"{model}.pt").exists()

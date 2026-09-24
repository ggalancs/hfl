# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Is a model already on disk? — without touching the network.

Routes that serve models straight from a Hub id (speech-to-text, images)
answer a caller who may not download with the models already here, and
use these checks to say "not on this server" instead of failing deep in
a library. The engines also get ``local_files_only``: these checks shape
the message; the flag is the guard.
"""

from __future__ import annotations

import os
from pathlib import Path


def hub_model_available_locally(model: str) -> bool:
    """A local path, or a Hub repo already in the Hugging Face cache."""
    if Path(model).expanduser().exists():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model, local_files_only=True)
    except Exception:
        return False
    return True


def whisper_available_locally(model: str) -> bool:
    """A Whisper size (``small``, ``large-v3``...) or repo id already on disk."""
    if Path(model).expanduser().exists():
        return True
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

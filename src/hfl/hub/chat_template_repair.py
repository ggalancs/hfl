# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Repair missing chat templates on freshly-pulled MLX models.

Many community MLX conversions (e.g. ``mlx-community/gemma-4-31b-it-4bit``)
drop the original ``chat_template.jinja`` / the ``chat_template`` field
from ``tokenizer_config.json`` during the quantisation pipeline. The
weights load fine, but the first ``apply_chat_template(...)`` call at
serve time crashes with ``ValueError: Cannot use chat template functions
because tokenizer.chat_template is not set``.

This module detects the gap and tries to repair it by fetching the
template from the base repo declared in the MLX model card (``card_data
.base_model``), falling back to a scalar list of well-known upstream
orgs when the field is absent.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["has_chat_template", "ensure_chat_template"]


def has_chat_template(local_path: Path) -> bool:
    """True if the tokenizer is already shipped with a chat template.

    A template can live in two places per Hugging Face convention:

    - ``chat_template.jinja`` as a standalone file (post-2025 style used
      by Gemma 3/4 and most recent Google/Meta releases).
    - ``tokenizer_config.json`` under the ``chat_template`` key (older
      models). When present, this wins over the ``.jinja`` file.
    """
    if (local_path / "chat_template.jinja").exists():
        return True
    cfg = local_path / "tokenizer_config.json"
    if not cfg.exists():
        return False
    try:
        with cfg.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("chat_template"))


def _base_repo_from_card(repo_id: str) -> str | None:
    """Read ``card_data.base_model`` from the repo's ModelCard.

    Returns the first declared base repo. Many MLX quantisation bots
    write this field when they convert an upstream model; for the
    rest we fall back to a heuristic.
    """
    try:
        from huggingface_hub import HfApi  # lazy: optional dep indirection
    except ImportError:
        return None

    try:
        info = HfApi().model_info(repo_id)
    except Exception as exc:  # network / auth / private
        logger.debug("model_info failed for %s: %s", repo_id, exc)
        return None

    card = getattr(info, "card_data", None)
    base = getattr(card, "base_model", None) if card is not None else None
    if isinstance(base, list) and base:
        return str(base[0])
    if isinstance(base, str) and base:
        return base
    return None


def _heuristic_base_repo(repo_id: str) -> str | None:
    """Fallback when ModelCard metadata doesn't declare a base repo.

    Handles the common ``{org}/{name}-[Mm][Ll][Xx]-{N}bit`` pattern. It
    strips the quantisation suffix and rewrites the org to ``google``
    for Gemma variants. Returns ``None`` if the name doesn't match any
    known pattern so the caller can give up cleanly.
    """
    try:
        org, name = repo_id.split("/", 1)
    except ValueError:
        return None

    # Strip ``-MLX-4bit`` / ``-4bit`` / ``-8bit`` / ``-mlx`` tails.
    lower = name.lower()
    stripped = name
    for suffix in (
        "-mlx-4bit",
        "-mlx-6bit",
        "-mlx-8bit",
        "-mlx-bf16",
        "-mlx",
        "-4bit",
        "-6bit",
        "-8bit",
        "-bf16",
    ):
        if lower.endswith(suffix):
            stripped = name[: -len(suffix)]
            break
    if stripped == name:
        return None

    # Well-known quantiser orgs → canonical upstream orgs.
    # Best-effort: callers must still verify the repo exists.
    known_org_map = {
        "mlx-community": None,  # resolved by family below
        "lmstudio-community": None,
    }
    if org not in known_org_map:
        return None

    low = stripped.lower()
    if "gemma" in low:
        return f"google/{stripped}"
    if "llama" in low:
        return f"meta-llama/{stripped}"
    if "qwen" in low:
        return f"Qwen/{stripped}"
    if "mistral" in low or "mixtral" in low:
        return f"mistralai/{stripped}"
    return None


def _resolve_base_repo(repo_id: str) -> str | None:
    """Pick a base repo: ModelCard first, heuristic second."""
    return _base_repo_from_card(repo_id) or _heuristic_base_repo(repo_id)


def _try_download(base_repo: str, filename: str, dest: Path) -> bool:
    """Copy ``filename`` from ``base_repo`` to ``dest``. Returns True on
    success."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        src = hf_hub_download(base_repo, filename)
    except Exception as exc:  # 404, auth, network
        logger.debug("fetch %s from %s failed: %s", filename, base_repo, exc)
        return False
    try:
        shutil.copy(src, dest)
    except OSError as exc:
        logger.debug("copy %s -> %s failed: %s", src, dest, exc)
        return False
    return True


def ensure_chat_template(local_path: Path, repo_id: str) -> bool:
    """Make sure the local model directory carries a chat template.

    Best-effort: returns True when the local dir either already had a
    template or one was successfully copied from the base repo.
    Returns False when we gave up (no base repo resolvable, or the
    base repo did not carry a template either).

    The function never raises — caller logic continues either way.
    """
    if has_chat_template(local_path):
        return True

    base = _resolve_base_repo(repo_id)
    if base is None:
        logger.info(
            "chat_template missing in %s and no base repo could be resolved; "
            "runtime apply_chat_template will fail unless the caller supplies a template",
            repo_id,
        )
        return False

    # Preferred: standalone ``chat_template.jinja`` file.
    if _try_download(base, "chat_template.jinja", local_path / "chat_template.jinja"):
        logger.info("Recovered chat_template.jinja from %s into %s", base, local_path)
        return True

    # Fallback: merge the base tokenizer_config.json's ``chat_template``
    # field into our local one (don't overwrite the rest, the MLX
    # conversion may have legitimate tokenisation changes).
    base_cfg_dest = local_path / "_base_tokenizer_config.tmp.json"
    if not _try_download(base, "tokenizer_config.json", base_cfg_dest):
        return False
    try:
        with base_cfg_dest.open() as f:
            base_cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        base_cfg_dest.unlink(missing_ok=True)
        return False
    template = base_cfg.get("chat_template")
    base_cfg_dest.unlink(missing_ok=True)
    if not template:
        return False

    local_cfg_path = local_path / "tokenizer_config.json"
    try:
        with local_cfg_path.open() as f:
            local_cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    local_cfg["chat_template"] = template
    try:
        with local_cfg_path.open("w") as f:
            json.dump(local_cfg, f, indent=2)
    except OSError:
        return False
    logger.info("Merged chat_template from %s into tokenizer_config.json of %s", base, repo_id)
    return True

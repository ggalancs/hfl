# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""V4 F5 — auto-pick a speculative-decoding draft model from the Hub.

Speculative decoding accelerates large-model generation by predicting
several tokens ahead with a much smaller "draft" model and accepting
the prefix that the large model would have produced anyway. The draft
must share the tokenizer and ideally the architecture family with the
target.

This module picks a sensible draft:

- Same family (Llama → Llama, Qwen → Qwen).
- Smaller parameter count (default at most 1/4 of the target).
- Heavily quantised (Q2_K / Q4_0 / mlx-4bit) so it fits without
  eating into the target's VRAM budget.

The picker is *advisory* — it returns a candidate repo id; the engine
does the actual pull + load. Pure logic + injectable HfApi so tests
exercise it without network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hfl.hub.discovery import (
    DiscoveryQuery,
    _family_for,
    _parameter_estimate_b,
    search_hub,
)

if TYPE_CHECKING:
    from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

__all__ = ["DraftPick", "pick_draft_for"]


@dataclass(frozen=True)
class DraftPick:
    """One draft candidate."""

    repo_id: str
    family: str | None
    parameter_estimate_b: float | None
    quantization: str | None
    rationale: str


# Family-specific defaults: when we can't find a smaller sibling on
# the Hub, we fall back to the canonical "tiny" reference for the
# family. These names are stable across the 2025-2026 generations.
_DEFAULT_DRAFTS: dict[str, str] = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma": "google/gemma-3-270m",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


def _is_smaller(target_b: float | None, candidate_b: float | None, ratio: float) -> bool:
    """True when candidate is at most ``ratio × target`` parameter count.

    Both ``None`` answers are treated conservatively as "we don't
    know" — defaulting to True so the caller can still propose the
    candidate. If you want strictness, set a min and max via the
    ``DiscoveryQuery``.
    """
    if target_b is None or candidate_b is None:
        return True
    return candidate_b <= target_b * ratio


def pick_draft_for(
    target_repo_id: str,
    *,
    api: "HfApi | None" = None,
    max_ratio: float = 0.25,
    prefer_quants: tuple[str, ...] = ("q4_k_m", "q4_0", "q2_k", "mlx-4bit"),
) -> DraftPick | None:
    """Return the best draft candidate for ``target_repo_id`` or
    ``None`` if no Hub query produces a usable sibling.

    Strategy:

    1. Detect the target family + parameter count from the repo id.
    2. Query the Hub for the same family, sorted by downloads.
    3. Filter to candidates with ≤ ``max_ratio × target_b`` params.
    4. Prefer heavily-quantised forks; reward popularity within the
       smaller bucket.
    5. Fall back to the canonical small reference for the family
       (``_DEFAULT_DRAFTS``) when the Hub didn't yield anything.
    """
    family = _family_for(target_repo_id, [])
    target_b = _parameter_estimate_b(target_repo_id, [])

    if family is None and target_b is None:
        return None

    query = DiscoveryQuery(
        family=family,
        page_size=40,
        min_likes=20,
    )
    try:
        candidates = search_hub(query, api=api)
    except Exception:
        logger.exception("draft picker Hub query failed for %s", target_repo_id)
        candidates = []

    smaller = [
        c
        for c in candidates
        if _is_smaller(target_b, c.parameter_estimate_b, max_ratio) and c.repo_id != target_repo_id
    ]

    # Score: matching prefer_quants → +3 each, log-scaled likes → +1.
    def _score(entry) -> float:
        s = 0.0
        if entry.quantization in prefer_quants:
            s += 3.0
        # Smaller is better (fewer parameters → faster predictor).
        if entry.parameter_estimate_b is not None and target_b is not None:
            s += max(0.0, 1.0 - entry.parameter_estimate_b / max(target_b, 1.0))
        # Mild popularity nudge so frontier-quality drafts beat odd
        # bespoke forks.
        s += min(1.0, max(0.0, entry.likes / 1000.0))
        return s

    if smaller:
        best = max(smaller, key=_score)
        rationale = (
            f"smaller sibling ({best.parameter_estimate_b or '?'}B vs {target_b or '?'}B target)"
        )
        return DraftPick(
            repo_id=best.repo_id,
            family=best.family,
            parameter_estimate_b=best.parameter_estimate_b,
            quantization=best.quantization,
            rationale=rationale,
        )

    fallback = _DEFAULT_DRAFTS.get(family or "")
    if fallback:
        return DraftPick(
            repo_id=fallback,
            family=family,
            parameter_estimate_b=None,
            quantization=None,
            rationale=f"no smaller fork found; canonical {family!r} draft",
        )

    return None

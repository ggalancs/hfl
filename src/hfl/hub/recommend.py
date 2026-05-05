# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Hardware-aware recommendation — V4 F1.2.

Combines :mod:`hfl.hub.discovery` with :mod:`hfl.hub.hw_profile` and
:mod:`hfl.hub.quant_table` to produce a top-N list of HF Hub models
that fit the current host. Score blends:

- ``hardware_fit``: 1.0 when the model fits comfortably, < 1.0 when
  it pushes the budget; 0.0 when it overflows.
- ``capability_fit``: how well the model's tags match the requested
  ``task`` (chat / code / vision / embeddings).
- ``popularity``: log-scaled likes × downloads; rewards mainstream
  picks for first-time users.
- ``recency``: newer models get a small boost so frontier releases
  appear sooner than legacy curiosities.

Score is deterministic given the same inputs — important so the
recommendation is reproducible in tests.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from hfl.hub.discovery import DiscoveryEntry, DiscoveryQuery, search_hub
from hfl.hub.hw_profile import HardwareProfile, get_hw_profile
from hfl.hub.quant_table import estimate_vram_gb

if TYPE_CHECKING:
    from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


Task = Literal["chat", "code", "vision", "embeddings", "tools"]


@dataclass
class Recommendation:
    """One top-N recommendation row."""

    repo_id: str
    family: str | None
    quantization: str | None
    parameter_estimate_b: float | None
    likes: int
    downloads: int
    license: str | None
    gated: bool
    estimated_vram_gb: float
    score: float
    reasoning: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------


def _hardware_fit(
    entry: DiscoveryEntry, profile: HardwareProfile
) -> tuple[float, float, list[str]]:
    """Return ``(score, estimated_vram_gb, reasoning)``.

    Score is in ``[0.0, 1.0]``: 1.0 when the model fits with > 30%
    headroom, 0.0 when it overflows.
    """
    reasons: list[str] = []
    params_b = entry.parameter_estimate_b or 7.0  # default to 7B
    quant = entry.quantization or "f16"

    # Pick a budget. CUDA: VRAM. Apple Silicon: 70% of unified RAM.
    # CPU-only / unknown: 70% of system RAM.
    if profile.gpu_kind == "cuda" and profile.gpu_vram_gb:
        budget = profile.gpu_vram_gb
    elif profile.gpu_kind == "metal" and profile.gpu_vram_gb:
        budget = profile.gpu_vram_gb
    else:
        budget = round((profile.system_ram_gb or 8.0) * 0.7, 1)

    estimate = estimate_vram_gb(params_b=params_b, quantization=quant)
    if estimate.total_gb == 0 or budget == 0:
        return 0.0, estimate.total_gb, reasons

    headroom = budget - estimate.total_gb
    if headroom < 0:
        reasons.append(f"overflows budget ({estimate.total_gb:.1f}/{budget:.1f} GB available)")
        return 0.0, estimate.total_gb, reasons

    # Linear ramp: full score above 30% headroom, half-score at 0%
    # headroom (i.e. just barely fits).
    ratio = headroom / max(budget, 1.0)
    if ratio >= 0.3:
        score = 1.0
        reasons.append(f"fits comfortably ({estimate.total_gb:.1f}/{budget:.1f} GB)")
    else:
        score = 0.5 + ratio  # 0.5..0.8 in [0..0.3] range
        reasons.append(f"tight fit ({estimate.total_gb:.1f}/{budget:.1f} GB)")
    return score, estimate.total_gb, reasons


_TASK_TAGS: dict[Task, tuple[str, ...]] = {
    "chat": ("instruct", "chat", "assistant"),
    "code": ("code", "coder", "starcoder", "deepseek-coder"),
    "vision": ("vision", "vl", "image", "multimodal"),
    "embeddings": ("embedding", "sentence-transformers", "feature-extraction"),
    "tools": ("tool", "function-calling", "agent"),
}


def _capability_fit(entry: DiscoveryEntry, task: Task | None) -> tuple[float, list[str]]:
    if task is None:
        return 0.5, []
    needles = _TASK_TAGS.get(task, ())
    pool = (entry.repo_id + " " + " ".join(entry.tags) + " " + (entry.pipeline_tag or "")).lower()
    for needle in needles:
        if needle in pool:
            return 1.0, [f"matches task '{task}' (tag: {needle!r})"]
    return 0.2, [f"weak match for task '{task}'"]


def _popularity(entry: DiscoveryEntry) -> tuple[float, list[str]]:
    """Log-scaled blend of likes and downloads, normalised to [0..1]."""
    log_likes = math.log10(max(entry.likes, 1))  # 0..6+
    log_dl = math.log10(max(entry.downloads, 1))  # 0..9+
    raw = (log_likes / 6.0) * 0.5 + (log_dl / 9.0) * 0.5
    score = max(0.0, min(1.0, raw))
    if score >= 0.7:
        return score, ["mainstream pick"]
    if score >= 0.3:
        return score, ["moderately popular"]
    return score, ["niche pick"]


def _recency(entry: DiscoveryEntry) -> tuple[float, list[str]]:
    """Decay over the last 24 months. Newer models get up to 1.0."""
    if not entry.last_modified:
        return 0.5, []
    try:
        ts = datetime.fromisoformat(entry.last_modified.replace("Z", "+00:00"))
    except ValueError:
        return 0.5, []
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - ts).days
    if age_days < 0:
        return 1.0, []
    score = max(0.0, 1.0 - age_days / 730.0)
    return score, []


# ---------------------------------------------------------------------------
# Top-level recommendation function
# ---------------------------------------------------------------------------


_WEIGHTS = {
    "hardware": 0.45,
    "capability": 0.30,
    "popularity": 0.20,
    "recency": 0.05,
}


def recommend_models(
    *,
    task: Task | None = None,
    profile: HardwareProfile | None = None,
    family: str | None = None,
    quantization: str | None = None,
    top_n: int = 10,
    api: "HfApi | None" = None,
) -> list[Recommendation]:
    """Build the top-N recommendation list.

    Pipeline:

    1. Build a base discovery query (large page so we have material
       to score against).
    2. Fetch from the Hub.
    3. Filter out anything that overflows the hardware budget.
    4. Score, sort by score desc, take top-N.
    """
    if profile is None:
        profile = get_hw_profile()

    base_query = DiscoveryQuery(
        family=family,
        quantization=quantization,
        page_size=60,
        min_likes=10,  # filter out long-tail before scoring
    )
    candidates = search_hub(base_query, api=api)

    scored: list[Recommendation] = []
    for entry in candidates:
        hw_score, vram_gb, hw_reasons = _hardware_fit(entry, profile)
        if hw_score == 0:
            continue  # overflows

        cap_score, cap_reasons = _capability_fit(entry, task)
        pop_score, pop_reasons = _popularity(entry)
        rec_score, rec_reasons = _recency(entry)

        total = (
            _WEIGHTS["hardware"] * hw_score
            + _WEIGHTS["capability"] * cap_score
            + _WEIGHTS["popularity"] * pop_score
            + _WEIGHTS["recency"] * rec_score
        )
        scored.append(
            Recommendation(
                repo_id=entry.repo_id,
                family=entry.family,
                quantization=entry.quantization,
                parameter_estimate_b=entry.parameter_estimate_b,
                likes=entry.likes,
                downloads=entry.downloads,
                license=entry.license,
                gated=entry.gated,
                estimated_vram_gb=vram_gb,
                score=round(total, 4),
                reasoning=hw_reasons + cap_reasons + pop_reasons + rec_reasons,
            )
        )

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_n]

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Smart pull — V4 F2.

Given a base repo (e.g. ``meta-llama/Llama-3.1-8B-Instruct``), find
the *best variant* available on the Hub for the current host:

- Apple Silicon + mlx-lm  →  MLX 4-bit when an ``mlx-community/...``
  fork exists, else canonical safetensors.
- CUDA / CPU              →  GGUF (Q4_K_M / Q5_K_M / Q8_0 by budget),
  preferring `bartowski`-class community quants over the base repo.

Returns a ``SmartPullPlan`` describing which repo to pull and which
file to register, leaving the actual byte transfer to ``/api/pull``
(or ``hfl pull``). The plan layer is pure so the recommendation can
be tested without network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from hfl.hub.hw_profile import HardwareProfile, get_hw_profile
from hfl.hub.quant_table import estimate_vram_gb

if TYPE_CHECKING:
    from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

__all__ = ["SmartPullPlan", "build_smart_plan"]


@dataclass
class SmartPullPlan:
    """Resolved best variant for the current hardware."""

    target_repo_id: str
    """The repo to pull from (may be a community quant fork, not the
    base repo the user passed)."""

    quantization: str
    """One of ``mlx-4bit``, ``mlx-8bit``, ``q4_k_m``, ``q5_k_m``,
    ``q8_0``, ``f16``."""

    estimated_vram_gb: float
    """Conservative VRAM estimate after :func:`estimate_vram_gb`."""

    reason: str
    """Human-readable justification — surfaced as part of the
    NDJSON ``preparing`` event."""

    fallback_chain: list[str]
    """Repo IDs probed and rejected (overflow, missing variant, ...).
    Useful for debugging / displayed verbatim by the CLI."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _params_b_from_repo(repo_id: str) -> float | None:
    """Re-use the discovery heuristic for parameter count detection."""
    from hfl.hub.discovery import _parameter_estimate_b

    return _parameter_estimate_b(repo_id, [])


def _budget_gb(profile: HardwareProfile) -> float:
    """Pick the relevant memory budget for the host."""
    if profile.gpu_kind == "cuda" and profile.gpu_vram_gb:
        return profile.gpu_vram_gb
    if profile.gpu_kind == "metal" and profile.gpu_vram_gb:
        return profile.gpu_vram_gb
    return round((profile.system_ram_gb or 8.0) * 0.7, 1)


def _quant_ladder_for(profile: HardwareProfile) -> list[str]:
    """Order of quantisations to probe, best-first.

    On Apple Silicon with MLX we prefer the MLX-native packings;
    elsewhere we walk the GGUF ladder from highest fidelity down.
    """
    if profile.has_mlx:
        return ["mlx-8bit", "mlx-4bit", "q5_k_m", "q4_k_m"]
    return ["q5_k_m", "q4_k_m", "q4_0", "q3_k_m"]


def _candidate_repos(base_repo_id: str, profile: HardwareProfile) -> list[str]:
    """Generate plausible candidate repos for the base.

    Order: most specific community fork first, then the base. On
    Apple Silicon we look at ``mlx-community/<basename>-4bit`` /
    ``-8bit``; elsewhere we look at ``bartowski/<basename>-GGUF`` and
    ``TheBloke/<basename>-GGUF``.
    """
    if "/" not in base_repo_id:
        return [base_repo_id]
    org, name = base_repo_id.split("/", 1)
    candidates: list[str] = []
    if profile.has_mlx:
        # mlx-community uses the bare model name (no org prefix).
        candidates.append(f"mlx-community/{name}-4bit")
        candidates.append(f"mlx-community/{name}-8bit")
    candidates.append(f"bartowski/{name}-GGUF")
    candidates.append(f"TheBloke/{name}-GGUF")
    candidates.append(base_repo_id)
    return candidates


def _repo_exists(api: "HfApi", repo_id: str) -> bool:
    try:
        api.model_info(repo_id)
        return True
    except Exception:
        return False


def _list_quants_in_repo(api: "HfApi", repo_id: str) -> set[str]:
    """Inspect the repo file list to detect available GGUF quants.

    GGUF community repos publish multiple files — ``model-q4_k_m.gguf``,
    ``model-q8_0.gguf`` — and we want to pick the right one. For
    MLX repos the fork name itself encodes the quant (``-4bit``,
    ``-8bit``).
    """
    try:
        info = api.model_info(repo_id)
    except Exception:
        return set()
    siblings = getattr(info, "siblings", []) or []
    seen: set[str] = set()
    for s in siblings:
        name = getattr(s, "rfilename", "") or ""
        lower = name.lower()
        for tag in (
            "q8_0",
            "q6_k",
            "q5_k_m",
            "q5_0",
            "q4_k_m",
            "q4_0",
            "q3_k_m",
            "q2_k",
            "f16",
        ):
            if tag in lower:
                seen.add(tag)
    return seen


def _quant_for_mlx_repo(repo_id: str) -> str | None:
    """Read the MLX quant level off the repo name."""
    lower = repo_id.lower()
    if "-4bit" in lower:
        return "mlx-4bit"
    if "-8bit" in lower:
        return "mlx-8bit"
    return None


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------


def build_smart_plan(
    base_repo_id: str,
    *,
    profile: HardwareProfile | None = None,
    api: "HfApi | None" = None,
    max_vram_gb: float | None = None,
) -> SmartPullPlan:
    """Resolve the best available variant for the current host.

    Probe order:

    1. Per ``_candidate_repos``, check if the repo exists on the Hub.
    2. For MLX forks, parse the quant off the name. For GGUF
       community forks, list files and intersect with the ladder
       returned by :func:`_quant_ladder_for`.
    3. For each (repo, quant), estimate VRAM. First combination
       that fits the budget wins.

    Raises ``ValueError`` when nothing fits — caller should bubble
    up as 400 with a helpful message ("model too large for this
    hardware; try ``--max-vram-gb`` or a smaller repo").
    """
    if profile is None:
        profile = get_hw_profile()
    budget = max_vram_gb or _budget_gb(profile)
    params_b = _params_b_from_repo(base_repo_id) or 7.0

    if api is None:
        from huggingface_hub import HfApi as _HfApi

        api = _HfApi()

    fallback: list[str] = []
    candidates = _candidate_repos(base_repo_id, profile)

    for repo in candidates:
        if not _repo_exists(api, repo):
            fallback.append(f"{repo}: not on Hub")
            continue

        # Determine the quant for this repo.
        mlx_quant = _quant_for_mlx_repo(repo)
        if mlx_quant is not None:
            quants = [mlx_quant]
        else:
            available = _list_quants_in_repo(api, repo)
            ladder = _quant_ladder_for(profile)
            # Keep ladder order, intersect with what's actually published.
            quants = [q for q in ladder if q in available]
            if not quants:
                # No GGUF files found — must be safetensors. Use f16
                # as the heaviest probe; the engine will quantise on
                # load if KV cache type asks for it.
                quants = ["f16"]

        for quant in quants:
            estimate = estimate_vram_gb(params_b=params_b, quantization=quant)
            if estimate.total_gb <= budget:
                reason = (
                    f"picked {repo} @ {quant} ({estimate.total_gb:.1f} GB / {budget:.1f} GB budget)"
                )
                return SmartPullPlan(
                    target_repo_id=repo,
                    quantization=quant,
                    estimated_vram_gb=estimate.total_gb,
                    reason=reason,
                    fallback_chain=fallback,
                )
            fallback.append(
                f"{repo}@{quant}: needs {estimate.total_gb:.1f} GB > budget {budget:.1f} GB"
            )

    raise ValueError(
        f"no variant of {base_repo_id} fits the {budget:.1f} GB budget. "
        f"Tried: {', '.join(fallback) or 'none'}"
    )


def reasons_for(plan: SmartPullPlan) -> Iterable[str]:
    """Render the plan's reasoning for NDJSON / CLI output."""
    yield plan.reason
    for skip in plan.fallback_chain:
        yield f"skipped: {skip}"

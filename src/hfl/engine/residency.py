# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Which models may stay resident, decided by memory rather than by count.

HFL keeps as many models loaded as the machine can hold. The rule an
operator sets is a percentage — ``HFL_MEMORY_BUDGET``, the share of total
RAM the machine may have in use after a load — because a model count says
nothing: two 7B models and two 70B models are both "2".

How much is in use is split in two, because measuring it whole does not
work on this platform. Measured on a 128 GiB Mac with two models loaded
and then unloaded: the process's resident memory fell by the model's full
size, but the system's "available" figure rose by a third of it, because
the pages of a memory-mapped GGUF stay in the file cache. So:

* **other programs** — memory in use minus HFL's own resident set, and
* **HFL's models** — the estimated footprint of each one
  (:mod:`hfl.engine.footprint`), which is what unloading one gives back.

The planner is pure: it takes numbers and returns a decision, so every
branch is testable without loading a model.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from hfl.engine.footprint import GIB

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryView",
    "ResidentView",
    "AdmissionPlan",
    "plan_admission",
    "current_memory",
    "budget_fraction",
    "describe_memory",
]


@dataclass(frozen=True)
class MemoryView:
    """The machine's memory at one instant, in bytes."""

    total: int
    in_use: int
    """Total minus what the OS reports as available."""
    hfl_rss: int
    """This process's resident set."""


@dataclass(frozen=True)
class ResidentView:
    """One loaded model, as the planner needs to see it."""

    name: str
    footprint: int
    last_used: float
    busy: bool = False
    """A request other than the one loading holds it — cannot be unloaded
    now, but will be free once that request ends."""
    mine: bool = False
    """The request doing the load holds it. Waiting for it would wait for
    ourselves, so it is simply kept."""


@dataclass(frozen=True)
class AdmissionPlan:
    fits: bool
    """True when the load may go ahead after the evictions listed."""
    evict: tuple[str, ...] = ()
    wait_for: tuple[str, ...] = ()
    """Busy models whose release would make room. Non-empty only when
    ``fits`` is False and waiting can change the answer."""
    reason: str = "fits"
    """``fits`` | ``evict`` | ``wait`` | ``too_big`` | ``blocked``."""
    limit: int = 0
    used_now: int = 0
    used_after: int = 0
    """Predicted memory in use after the load and the evictions."""
    floor: int = 0
    """Predicted use with every HFL model unloaded and this one loaded —
    the best this machine can do for this model right now."""


def budget_fraction() -> float:
    """``HFL_MEMORY_BUDGET`` as a fraction of total RAM (default 85 %)."""
    from hfl.config import config

    raw = getattr(config, "memory_budget_percent", 85.0)
    try:
        pct = float(raw)
    except (TypeError, ValueError):
        pct = 85.0
    return min(max(pct, 10.0), 100.0) / 100.0


def current_memory() -> MemoryView | None:
    """Measure the machine now, or None when psutil is unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        vm = psutil.virtual_memory()
        rss = psutil.Process(os.getpid()).memory_info().rss
    except Exception as exc:  # pragma: no cover - platform-specific failure
        logger.debug("memory measurement failed: %s", exc)
        return None
    return MemoryView(total=int(vm.total), in_use=int(vm.total - vm.available), hfl_rss=int(rss))


def plan_admission(
    new_footprint: int,
    memory: MemoryView,
    residents: list[ResidentView],
    budget: float,
    max_models: int = 0,
) -> AdmissionPlan:
    """Decide whether a model of ``new_footprint`` bytes may load.

    Order of resort, cheapest first: load alongside everything; unload idle
    models, least recently used first; wait for busy ones to finish; refuse.
    ``max_models`` > 0 adds a count ceiling (Ollama's
    ``OLLAMA_MAX_LOADED_MODELS``); 0 means memory alone decides.
    """
    limit = int(memory.total * budget)
    accounted = sum(r.footprint for r in residents)
    others = max(0, memory.in_use - memory.hfl_rss)
    overhead = max(0, memory.hfl_rss - accounted)  # interpreter, libraries
    base = others + overhead
    floor = base + new_footprint

    def used_with(kept: list[ResidentView]) -> int:
        return base + sum(r.footprint for r in kept) + new_footprint

    def count_ok(kept: list[ResidentView]) -> bool:
        return max_models <= 0 or len(kept) + 1 <= max_models

    def ok(kept: list[ResidentView]) -> bool:
        return used_with(kept) <= limit and count_ok(kept)

    def plan(
        fits: bool,
        reason: str,
        used_after: int,
        evict: tuple[str, ...] = (),
        wait_for: tuple[str, ...] = (),
    ) -> AdmissionPlan:
        return AdmissionPlan(
            fits,
            evict=evict,
            wait_for=wait_for,
            reason=reason,
            limit=limit,
            used_now=memory.in_use,
            used_after=used_after,
            floor=floor,
        )

    if floor > limit:
        # Not even with every HFL model gone. Nothing to evict or wait for.
        return plan(False, "too_big", floor)

    kept = list(residents)
    if ok(kept):
        return plan(True, "fits", used_with(kept))

    evict: list[str] = []
    for victim in sorted(
        (r for r in residents if not r.busy and not r.mine), key=lambda r: r.last_used
    ):
        if ok(kept):
            break
        kept.remove(victim)
        evict.append(victim.name)
    if ok(kept):
        return plan(True, "evict", used_with(kept), evict=tuple(evict))

    wait: list[str] = []
    for victim in sorted((r for r in kept if r.busy and not r.mine), key=lambda r: r.last_used):
        if ok(kept):
            break
        kept.remove(victim)
        wait.append(victim.name)
    if ok(kept):
        return plan(False, "wait", used_with(kept), evict=tuple(evict), wait_for=tuple(wait))
    # Only models the loading request itself holds stand in the way.
    return plan(False, "blocked", floor, evict=tuple(evict))


def _gb(n: int) -> str:
    return f"{n / GIB:.1f} GB"


def describe_memory(memory: MemoryView, budget: float) -> str:
    """One line: how much is in use and how much the budget allows."""
    pct = 100.0 * memory.in_use / memory.total if memory.total else 0.0
    return (
        f"{_gb(memory.in_use)} of {_gb(memory.total)} in use ({pct:.0f}%), "
        f"budget {budget * 100:.0f}% = {_gb(int(memory.total * budget))}"
    )


@dataclass(frozen=True)
class StandaloneCheck:
    """Admission of one model into a process that holds no other model
    (``hfl run``, a preload before the server starts)."""

    footprint: int
    kv_known: bool
    memory: MemoryView | None
    plan: AdmissionPlan | None
    """None when the check could not run: no measurement, unknown size, or
    ``HFL_DISABLE_MEMORY_PREFLIGHT``."""
    budget: float


def check_standalone_load(model_path: str, n_ctx: int = 0) -> StandaloneCheck:
    """Would this model fit, alone, under the memory budget right now?"""
    from hfl.engine.footprint import estimate_footprint

    fp = estimate_footprint(model_path, n_ctx)
    budget = budget_fraction()
    disabled = os.environ.get("HFL_DISABLE_MEMORY_PREFLIGHT", "").lower() in ("1", "true", "yes")
    memory = current_memory()
    if disabled or memory is None or not fp.known:
        return StandaloneCheck(fp.total_bytes, fp.kv_known, memory, None, budget)
    plan = plan_admission(fp.total_bytes, memory, [], budget)
    return StandaloneCheck(fp.total_bytes, fp.kv_known, memory, plan, budget)

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
    "current_gpu_memory",
    "discrete_gpu_unmeasured",
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
    constraint: str = "ram"
    """Which memory decides a refusal: ``ram`` or ``gpu``."""
    gpu_limit: int = 0
    gpu_used_now: int = 0
    gpu_used_after: int = 0
    gpu_floor: int = 0
    """The same figures for a discrete GPU's memory, when one is measured."""


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
    gpu: MemoryView | None = None,
) -> AdmissionPlan:
    """Decide whether a model of ``new_footprint`` bytes may load.

    Order of resort, cheapest first: load alongside everything; unload idle
    models, least recently used first; wait for busy ones to finish; refuse.
    ``max_models`` > 0 adds a count ceiling (Ollama's
    ``OLLAMA_MAX_LOADED_MODELS``); 0 means memory alone decides.

    ``gpu`` is a discrete GPU's memory. With one, a model must fit BOTH: its
    weights and KV cache go to VRAM (full offload is the default), while
    the process and the OS still take RAM. The same footprint is charged to
    each — an upper bound on either. Unified memory (Apple Silicon) has no
    separate pool and passes None.
    """
    pools = [("ram", memory)] + ([("gpu", gpu)] if gpu is not None else [])
    accounted = sum(r.footprint for r in residents)

    def base_of(view: MemoryView) -> int:
        others = max(0, view.in_use - view.hfl_rss)
        overhead = max(0, view.hfl_rss - accounted)  # interpreter, libraries, CUDA context
        return others + overhead

    limits = {name: int(view.total * budget) for name, view in pools}
    bases = {name: base_of(view) for name, view in pools}
    floors = {name: bases[name] + new_footprint for name, _ in pools}

    def used_with(kept: list[ResidentView], pool: str = "ram") -> int:
        return bases[pool] + sum(r.footprint for r in kept) + new_footprint

    def count_ok(kept: list[ResidentView]) -> bool:
        return max_models <= 0 or len(kept) + 1 <= max_models

    def over(kept: list[ResidentView]) -> str | None:
        for name, _ in pools:
            if used_with(kept, name) > limits[name]:
                return name
        return None

    def ok(kept: list[ResidentView]) -> bool:
        return over(kept) is None and count_ok(kept)

    def plan(
        fits: bool,
        reason: str,
        kept: list[ResidentView] | None,
        evict: tuple[str, ...] = (),
        wait_for: tuple[str, ...] = (),
        constraint: str = "ram",
    ) -> AdmissionPlan:
        def after(pool: str) -> int:
            return floors[pool] if kept is None else used_with(kept, pool)

        return AdmissionPlan(
            fits,
            evict=evict,
            wait_for=wait_for,
            reason=reason,
            limit=limits["ram"],
            used_now=memory.in_use,
            used_after=after("ram"),
            floor=floors["ram"],
            constraint=constraint,
            gpu_limit=limits.get("gpu", 0),
            gpu_used_now=gpu.in_use if gpu is not None else 0,
            gpu_used_after=after("gpu") if gpu is not None else 0,
            gpu_floor=floors.get("gpu", 0),
        )

    for name, _ in pools:
        if floors[name] > limits[name]:
            # Not even with every HFL model gone. Nothing to evict or wait for.
            return plan(False, "too_big", None, constraint=name)

    kept = list(residents)
    if ok(kept):
        return plan(True, "fits", kept)

    evict: list[str] = []
    for victim in sorted(
        (r for r in residents if not r.busy and not r.mine), key=lambda r: r.last_used
    ):
        if ok(kept):
            break
        kept.remove(victim)
        evict.append(victim.name)
    if ok(kept):
        return plan(True, "evict", kept, evict=tuple(evict))

    wait: list[str] = []
    for victim in sorted((r for r in kept if r.busy and not r.mine), key=lambda r: r.last_used):
        if ok(kept):
            break
        kept.remove(victim)
        wait.append(victim.name)
    if ok(kept):
        return plan(False, "wait", kept, evict=tuple(evict), wait_for=tuple(wait))
    # Only models the loading request itself holds stand in the way.
    return plan(False, "blocked", None, evict=tuple(evict), constraint=over(kept) or "ram")


# ----------------------------------------------------------------------
# Discrete GPU memory
# ----------------------------------------------------------------------

_MIB = 1024**2


def _nvidia_smi(args: list[str]) -> list[list[str]] | None:
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *args, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("nvidia-smi failed: %s", exc)
        return None
    return [[cell.strip() for cell in line.split(",")] for line in out.splitlines() if line.strip()]


def current_gpu_memory() -> MemoryView | None:
    """NVIDIA memory across all GPUs, with this process's share, or None.

    Read from ``nvidia-smi`` (installed with the driver) rather than torch:
    llama.cpp's CUDA build puts models in VRAM without torch present. All
    GPUs are summed, because llama.cpp splits layers across them.
    """
    gpus = _nvidia_smi(["--query-gpu=memory.total,memory.used"])
    if not gpus:
        return None
    try:
        total = sum(int(float(row[0])) for row in gpus) * _MIB
        used = sum(int(float(row[1])) for row in gpus) * _MIB
    except (ValueError, IndexError):
        return None
    mine = 0
    for row in _nvidia_smi(["--query-compute-apps=pid,used_memory"]) or []:
        try:
            if int(row[0]) == os.getpid():
                mine += int(float(row[1])) * _MIB
        except (ValueError, IndexError):
            continue
    return MemoryView(total=total, in_use=used, hfl_rss=mine)


_UNMEASURED_GPU: bool | None = None


def discrete_gpu_unmeasured() -> bool:
    """A GPU that models may be offloaded to, whose memory we cannot read.

    ROCm, or CUDA without ``nvidia-smi``. Admission by RAM alone would then
    put several models into a VRAM it cannot see, so the caller falls back
    to one resident model. Cached: the answer does not change while the
    process runs, and probing may import torch.
    """
    global _UNMEASURED_GPU
    if _UNMEASURED_GPU is not None:
        return _UNMEASURED_GPU
    import importlib.util
    import shutil

    unmeasured = False
    if current_gpu_memory() is None:
        if shutil.which("rocm-smi") is not None:
            unmeasured = True
        elif importlib.util.find_spec("torch") is not None:
            try:
                import torch

                unmeasured = bool(torch.cuda.is_available() or getattr(torch.version, "hip", None))
            except Exception:  # pragma: no cover - broken torch install
                unmeasured = False
    _UNMEASURED_GPU = unmeasured
    return unmeasured


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
    plan = plan_admission(fp.total_bytes, memory, [], budget, gpu=current_gpu_memory())
    return StandaloneCheck(fp.total_bytes, fp.kv_known, memory, plan, budget)

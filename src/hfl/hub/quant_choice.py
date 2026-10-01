# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Which GGUF level a converted model should get on this machine.

``hfl pull`` of a safetensors LLM converts it to GGUF, and used to do so at
a fixed Q4_K_M: a 3B model that fits in F16 lost precision for nothing, and
a 70B was downloaded only to fail at load. The level is now the most
precise one that fits this machine's memory, offered before anything is
downloaded; a GPU that is too small but has RAM beside it is told that
llama.cpp will split the layers (slower); and a model that fits nowhere is
refused before the download, with how much it needs.

Sizes come from :mod:`hfl.hub.quant_table` (weights + a 4k-token KV cache
+ overhead, ×1.2): estimates, conservative on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hfl.hub.hw_profile import HardwareProfile
from hfl.hub.quant_table import estimate_vram_gb

# Most precise first. Each is a type llama-quantize accepts (F16: the
# converter's own output, nothing quantized).
LADDER = ("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")
# How far down the ladder a recommendation goes. Below Q3_K_M the loss is
# large enough that it is listed, never recommended.
LOWEST_RECOMMENDED = "Q3_K_M"
# Offered when only GPU + RAM together fit the model.
SPLIT_LEVEL = "Q4_K_M"
# The share of RAM a model may take: the system keeps the rest. (On Apple
# Silicon the profile's ``gpu_vram_gb`` is already the GPU's share.)
RAM_SHARE = 0.7


@dataclass(frozen=True)
class Level:
    name: str
    size_gb: float
    fits: bool  # in the fast memory (the GPU's, or RAM without one)
    fits_split: bool  # in GPU + RAM, llama.cpp splitting the layers


@dataclass
class Choice:
    """The levels, and which one to offer."""

    memory: str  # "cuda", "metal" or "cpu"
    fast_gb: float  # memory the model runs fastest in
    total_gb: float  # with RAM beside a discrete GPU (== fast_gb otherwise)
    levels: list[Level] = field(default_factory=list)
    recommended: str | None = None  # None: nothing fits (see ``needed_gb``)
    split: bool = False  # the recommended level runs split GPU + CPU

    @property
    def needed_gb(self) -> float:
        """What the least precise recommendable level would need."""
        return next(level.size_gb for level in self.levels if level.name == LOWEST_RECOMMENDED)


def budgets(profile: HardwareProfile) -> tuple[str, float, float]:
    """The kind of memory, what runs fast in it and what fits at all (GB)."""
    ram = max(profile.system_ram_gb or 0.0, 0.0) * RAM_SHARE
    if profile.gpu_kind == "cuda" and profile.gpu_vram_gb:
        return "cuda", profile.gpu_vram_gb, profile.gpu_vram_gb + ram
    if profile.gpu_kind == "metal":
        unified = profile.gpu_vram_gb or ram
        return "metal", unified, unified
    return "cpu", ram, ram


def choose(
    params_b: float,
    profile: HardwareProfile,
    *,
    active_params_b: float | None = None,
    n_ctx: int = 4096,
) -> Choice:
    """Every level's size here, and the most precise one that fits."""
    memory, fast, total = budgets(profile)
    choice = Choice(memory=memory, fast_gb=round(fast, 1), total_gb=round(total, 1))
    for name in LADDER:
        size = estimate_vram_gb(
            params_b=params_b,
            quantization=name,
            n_ctx=n_ctx,
            active_params_b=active_params_b,
        ).total_gb
        choice.levels.append(Level(name, size, size <= fast, size <= total))
    recommendable = LADDER[: LADDER.index(LOWEST_RECOMMENDED) + 1]
    fast_fit = [lv for lv in choice.levels if lv.fits and lv.name in recommendable]
    if fast_fit:
        choice.recommended = fast_fit[0].name
        return choice
    split_fit = [lv.name for lv in choice.levels if lv.fits_split and lv.name in recommendable]
    if split_fit:
        # Split, the most precise level that fits is not the best trade:
        # every layer left on the CPU slows each token. Q4_K_M (the usual
        # balance) when it fits, else the smallest recommendable.
        choice.recommended = SPLIT_LEVEL if SPLIT_LEVEL in split_fit else split_fit[-1]
        choice.split = True
    return choice


def conversion_hint(manifest: object) -> str | None:
    """What makes a safetensors model that does not fit load, for the
    memory error: the GGUF level that fits (as ``hfl pull`` offers it)
    and, on NVIDIA, bitsandbytes for an architecture llama.cpp cannot
    convert. None for any other model, or when nothing would help."""
    from pathlib import Path

    from hfl.hub.hw_profile import get_hw_profile

    if str(getattr(manifest, "format", "")).lower() != "safetensors":
        return None
    path = Path(str(getattr(manifest, "local_path", "")))
    files = list(path.glob("*.safetensors")) if path.is_dir() else []
    if not files:
        return None
    params_b = sum(f.stat().st_size for f in files) / 2 / 1e9  # 16-bit weights
    choice = choose(params_b, get_hw_profile())
    hints = []
    if choice.recommended:
        split = ", split between GPU and CPU (slower)" if choice.split else ""
        hints.append(
            f"Convert it to GGUF at the most precise level that fits here "
            f"({choice.recommended}{split}): hfl pull {getattr(manifest, 'repo_id', '')} "
            f"--format gguf -q {choice.recommended}."
        )
    if choice.memory == "cuda":
        hints.append(
            "If llama.cpp cannot convert its architecture, HFL_TRANSFORMERS_QUANT=8bit "
            "(or 4bit) loads it quantized with bitsandbytes."
        )
    return " ".join(hints) or None

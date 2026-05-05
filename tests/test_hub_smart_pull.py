# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl/hub/smart_pull.py`` — V4 F2.1."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from hfl.hub.hw_profile import HardwareProfile
from hfl.hub.smart_pull import build_smart_plan


@dataclass
class _FakeSibling:
    rfilename: str


@dataclass
class _FakeRepoInfo:
    siblings: list[_FakeSibling]


def _mac_mlx_profile(vram_gb: float = 16.0) -> HardwareProfile:
    return HardwareProfile(
        os="darwin",
        arch="arm64",
        system_ram_gb=vram_gb / 0.7,
        gpu_kind="metal",
        gpu_vram_gb=vram_gb,
        has_mlx=True,
        has_cuda=False,
        has_rocm=False,
    )


def _cuda_profile(vram_gb: float = 24.0) -> HardwareProfile:
    return HardwareProfile(
        os="linux",
        arch="x86_64",
        system_ram_gb=64.0,
        gpu_kind="cuda",
        gpu_vram_gb=vram_gb,
        has_mlx=False,
        has_cuda=True,
        has_rocm=False,
    )


def _cpu_profile(ram_gb: float = 16.0) -> HardwareProfile:
    return HardwareProfile(
        os="linux",
        arch="x86_64",
        system_ram_gb=ram_gb,
        gpu_kind="none",
        gpu_vram_gb=None,
        has_mlx=False,
        has_cuda=False,
        has_rocm=False,
    )


def _api_with(repos: dict[str, list[str] | None]):
    """Build an HfApi mock that knows which repos exist and what
    GGUF files they advertise.

    ``repos`` is keyed by repo_id; the value is either ``None``
    (repo doesn't exist on the Hub) or a list of file names.
    """
    api = MagicMock()

    def _model_info(repo_id):
        if repo_id not in repos or repos[repo_id] is None:
            raise RuntimeError(f"404: {repo_id}")
        return _FakeRepoInfo(siblings=[_FakeSibling(f) for f in repos[repo_id]])

    api.model_info.side_effect = _model_info
    return api


# --- Apple Silicon path -----------------------------------------------------


class TestAppleSiliconPath:
    def test_picks_mlx_4bit_when_available(self):
        api = _api_with(
            {
                "mlx-community/Llama-3.1-8B-Instruct-4bit": ["model.safetensors"],
                "meta-llama/Llama-3.1-8B-Instruct": ["model.safetensors"],
            }
        )
        plan = build_smart_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            profile=_mac_mlx_profile(vram_gb=16.0),
            api=api,
        )
        assert plan.target_repo_id == "mlx-community/Llama-3.1-8B-Instruct-4bit"
        assert plan.quantization == "mlx-4bit"

    def test_falls_back_to_8bit_when_only_8bit_exists(self):
        api = _api_with(
            {
                "mlx-community/Llama-3.1-8B-Instruct-4bit": None,
                "mlx-community/Llama-3.1-8B-Instruct-8bit": ["model.safetensors"],
                "meta-llama/Llama-3.1-8B-Instruct": ["model.safetensors"],
            }
        )
        plan = build_smart_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            profile=_mac_mlx_profile(vram_gb=24.0),
            api=api,
        )
        assert plan.target_repo_id == "mlx-community/Llama-3.1-8B-Instruct-8bit"
        assert plan.quantization == "mlx-8bit"


# --- CUDA path --------------------------------------------------------------


class TestCudaPath:
    def test_picks_q5_k_m_from_bartowski_fork(self):
        api = _api_with(
            {
                "bartowski/Llama-3.1-8B-Instruct-GGUF": [
                    "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    "Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                    "Llama-3.1-8B-Instruct-Q8_0.gguf",
                ],
                "meta-llama/Llama-3.1-8B-Instruct": ["model.safetensors"],
            }
        )
        plan = build_smart_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            profile=_cuda_profile(vram_gb=24.0),
            api=api,
        )
        assert plan.target_repo_id == "bartowski/Llama-3.1-8B-Instruct-GGUF"
        # Q5_K_M is highest-fidelity that fits 8B in 24 GB headroom.
        assert plan.quantization == "q5_k_m"

    def test_falls_back_to_q4_when_only_q4_published(self):
        api = _api_with(
            {
                "bartowski/Llama-3.1-8B-Instruct-GGUF": ["Llama-3.1-8B-Instruct-Q4_K_M.gguf"],
                "meta-llama/Llama-3.1-8B-Instruct": ["model.safetensors"],
            }
        )
        plan = build_smart_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            profile=_cuda_profile(vram_gb=24.0),
            api=api,
        )
        assert plan.quantization == "q4_k_m"


# --- Tight budget / overflow ------------------------------------------------


class TestTightBudget:
    def test_70b_overflows_8gb_budget(self):
        api = _api_with(
            {
                "bartowski/Llama-3.1-70B-Instruct-GGUF": [
                    "Llama-3.1-70B-Instruct-Q4_K_M.gguf",
                    "Llama-3.1-70B-Instruct-Q3_K_M.gguf",
                ],
                "meta-llama/Llama-3.1-70B-Instruct": ["model.safetensors"],
            }
        )
        with pytest.raises(ValueError) as exc_info:
            build_smart_plan(
                "meta-llama/Llama-3.1-70B-Instruct",
                profile=_cpu_profile(ram_gb=16.0),
                api=api,
            )
        assert "fits the" in str(exc_info.value)
        # And the message includes the candidates probed so the
        # operator sees what was tried.
        assert "Tried" in str(exc_info.value)

    def test_max_vram_gb_overrides_profile(self):
        """Operator may pass ``max_vram_gb`` to constrain below the
        detected budget — useful when sharing the GPU."""
        api = _api_with(
            {
                "bartowski/Llama-3.1-8B-Instruct-GGUF": [
                    "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    "Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                ],
                "meta-llama/Llama-3.1-8B-Instruct": ["model.safetensors"],
            }
        )
        # Without the cap a 24GB profile picks Q5; cap to 6GB and
        # nothing fits.
        with pytest.raises(ValueError):
            build_smart_plan(
                "meta-llama/Llama-3.1-8B-Instruct",
                profile=_cuda_profile(vram_gb=24.0),
                api=api,
                max_vram_gb=6.0,
            )


# --- No community fork available --------------------------------------------


class TestBaseRepoFallback:
    def test_uses_base_repo_safetensors_when_no_fork_exists(self):
        api = _api_with(
            {
                "mlx-community/Llama-3.1-8B-Instruct-4bit": None,
                "mlx-community/Llama-3.1-8B-Instruct-8bit": None,
                "bartowski/Llama-3.1-8B-Instruct-GGUF": None,
                "TheBloke/Llama-3.1-8B-Instruct-GGUF": None,
                "meta-llama/Llama-3.1-8B-Instruct": ["model.safetensors"],
            }
        )
        plan = build_smart_plan(
            "meta-llama/Llama-3.1-8B-Instruct",
            profile=_cuda_profile(vram_gb=80.0),  # generous so f16 fits
            api=api,
        )
        assert plan.target_repo_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert plan.quantization == "f16"

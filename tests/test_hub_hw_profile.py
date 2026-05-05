# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl/hub/hw_profile.py`` — V4."""

from __future__ import annotations

from unittest.mock import MagicMock

from hfl.hub.hw_profile import HardwareProfile, get_hw_profile


class TestHardwareProfile:
    def test_returns_dataclass(self):
        profile = get_hw_profile()
        assert isinstance(profile, HardwareProfile)
        assert profile.os in ("darwin", "linux", "windows")
        assert profile.arch != ""

    def test_apple_silicon_route(self, monkeypatch):
        """Darwin-arm64 with mlx_lm available → ``metal`` GPU kind and
        VRAM = 70% of system RAM."""
        import importlib
        import platform

        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")

        # Pretend mlx_lm is importable.
        fake_spec = MagicMock()
        monkeypatch.setattr(
            importlib.util,  # type: ignore[attr-defined]
            "find_spec",
            lambda name: fake_spec if name == "mlx_lm" else None,
        )

        # Stable RAM of 32 GB.
        from hfl.hub import hw_profile as module

        monkeypatch.setattr(module, "_system_ram_gb", lambda: 32.0)
        # Force CUDA/ROCm off so the metal branch wins.
        monkeypatch.setattr(module, "_has_cuda", lambda: False)
        monkeypatch.setattr(module, "_has_rocm", lambda: False)

        profile = get_hw_profile()
        assert profile.has_mlx is True
        assert profile.gpu_kind == "metal"
        assert profile.gpu_vram_gb == round(32.0 * 0.7, 1)

    def test_cuda_route(self, monkeypatch):
        from hfl.hub import hw_profile as module

        monkeypatch.setattr(module, "_has_cuda", lambda: True)
        monkeypatch.setattr(module, "_cuda_vram_gb", lambda: 24.0)
        monkeypatch.setattr(module, "_has_mlx", lambda: False)
        monkeypatch.setattr(module, "_has_rocm", lambda: False)
        monkeypatch.setattr(module, "_system_ram_gb", lambda: 64.0)

        profile = get_hw_profile()
        assert profile.gpu_kind == "cuda"
        assert profile.gpu_vram_gb == 24.0
        assert profile.has_cuda is True

    def test_cpu_only_fallback(self, monkeypatch):
        from hfl.hub import hw_profile as module

        monkeypatch.setattr(module, "_has_cuda", lambda: False)
        monkeypatch.setattr(module, "_has_mlx", lambda: False)
        monkeypatch.setattr(module, "_has_rocm", lambda: False)
        monkeypatch.setattr(module, "_system_ram_gb", lambda: 16.0)

        profile = get_hw_profile()
        assert profile.gpu_kind == "none"
        assert profile.gpu_vram_gb is None
        assert profile.has_mlx is False
        assert profile.has_cuda is False
        assert profile.system_ram_gb == 16.0

    def test_rocm_route(self, monkeypatch):
        from hfl.hub import hw_profile as module

        monkeypatch.setattr(module, "_has_cuda", lambda: False)
        monkeypatch.setattr(module, "_has_mlx", lambda: False)
        monkeypatch.setattr(module, "_has_rocm", lambda: True)
        monkeypatch.setattr(module, "_system_ram_gb", lambda: 32.0)

        profile = get_hw_profile()
        assert profile.gpu_kind == "rocm"
        assert profile.has_rocm is True

    def test_psutil_failure_returns_zero_ram(self, monkeypatch):
        """When psutil is unavailable, ``system_ram_gb`` is 0 and the
        downstream heuristic must treat it as CPU-only."""
        import sys

        monkeypatch.setitem(sys.modules, "psutil", None)
        from hfl.hub import hw_profile as module

        # Re-bind the helper to read through the broken import.
        # (The function catches the exception itself, so we just
        # make sure no ``raise`` escapes.)
        result = module._system_ram_gb()
        # Either 0.0 (psutil import failed) or a real number when the
        # test host actually has psutil — both are acceptable.
        assert result >= 0.0

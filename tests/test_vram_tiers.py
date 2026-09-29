# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for VRAM-aware context-size tiering (Phase 11 P1 — V2 row 13)."""

from __future__ import annotations

from hfl.engine import vram


class TestTierSelection:
    def test_ge_48_gib_gets_256k(self):
        tier = vram.pick_ctx_size(48.0)
        assert tier.ctx == 262144
        tier = vram.pick_ctx_size(80.0)
        assert tier.ctx == 262144

    def test_24_to_48_gib_gets_32k(self):
        assert vram.pick_ctx_size(24.0).ctx == 32768
        assert vram.pick_ctx_size(40.0).ctx == 32768
        assert vram.pick_ctx_size(47.9).ctx == 32768

    def test_lt_24_gets_4k(self):
        assert vram.pick_ctx_size(0.5).ctx == 4096
        assert vram.pick_ctx_size(16.0).ctx == 4096
        assert vram.pick_ctx_size(23.9).ctx == 4096

    def test_none_defaults_to_floor(self, monkeypatch):
        monkeypatch.setattr(vram, "detect_vram_gib", lambda: None)
        tier = vram.pick_ctx_size(None)
        assert tier.ctx == 4096
        assert tier.vram_gib is None


class TestEnvOverride:
    def test_override_wins_over_probes(self, monkeypatch):
        monkeypatch.setenv("HFL_VRAM_OVERRIDE_GIB", "40")
        monkeypatch.setattr(vram, "_probe_nvidia", lambda: 1000.0)
        monkeypatch.setattr(vram, "_probe_metal", lambda: 1000.0)
        monkeypatch.setattr(vram, "_probe_rocm", lambda: 1000.0)
        assert vram.detect_vram_gib() == 40.0

    def test_malformed_override_is_ignored(self, monkeypatch):
        monkeypatch.setenv("HFL_VRAM_OVERRIDE_GIB", "not-a-number")
        monkeypatch.setattr(vram, "_probe_nvidia", lambda: 12.0)
        monkeypatch.setattr(vram, "_probe_metal", lambda: None)
        monkeypatch.setattr(vram, "_probe_rocm", lambda: None)
        assert vram.detect_vram_gib() == 12.0


class TestProbeFallbackChain:
    def test_nvidia_preferred(self, monkeypatch):
        monkeypatch.delenv("HFL_VRAM_OVERRIDE_GIB", raising=False)
        monkeypatch.setattr(vram, "_probe_nvidia", lambda: 24.0)
        monkeypatch.setattr(vram, "_probe_metal", lambda: 48.0)
        monkeypatch.setattr(vram, "_probe_rocm", lambda: 8.0)
        assert vram.detect_vram_gib() == 24.0

    def test_metal_when_no_nvidia(self, monkeypatch):
        monkeypatch.delenv("HFL_VRAM_OVERRIDE_GIB", raising=False)
        monkeypatch.setattr(vram, "_probe_nvidia", lambda: None)
        monkeypatch.setattr(vram, "_probe_metal", lambda: 32.0)
        monkeypatch.setattr(vram, "_probe_rocm", lambda: 8.0)
        assert vram.detect_vram_gib() == 32.0

    def test_rocm_last_resort(self, monkeypatch):
        monkeypatch.delenv("HFL_VRAM_OVERRIDE_GIB", raising=False)
        monkeypatch.setattr(vram, "_probe_nvidia", lambda: None)
        monkeypatch.setattr(vram, "_probe_metal", lambda: None)
        monkeypatch.setattr(vram, "_probe_rocm", lambda: 16.0)
        assert vram.detect_vram_gib() == 16.0

    def test_all_none_returns_none(self, monkeypatch):
        monkeypatch.delenv("HFL_VRAM_OVERRIDE_GIB", raising=False)
        monkeypatch.setattr(vram, "_probe_nvidia", lambda: None)
        monkeypatch.setattr(vram, "_probe_metal", lambda: None)
        monkeypatch.setattr(vram, "_probe_rocm", lambda: None)
        assert vram.detect_vram_gib() is None

    def test_probe_exception_is_swallowed(self, monkeypatch):
        def _boom():
            raise RuntimeError("probe blew up")

        monkeypatch.delenv("HFL_VRAM_OVERRIDE_GIB", raising=False)
        monkeypatch.setattr(vram, "_probe_nvidia", _boom)
        monkeypatch.setattr(vram, "_probe_metal", lambda: 24.0)
        monkeypatch.setattr(vram, "_probe_rocm", lambda: None)
        assert vram.detect_vram_gib() == 24.0


class TestNvidiaWithoutPynvml:
    """No extra installs pynvml, so the NVIDIA probe found nothing and every
    model opened with 4096 tokens, on any card. It reads nvidia-smi then."""

    def _no_pynvml(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    def test_two_cards_read_through_nvidia_smi(self, monkeypatch):
        from hfl.engine import residency, vram

        self._no_pynvml(monkeypatch)
        monkeypatch.setattr(residency, "_nvidia_smi", lambda args: [["40960"], ["40960"]])
        assert vram._probe_nvidia() == 80.0
        assert vram.pick_ctx_size().ctx == 262144

    def test_no_nvidia_smi_is_still_nothing(self, monkeypatch):
        from hfl.engine import residency, vram

        self._no_pynvml(monkeypatch)
        monkeypatch.setattr(residency, "_nvidia_smi", lambda args: None)
        assert vram._probe_nvidia() is None

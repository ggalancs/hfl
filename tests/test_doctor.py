# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl doctor`` (Phase 15 P2 — V2 row 15)."""

from __future__ import annotations

from hfl.cli.commands import doctor


class TestBuildReport:
    def test_returns_non_empty_report(self):
        report = doctor.build_report()
        assert report.python_version
        assert report.platform_system
        assert report.recommended_ctx in (4096, 32768, 262144)

    def test_records_at_least_one_accelerator_on_darwin(self, monkeypatch):
        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(doctor, "_probe_nvidia", lambda: [])
        monkeypatch.setattr(doctor, "_probe_rocm", lambda: [])
        monkeypatch.setattr(doctor, "_probe_metal", lambda: True)
        report = doctor.build_report()
        assert report.metal_available is True

    def test_no_accelerator_emits_recommendation(self, monkeypatch):
        monkeypatch.setattr(doctor, "_probe_nvidia", lambda: [])
        monkeypatch.setattr(doctor, "_probe_rocm", lambda: [])
        monkeypatch.setattr(doctor, "_probe_metal", lambda: False)
        report = doctor.build_report()
        assert any("CPU-only" in rec for rec in report.recommendations)

    def test_mlx_recommendation_on_apple_silicon(self, monkeypatch):
        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
        monkeypatch.setattr(doctor, "_probe_optional", lambda name: False)
        report = doctor.build_report()
        assert any("MLX" in rec for rec in report.recommendations)


class TestFormatReport:
    def test_renders_all_headers(self):
        report = doctor.build_report()
        text = doctor.format_report(report)
        assert "hfl doctor" in text
        assert "Backends:" in text
        assert "Accelerators:" in text
        assert "num_ctx" in text

    def test_checkmarks_accurate(self):
        report = doctor.DoctorReport(
            python_version="3.12.3",
            platform_system="Darwin",
            platform_machine="arm64",
            llama_cpp_available=True,
            metal_available=True,
        )
        text = doctor.format_report(report)
        assert "llama-cpp       ✓" in text
        assert "Apple Metal     ✓" in text


class TestProbeOptional:
    def test_false_when_module_missing(self):
        assert doctor._probe_optional("no_such_module_xyz") is False

    def test_true_for_known_stdlib_module(self):
        assert doctor._probe_optional("json") is True

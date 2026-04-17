# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the optional sandbox module (Phase 18 P3 — V2 row 37)."""

from __future__ import annotations

from hfl.core import sandbox


class TestNoOpCases:
    def test_none_mode_is_noop(self):
        result = sandbox.apply_sandbox("none")
        assert result.applied is False
        assert result.mode == "none"

    def test_empty_mode_is_noop(self):
        result = sandbox.apply_sandbox("")
        assert result.applied is False
        assert result.mode == "none"

    def test_unknown_mode_warns_and_noop(self, caplog):
        with caplog.at_level("WARNING"):
            result = sandbox.apply_sandbox("rosetta-stone")
        assert result.applied is False
        assert result.mode == "none"


class TestLinuxSeccomp:
    def test_non_linux_platforms_report_reason(self, monkeypatch):
        monkeypatch.setattr(sandbox.platform, "system", lambda: "Darwin")
        result = sandbox.apply_sandbox("seccomp")
        assert result.applied is False
        assert result.mode == "seccomp"
        assert "not-linux" in result.reason

    def test_prctl_failure_reports_errno(self, monkeypatch):
        monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")

        class _FakeLibc:
            def prctl(self, *_args):
                return -1

        monkeypatch.setattr(sandbox.ctypes, "CDLL", lambda *_a, **_k: _FakeLibc())
        monkeypatch.setattr(sandbox.ctypes, "get_errno", lambda: 13)
        result = sandbox.apply_sandbox("seccomp")
        assert result.applied is False
        assert "prctl" in result.reason

    def test_prctl_success_without_seccomp_lib_is_partial(self, monkeypatch):
        monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")

        class _FakeLibc:
            def prctl(self, *_args):
                return 0

        monkeypatch.setattr(sandbox.ctypes, "CDLL", lambda *_a, **_k: _FakeLibc())
        # Force the pyseccomp import to fail so we cover the "NO_NEW_PRIVS only" branch.
        real = __import__

        def _deny(name, *a, **k):
            if name == "seccomp":
                raise ImportError("no seccomp")
            return real(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", _deny)

        result = sandbox.apply_sandbox("seccomp")
        assert result.applied is True
        assert "NO_NEW_PRIVS" in result.reason


class TestMacOSHint:
    def test_non_darwin_reports_reason(self, monkeypatch):
        monkeypatch.setattr(sandbox.platform, "system", lambda: "Linux")
        result = sandbox.apply_sandbox("macos")
        assert result.applied is False

    def test_darwin_returns_advisory(self, monkeypatch):
        monkeypatch.setattr(sandbox.platform, "system", lambda: "Darwin")
        result = sandbox.apply_sandbox("macos")
        assert result.applied is True
        assert "advisory" in result.reason


class TestSupportedModes:
    def test_registry(self):
        assert set(sandbox.SUPPORTED_MODES) == {"none", "seccomp", "macos"}

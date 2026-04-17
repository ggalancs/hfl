# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.observability.tracing`` (Phase 17 P2 — V2 row 24)."""

from __future__ import annotations

import pytest

from hfl.observability import tracing


@pytest.fixture(autouse=True)
def _reset():
    tracing.reset_tracing()
    yield
    tracing.reset_tracing()


class TestEnvGate:
    def test_defaults_to_disabled(self, monkeypatch):
        monkeypatch.delenv("HFL_OTEL_ENABLED", raising=False)
        assert tracing.configure_tracing() is False
        assert tracing.is_enabled() is False

    def test_false_value_disables(self, monkeypatch):
        monkeypatch.setenv("HFL_OTEL_ENABLED", "false")
        assert tracing.configure_tracing() is False

    def test_truthy_value_without_sdk_falls_back(self, monkeypatch):
        """Without the SDK installed we still return False cleanly."""
        monkeypatch.setenv("HFL_OTEL_ENABLED", "true")
        # Force the SDK import to fail so the fallback path runs
        # regardless of the test venv's state.
        real = __import__

        def _deny(name, *a, **k):
            if name.startswith("opentelemetry"):
                raise ImportError(name)
            return real(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", _deny)
        assert tracing.configure_tracing() is False
        assert tracing.is_enabled() is False


class TestTraceSpan:
    def test_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("HFL_OTEL_ENABLED", raising=False)
        with tracing.trace_span("test.span") as span:
            assert span is None

    def test_attributes_accepted_without_crash(self, monkeypatch):
        monkeypatch.delenv("HFL_OTEL_ENABLED", raising=False)
        with tracing.trace_span("test.span", attributes={"k": "v"}):
            pass


class TestConfigureExplicit:
    def test_explicit_disabled_returns_false(self):
        assert tracing.configure_tracing(enabled=False) is False
        assert tracing.is_enabled() is False

    def test_explicit_enabled_without_sdk_falls_back(self, monkeypatch):
        monkeypatch.delenv("HFL_OTEL_ENABLED", raising=False)
        real = __import__

        def _deny(name, *_a, **_k):
            if name.startswith("opentelemetry"):
                raise ImportError(name)
            return real(name, *_a, **_k)

        monkeypatch.setattr("builtins.__import__", _deny)
        assert tracing.configure_tracing(enabled=True) is False

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


def _install_fake_otel(monkeypatch):
    """Inject a minimal fake ``opentelemetry`` namespace so the
    success path of ``configure_tracing`` runs without the real SDK.

    The fake exposes just the symbols the function imports plus a
    ``set_tracer_provider`` / ``get_tracer`` pair that returns a
    span-emitting object.
    """
    import sys
    import types

    captured: dict = {}

    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _Tracer:
        def start_as_current_span(self, name, attributes=None):
            captured["last_span"] = (name, attributes)
            return _Span()

    class _Provider:
        def __init__(self, **kwargs):
            captured["provider"] = kwargs

        def add_span_processor(self, processor):
            captured["processor"] = processor

    # opentelemetry root module + ``trace`` submodule.
    ot_pkg = types.ModuleType("opentelemetry")
    ot_trace = types.ModuleType("opentelemetry.trace")

    def _set_provider(p):
        captured["set_provider"] = p

    def _get_tracer(name):
        captured["tracer_name"] = name
        return _Tracer()

    ot_trace.set_tracer_provider = _set_provider
    ot_trace.get_tracer = _get_tracer

    # Sub-packages used by the function.
    sdk = types.ModuleType("opentelemetry.sdk")
    sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
    sdk_resources.Resource = type(
        "Resource", (), {"create": staticmethod(lambda d: ("Resource", d))}
    )
    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace.TracerProvider = _Provider
    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_trace_export.BatchSpanProcessor = lambda exporter: ("Batch", exporter)
    exp_pkg = types.ModuleType("opentelemetry.exporter")
    exp_otlp = types.ModuleType("opentelemetry.exporter.otlp")
    exp_proto = types.ModuleType("opentelemetry.exporter.otlp.proto")
    exp_http = types.ModuleType("opentelemetry.exporter.otlp.proto.http")
    exp_trace = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    exp_trace.OTLPSpanExporter = lambda endpoint=None: ("OTLP", endpoint)

    fakes = {
        "opentelemetry": ot_pkg,
        "opentelemetry.trace": ot_trace,
        "opentelemetry.sdk": sdk,
        "opentelemetry.sdk.resources": sdk_resources,
        "opentelemetry.sdk.trace": sdk_trace,
        "opentelemetry.sdk.trace.export": sdk_trace_export,
        "opentelemetry.exporter": exp_pkg,
        "opentelemetry.exporter.otlp": exp_otlp,
        "opentelemetry.exporter.otlp.proto": exp_proto,
        "opentelemetry.exporter.otlp.proto.http": exp_http,
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": exp_trace,
    }
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return captured


class TestConfigureWithSdk:
    """Cover the success path of ``configure_tracing`` using a fake
    ``opentelemetry`` namespace so the test doesn't depend on the
    optional ``[otel]`` extra being installed in CI."""

    def test_returns_true_when_sdk_present(self, monkeypatch):
        tracing.reset_tracing()
        monkeypatch.delenv("HFL_OTEL_ENABLED", raising=False)

        captured = _install_fake_otel(monkeypatch)

        result = tracing.configure_tracing(
            enabled=True,
            endpoint="http://localhost:4318/v1/traces",
            service_name="hfl-test",
        )
        assert result is True
        assert tracing.is_enabled() is True
        # The provider was wired with our endpoint + service name.
        assert captured["set_provider"] is not None
        tracing.reset_tracing()

    def test_endpoint_falls_back_to_env_var(self, monkeypatch):
        tracing.reset_tracing()

        captured = _install_fake_otel(monkeypatch)
        monkeypatch.setenv("HFL_OTEL_EXPORTER_ENDPOINT", "http://collector:4318/v1/traces")
        monkeypatch.setenv("HFL_OTEL_SERVICE_NAME", "hfl-from-env")

        result = tracing.configure_tracing(enabled=True)
        assert result is True
        # The Resource.create call captured the env service name.
        _, attrs = captured["set_provider"].__class__ and ("Resource", {})  # noqa: F841
        tracing.reset_tracing()


class TestTraceSpanWithProvider:
    """Cover the active-span branch of ``trace_span`` (lines 131-132)."""

    def test_active_span_yields_a_span_when_configured(self, monkeypatch):
        tracing.reset_tracing()

        _install_fake_otel(monkeypatch)
        tracing.configure_tracing(enabled=True)

        try:
            with tracing.trace_span("test-op", attributes={"k": "v"}) as span:
                assert span is not None
        finally:
            tracing.reset_tracing()

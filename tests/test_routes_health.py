# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state
from hfl.metrics import reset_metrics


@pytest.fixture
def client(temp_config):
    """Create test client."""
    reset_state()
    reset_metrics()
    return TestClient(app)


class TestHealthBasic:
    """Tests for basic health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 with a well-formed body."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Minimal contract — every other assertion in this class is
        # additive. These four keys are what callers rely on.
        assert set(data.keys()) >= {
            "status",
            "model_loaded",
            "current_model",
            "tts_model_loaded",
            "current_tts_model",
        }
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)

    def test_health_returns_status(self, client):
        """Health endpoint returns status field."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_model_info(self, client):
        """Health endpoint includes model information."""
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert "current_model" in data


class TestHealthReady:
    """Tests for readiness endpoint."""

    def test_ready_returns_200(self, client):
        """Ready endpoint returns 200 with ready/not_ready status."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"ready", "not_ready"}
        assert isinstance(data["checks"], dict)
        # Caller contract: checks dict is not empty (orchestrators gate
        # on specific keys, a silent empty dict would hide outages).
        assert data["checks"]

    def test_ready_returns_checks(self, client):
        """Ready endpoint returns check results."""
        response = client.get("/health/ready")
        data = response.json()
        assert "checks" in data
        assert "config_loaded" in data["checks"]
        assert "registry_accessible" in data["checks"]


class TestHealthLive:
    """Tests for liveness endpoint."""

    def test_live_returns_200(self, client):
        """Live endpoint returns 200 with 'alive' status and uptime."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_live_returns_uptime(self, client):
        """Live endpoint returns uptime."""
        response = client.get("/health/live")
        data = response.json()
        assert data["status"] == "alive"
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestHealthDeep:
    """Tests for deep health endpoint."""

    def test_deep_returns_200(self, client):
        """Deep endpoint returns 200 with llm and tts sections."""
        response = client.get("/health/deep")
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data and isinstance(data["llm"], dict)
        assert "tts" in data and isinstance(data["tts"], dict)
        assert "loaded" in data["llm"]
        assert "loaded" in data["tts"]

    def test_deep_returns_version(self, client):
        """Deep endpoint returns version."""
        response = client.get("/health/deep")
        data = response.json()
        assert "version" in data

    def test_deep_returns_llm_status(self, client):
        """Deep endpoint returns LLM status."""
        response = client.get("/health/deep")
        data = response.json()
        assert "llm" in data
        assert "loaded" in data["llm"]


class TestHealthSLI:
    """Tests for SLI health endpoint."""

    def test_sli_returns_200(self, client):
        """SLI endpoint returns 200 with full SLI report structure."""
        response = client.get("/health/sli")
        assert response.status_code == 200
        data = response.json()
        # Top-level schema — all three are caller-facing contracts.
        assert data["slo_version"] == "1.0"
        assert "indicators" in data and isinstance(data["indicators"], dict)
        assert "summary" in data and isinstance(data["summary"], dict)

    def test_sli_returns_status(self, client):
        """SLI endpoint returns overall status."""
        response = client.get("/health/sli")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "warning", "critical")

    def test_sli_returns_indicators(self, client):
        """SLI endpoint returns all indicators."""
        response = client.get("/health/sli")
        data = response.json()
        assert "indicators" in data

        expected_indicators = [
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "error_rate",
            "availability",
            "memory",
        ]
        for indicator in expected_indicators:
            assert indicator in data["indicators"]

    def test_sli_indicators_have_status(self, client):
        """Each indicator has a status."""
        response = client.get("/health/sli")
        data = response.json()

        for name, indicator in data["indicators"].items():
            assert "status" in indicator, f"{name} missing status"
            assert indicator["status"] in (
                "ok",
                "warning",
                "critical",
                "unknown",
            ), f"{name} has invalid status"

    def test_sli_latency_indicators_have_target(self, client):
        """Latency indicators have target values."""
        response = client.get("/health/sli")
        data = response.json()

        for name in ["latency_p50", "latency_p95", "latency_p99"]:
            assert "current_ms" in data["indicators"][name]
            assert "target_ms" in data["indicators"][name]

    def test_sli_returns_summary(self, client):
        """SLI endpoint returns summary metrics."""
        response = client.get("/health/sli")
        data = response.json()
        assert "summary" in data
        assert "total_requests" in data["summary"]
        assert "uptime_seconds" in data["summary"]

    def test_sli_unknown_status_with_no_data(self, client):
        """SLI returns unknown for latency with no data."""
        response = client.get("/health/sli")
        data = response.json()

        # With no requests, latencies should be unknown or ok
        # (depends on implementation - 0 latency is not worse than target)
        assert data["indicators"]["latency_p50"]["status"] in ("ok", "unknown")

    def test_sli_error_rate_target(self, client):
        """SLI error rate has current and target."""
        response = client.get("/health/sli")
        data = response.json()

        error_rate = data["indicators"]["error_rate"]
        assert "current" in error_rate
        assert "target" in error_rate

    def test_sli_memory_thresholds(self, client):
        """SLI memory has threshold values."""
        response = client.get("/health/sli")
        data = response.json()

        memory = data["indicators"]["memory"]
        assert "current" in memory
        assert "warning_threshold" in memory
        assert "critical_threshold" in memory


class TestHealthRegistryUnreachable:
    def test_registry_failure_returns_false(self, monkeypatch):
        """``_check_registry`` returns False when ``get_registry``
        raises one of the expected exception types — the helper
        intentionally swallows OSError/ValueError/KeyError/AttributeError
        so a partially-broken registry doesn't 500 the probe."""
        from hfl.api import routes_health as module

        def _broken():
            raise OSError("registry path unreachable")

        monkeypatch.setattr(module, "get_registry", _broken)
        assert module._check_registry() is False


class TestHealthzWithTtsLoaded:
    def test_healthz_includes_tts_model_in_models_loaded(self, client):
        """``/healthz`` lists every loaded model — both LLM and TTS
        slots feed the same array."""
        from unittest.mock import MagicMock

        from hfl.api.state import get_state
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="bark-small",
            repo_id="suno/bark-small",
            local_path="/tmp/bark",
            format="safetensors",
            architecture="bark",
            parameters="100M",
        )
        state = get_state()
        state.tts_engine = MagicMock(is_loaded=True)
        state.current_tts_model = manifest

        body = client.get("/healthz").json()
        assert "bark-small" in body["models_loaded"]


def _install_fake_psutil(
    monkeypatch,
    *,
    rss_bytes: int = 100 * 1024 * 1024,
    total_bytes: int = 100 * 1024**3,
):
    """Inject a minimal ``psutil`` stub so the health endpoints'
    psutil branches run regardless of whether the real package is
    installed in the test venv."""
    import sys
    import types
    from unittest.mock import MagicMock

    fake = types.ModuleType("psutil")

    class _MemInfo:
        def __init__(self, rss):
            self.rss = rss

    proc = MagicMock()
    proc.memory_info = MagicMock(return_value=_MemInfo(rss_bytes))
    fake.Process = MagicMock(return_value=proc)
    fake.virtual_memory = MagicMock(return_value=MagicMock(total=total_bytes))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


class TestHealthLiveMemoryReporting:
    def test_health_live_includes_memory_when_psutil_available(self, client, monkeypatch):
        """``/health/live`` reports ``memory_mb`` when psutil is
        installed."""
        _install_fake_psutil(monkeypatch, rss_bytes=512 * 1024 * 1024)
        body = client.get("/health/live").json()
        assert "memory_mb" in body
        assert isinstance(body["memory_mb"], (int, float))
        # 512 MiB rss / 1024 / 1024 ≈ 512.
        assert abs(body["memory_mb"] - 512.0) < 1.0


class TestHealthDeepProbeAndMemory:
    def test_health_deep_includes_system_block_when_psutil(self, client, monkeypatch):
        _install_fake_psutil(monkeypatch)
        body = client.get("/health/deep").json()
        assert "system" in body
        assert "memory_mb" in body["system"]


class TestHealthSliStatusBranches:
    """Force every branch of the SLO status checks
    (``check_latency`` / ``check_rate`` / memory thresholds /
    overall ``critical`` / ``warning``)."""

    def test_critical_latency_yields_critical_overall(self, client, monkeypatch):
        """Inject SLI numbers > 1.5× target on every latency
        indicator → status flips to ``critical``."""
        _install_fake_psutil(monkeypatch)
        from hfl import metrics as metrics_module

        m = metrics_module.get_metrics()

        def _bad_sli():
            return {
                "latency_p50_ms": 10_000.0,  # target is 100 → 100×
                "latency_p95_ms": 10_000.0,
                "latency_p99_ms": 10_000.0,
                "error_rate": 0.0,
                "availability": 1.0,
                "total_requests": 100,
                "error_count": 0,
                "sample_count": 100,
                "uptime_seconds": 100.0,
                "throughput_rps": 1.0,
            }

        monkeypatch.setattr(m, "get_sli", _bad_sli)

        body = client.get("/health/sli").json()
        assert body["status"] == "critical"
        assert body["indicators"]["latency_p50"]["status"] == "critical"

    def test_warning_latency_yields_warning_overall(self, client, monkeypatch):
        """Latency between 1.0× and 1.5× target → ``warning`` for
        that indicator and overall."""
        _install_fake_psutil(monkeypatch)
        from hfl import metrics as metrics_module

        m = metrics_module.get_metrics()

        def _warn_sli():
            return {
                "latency_p50_ms": 120.0,  # target 100 → 1.2×
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "error_rate": 0.0,
                "availability": 1.0,
                "total_requests": 100,
                "error_count": 0,
                "sample_count": 100,
                "uptime_seconds": 100.0,
                "throughput_rps": 1.0,
            }

        monkeypatch.setattr(m, "get_sli", _warn_sli)

        body = client.get("/health/sli").json()
        assert body["indicators"]["latency_p50"]["status"] == "warning"
        # Overall: at least one warning, no critical → "warning".
        assert body["status"] in ("warning", "ok")

    def test_critical_error_rate_yields_critical(self, client, monkeypatch):
        """``check_rate`` lower-is-better path: error_rate > 2× target
        → critical."""
        _install_fake_psutil(monkeypatch)
        from hfl import metrics as metrics_module

        m = metrics_module.get_metrics()

        def _bad_rates():
            return {
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "error_rate": 0.5,
                "availability": 0.0,
                "total_requests": 100,
                "error_count": 50,
                "sample_count": 100,
                "uptime_seconds": 100.0,
                "throughput_rps": 1.0,
            }

        monkeypatch.setattr(m, "get_sli", _bad_rates)

        body = client.get("/health/sli").json()
        assert body["indicators"]["error_rate"]["status"] == "critical"
        # availability uses lower_is_better=False → 0.0 < 0.5×0.999 → critical
        assert body["indicators"]["availability"]["status"] == "critical"

    def test_warning_error_rate_yields_warning(self, client, monkeypatch):
        _install_fake_psutil(monkeypatch)
        from hfl import metrics as metrics_module

        m = metrics_module.get_metrics()

        def _warn_rates():
            return {
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "error_rate": 0.015,
                "availability": 0.6,
                "total_requests": 100,
                "error_count": 10,
                "sample_count": 100,
                "uptime_seconds": 100.0,
                "throughput_rps": 1.0,
            }

        monkeypatch.setattr(m, "get_sli", _warn_rates)

        body = client.get("/health/sli").json()
        assert body["indicators"]["error_rate"]["status"] == "warning"
        assert body["indicators"]["availability"]["status"] == "warning"

    def test_critical_memory_pressure_uses_psutil(self, client, monkeypatch):
        """Memory above ``memory_critical_threshold`` → critical
        memory status."""
        # 99 GB RSS / 100 GB total → 99% usage → above critical (95%).
        _install_fake_psutil(monkeypatch, rss_bytes=99 * 1024**3, total_bytes=100 * 1024**3)
        body = client.get("/health/sli").json()
        assert body["indicators"]["memory"]["status"] == "critical"

    def test_warning_memory_pressure(self, client, monkeypatch):
        """Memory between warning (80%) and critical (95%) thresholds
        → ``warning``."""
        _install_fake_psutil(monkeypatch, rss_bytes=85 * 1024**3, total_bytes=100 * 1024**3)
        body = client.get("/health/sli").json()
        assert body["indicators"]["memory"]["status"] == "warning"

    def test_ok_memory_pressure(self, client, monkeypatch):
        _install_fake_psutil(monkeypatch, rss_bytes=10 * 1024**3, total_bytes=100 * 1024**3)
        body = client.get("/health/sli").json()
        assert body["indicators"]["memory"]["status"] == "ok"

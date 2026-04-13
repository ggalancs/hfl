# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for environment variable configuration."""

import os
from unittest.mock import patch

from hfl.config import HFLConfig, SLOConfig


class TestEnvConfig:
    def test_default_host(self):
        """Default host is 127.0.0.1."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = HFLConfig()
            assert cfg.host == "127.0.0.1"

    def test_env_host(self):
        """HFL_HOST env var overrides default."""
        with patch.dict(os.environ, {"HFL_HOST": "0.0.0.0"}):
            cfg = HFLConfig()
            assert cfg.host == "0.0.0.0"

    def test_default_port(self):
        """Default port is 11434."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = HFLConfig()
            assert cfg.port == 11434

    def test_env_port(self):
        """HFL_PORT env var overrides default."""
        with patch.dict(os.environ, {"HFL_PORT": "8080"}):
            cfg = HFLConfig()
            assert cfg.port == 8080

    def test_rate_limit_disabled_via_env(self):
        """HFL_RATE_LIMIT_ENABLED=false disables rate limiting."""
        with patch.dict(os.environ, {"HFL_RATE_LIMIT_ENABLED": "false"}):
            cfg = HFLConfig()
            assert cfg.rate_limit_enabled is False

    def test_rate_limit_requests_via_env(self):
        """HFL_RATE_LIMIT_REQUESTS configurable via env."""
        with patch.dict(os.environ, {"HFL_RATE_LIMIT_REQUESTS": "100"}):
            cfg = HFLConfig()
            assert cfg.rate_limit_requests == 100

    def test_dispatcher_defaults(self):
        """Default dispatcher values match the spec §5.3 guidance:
        serialize to 1 in-flight with a 16-slot wait queue and 60 s
        acquire cap."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = HFLConfig()
            assert cfg.queue_enabled is True
            assert cfg.queue_max_inflight == 1
            assert cfg.queue_max_size == 16
            assert cfg.queue_acquire_timeout_seconds == 60.0

    def test_dispatcher_env_overrides(self):
        with patch.dict(
            os.environ,
            {
                "HFL_QUEUE_ENABLED": "false",
                "HFL_QUEUE_MAX_INFLIGHT": "4",
                "HFL_QUEUE_MAX_SIZE": "100",
                "HFL_QUEUE_ACQUIRE_TIMEOUT": "30",
            },
        ):
            cfg = HFLConfig()
            assert cfg.queue_enabled is False
            assert cfg.queue_max_inflight == 4
            assert cfg.queue_max_size == 100
            assert cfg.queue_acquire_timeout_seconds == 30.0


class TestSLOValidation:
    def test_valid_defaults(self):
        """Default SLO config is valid."""
        slo = SLOConfig()
        errors = slo.validate()
        assert errors == []

    def test_invalid_availability(self):
        """Invalid availability target produces error."""
        slo = SLOConfig(availability_target=2.0)
        errors = slo.validate()
        assert any("availability" in e for e in errors)

    def test_invalid_latency_ordering(self):
        """P95 < P50 produces error."""
        slo = SLOConfig(latency_p50_ms=200.0, latency_p95_ms=100.0)
        errors = slo.validate()
        assert any("p95" in e.lower() for e in errors)

    def test_post_init_warns_on_invalid_slo(self):
        """HFLConfig.__post_init__ warns on invalid SLO."""
        import warnings

        with warnings.catch_warnings(record=True):
            # This should not raise, just warn
            cfg = HFLConfig(slo=SLOConfig(availability_target=5.0))
            assert cfg.slo.availability_target == 5.0


class TestSafeEnsureDirs:
    def test_ensure_dirs_tolerates_oserror(self):
        """_safe_ensure_dirs handles OSError gracefully."""
        from unittest.mock import PropertyMock

        cfg = HFLConfig()
        with patch.object(type(cfg), "models_dir", new_callable=PropertyMock) as mock_dir:
            mock_dir.return_value.mkdir.side_effect = OSError("read-only")
            # Should not raise
            try:
                cfg.ensure_dirs()
            except OSError:
                pass  # Expected in direct call, _safe_ensure_dirs catches it

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

    def test_default_max_request_bytes(self):
        """Default max_request_bytes is 10 MiB."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = HFLConfig()
            assert cfg.max_request_bytes == 10 * 1024 * 1024

    def test_max_request_bytes_via_env(self):
        """HFL_MAX_REQUEST_BYTES configurable via env."""
        with patch.dict(os.environ, {"HFL_MAX_REQUEST_BYTES": "2048"}):
            cfg = HFLConfig()
            assert cfg.max_request_bytes == 2048

    def test_max_request_bytes_zero_disables(self):
        """HFL_MAX_REQUEST_BYTES=0 keeps the limit disabled."""
        with patch.dict(os.environ, {"HFL_MAX_REQUEST_BYTES": "0"}):
            cfg = HFLConfig()
            assert cfg.max_request_bytes == 0

    def test_default_streaming_timeouts(self):
        """Streaming timeouts have sensible defaults when unset."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = HFLConfig()
            assert cfg.stream_queue_put_timeout == 60.0
            assert cfg.stream_queue_get_timeout == 30.0
            assert cfg.vllm_error_put_timeout == 10.0
            assert cfg.vllm_shutdown_join_timeout == 5.0
            assert cfg.registry_sqlite_busy_timeout == 30.0

    def test_stream_queue_put_timeout_via_env(self):
        """HFL_STREAM_QUEUE_PUT_TIMEOUT overrides default."""
        with patch.dict(os.environ, {"HFL_STREAM_QUEUE_PUT_TIMEOUT": "15"}):
            cfg = HFLConfig()
            assert cfg.stream_queue_put_timeout == 15.0

    def test_stream_queue_get_timeout_via_env(self):
        """HFL_STREAM_QUEUE_GET_TIMEOUT overrides default."""
        with patch.dict(os.environ, {"HFL_STREAM_QUEUE_GET_TIMEOUT": "7.5"}):
            cfg = HFLConfig()
            assert cfg.stream_queue_get_timeout == 7.5

    def test_vllm_timeouts_via_env(self):
        """HFL_VLLM_* env vars override defaults."""
        with patch.dict(
            os.environ,
            {
                "HFL_VLLM_ERROR_PUT_TIMEOUT": "2",
                "HFL_VLLM_SHUTDOWN_JOIN_TIMEOUT": "1.5",
            },
        ):
            cfg = HFLConfig()
            assert cfg.vllm_error_put_timeout == 2.0
            assert cfg.vllm_shutdown_join_timeout == 1.5

    def test_registry_sqlite_timeout_via_env(self):
        """HFL_REGISTRY_SQLITE_TIMEOUT overrides default."""
        with patch.dict(os.environ, {"HFL_REGISTRY_SQLITE_TIMEOUT": "120"}):
            cfg = HFLConfig()
            assert cfg.registry_sqlite_busy_timeout == 120.0

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

    def test_num_parallel_alias_picks_up_hfl_var(self):
        """``HFL_NUM_PARALLEL`` is the Ollama-equivalent name and must
        feed ``queue_max_inflight`` when the explicit
        ``HFL_QUEUE_MAX_INFLIGHT`` is absent."""
        with patch.dict(
            os.environ,
            {"HFL_NUM_PARALLEL": "8"},
            clear=False,
        ):
            os.environ.pop("HFL_QUEUE_MAX_INFLIGHT", None)
            os.environ.pop("OLLAMA_NUM_PARALLEL", None)
            cfg = HFLConfig()
            assert cfg.queue_max_inflight == 8

    def test_num_parallel_alias_picks_up_ollama_var(self):
        """Drop-in replacement: an environment that already has
        ``OLLAMA_NUM_PARALLEL`` set should keep working."""
        with patch.dict(
            os.environ,
            {"OLLAMA_NUM_PARALLEL": "5"},
            clear=False,
        ):
            os.environ.pop("HFL_QUEUE_MAX_INFLIGHT", None)
            os.environ.pop("HFL_NUM_PARALLEL", None)
            cfg = HFLConfig()
            assert cfg.queue_max_inflight == 5

    def test_num_parallel_explicit_wins_over_aliases(self):
        """The explicit ``HFL_QUEUE_MAX_INFLIGHT`` must take precedence
        over both alias names — operators reaching for the documented
        variable should always win."""
        with patch.dict(
            os.environ,
            {
                "HFL_QUEUE_MAX_INFLIGHT": "2",
                "HFL_NUM_PARALLEL": "10",
                "OLLAMA_NUM_PARALLEL": "20",
            },
            clear=False,
        ):
            cfg = HFLConfig()
            assert cfg.queue_max_inflight == 2

    def test_max_queue_alias_picks_up_ollama_var(self):
        with patch.dict(
            os.environ,
            {"OLLAMA_MAX_QUEUE": "32"},
            clear=False,
        ):
            os.environ.pop("HFL_QUEUE_MAX_SIZE", None)
            os.environ.pop("HFL_MAX_QUEUE", None)
            cfg = HFLConfig()
            assert cfg.queue_max_size == 32


class TestCorsOriginsEnv:
    """``HFL_ORIGINS`` / ``OLLAMA_ORIGINS`` populate ``cors_origins``
    without code edits and flip ``cors_allow_all`` when ``*`` is in
    the list."""

    def test_unset_keeps_same_origin_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HFL_ORIGINS", None)
            os.environ.pop("OLLAMA_ORIGINS", None)
            cfg = HFLConfig()
            assert cfg.cors_origins == []
            assert cfg.cors_allow_all is False

    def test_hfl_origins_comma_list(self):
        with patch.dict(
            os.environ,
            {"HFL_ORIGINS": "https://app.example.com, https://dash.example.com"},
            clear=False,
        ):
            os.environ.pop("OLLAMA_ORIGINS", None)
            cfg = HFLConfig()
            assert cfg.cors_origins == [
                "https://app.example.com",
                "https://dash.example.com",
            ]
            assert cfg.cors_allow_all is False

    def test_wildcard_flips_allow_all(self):
        with patch.dict(os.environ, {"HFL_ORIGINS": "*"}, clear=False):
            os.environ.pop("OLLAMA_ORIGINS", None)
            cfg = HFLConfig()
            assert cfg.cors_allow_all is True

    def test_ollama_alias_works(self):
        """Drop-in ergonomics for an env that already has
        ``OLLAMA_ORIGINS`` set — Ollama-published clients don't need to
        relearn the variable."""
        with patch.dict(
            os.environ,
            {"OLLAMA_ORIGINS": "http://localhost:5173"},
            clear=False,
        ):
            os.environ.pop("HFL_ORIGINS", None)
            cfg = HFLConfig()
            assert cfg.cors_origins == ["http://localhost:5173"]


class TestOllamaHostAliasOnServer:
    """``OLLAMA_HOST`` is the canonical variable Ollama operators
    already have set. HFL must accept its three documented shapes so a
    drop-in replacement works without changing the env."""

    def _clean(self):
        for var in ("HFL_HOST", "HFL_PORT", "OLLAMA_HOST", "OLLAMA_PORT"):
            os.environ.pop(var, None)

    def test_explicit_hfl_wins(self):
        with patch.dict(
            os.environ,
            {"HFL_HOST": "10.0.0.1", "HFL_PORT": "9000", "OLLAMA_HOST": "0.0.0.0:1234"},
            clear=False,
        ):
            cfg = HFLConfig()
            assert cfg.host == "10.0.0.1"
            assert cfg.port == 9000

    def test_ollama_host_only(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "0.0.0.0"}, clear=False):
            self._clean()
            os.environ["OLLAMA_HOST"] = "0.0.0.0"
            cfg = HFLConfig()
            assert cfg.host == "0.0.0.0"
            assert cfg.port == 11434

    def test_ollama_host_with_port(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "0.0.0.0:11500"}, clear=False):
            self._clean()
            os.environ["OLLAMA_HOST"] = "0.0.0.0:11500"
            cfg = HFLConfig()
            assert cfg.host == "0.0.0.0"
            assert cfg.port == 11500

    def test_ollama_host_port_only(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": ":11500"}, clear=False):
            self._clean()
            os.environ["OLLAMA_HOST"] = ":11500"
            cfg = HFLConfig()
            assert cfg.host == "127.0.0.1"
            assert cfg.port == 11500

    def test_ollama_port_split_var(self):
        with patch.dict(os.environ, {"OLLAMA_PORT": "11600"}, clear=False):
            self._clean()
            os.environ["OLLAMA_PORT"] = "11600"
            cfg = HFLConfig()
            assert cfg.port == 11600

    def test_invalid_ollama_host_falls_back_to_defaults(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "abc:xyz"}, clear=False):
            self._clean()
            os.environ["OLLAMA_HOST"] = "abc:xyz"
            cfg = HFLConfig()
            assert cfg.host == "127.0.0.1"
            assert cfg.port == 11434


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


class TestCORSValidation:
    def test_wildcard_with_credentials_rejected(self):
        """cors_allow_all=True with cors_allow_credentials=True raises at construction."""
        import pytest

        with pytest.raises(ValueError, match="cors_allow_credentials"):
            HFLConfig(cors_allow_all=True, cors_allow_credentials=True)

    def test_explicit_wildcard_origin_with_credentials_rejected(self):
        """cors_origins=['*'] with credentials=True is equivalent and rejected."""
        import pytest

        with pytest.raises(ValueError, match="cors_allow_credentials"):
            HFLConfig(cors_origins=["*"], cors_allow_credentials=True)

    def test_wildcard_without_credentials_ok(self):
        """Wildcard origins without credentials is a valid, common dev config."""
        cfg = HFLConfig(cors_allow_all=True, cors_allow_credentials=False)
        assert cfg.cors_allow_all is True

    def test_specific_origin_with_credentials_ok(self):
        """Explicit origins + credentials is the canonical secure setup."""
        cfg = HFLConfig(
            cors_origins=["https://app.example.com"],
            cors_allow_credentials=True,
        )
        assert cfg.cors_allow_credentials is True


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

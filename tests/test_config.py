# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the configuration module."""

from pathlib import Path

from hfl.config import SLOConfig


class TestSLOConfig:
    """Tests for SLOConfig."""

    def test_default_values(self):
        """Test default SLO values are reasonable."""
        slo = SLOConfig()

        assert slo.availability_target == 0.999
        assert slo.latency_p50_ms == 100.0
        assert slo.latency_p95_ms == 500.0
        assert slo.latency_p99_ms == 1000.0
        assert slo.error_rate_target == 0.01
        assert slo.min_throughput_rps == 0.0
        assert slo.memory_warning_threshold == 0.8
        assert slo.memory_critical_threshold == 0.95

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        slo = SLOConfig()
        errors = slo.validate()
        assert errors == []

    def test_validate_invalid_availability(self):
        """Test validation fails for invalid availability."""
        slo = SLOConfig(availability_target=1.5)
        errors = slo.validate()
        assert any("availability_target" in e for e in errors)

    def test_validate_negative_availability(self):
        """Test validation fails for negative availability."""
        slo = SLOConfig(availability_target=-0.1)
        errors = slo.validate()
        assert any("availability_target" in e for e in errors)

    def test_validate_invalid_error_rate(self):
        """Test validation fails for invalid error rate."""
        slo = SLOConfig(error_rate_target=2.0)
        errors = slo.validate()
        assert any("error_rate_target" in e for e in errors)

    def test_validate_negative_latency(self):
        """Test validation fails for negative latency."""
        slo = SLOConfig(latency_p50_ms=-10)
        errors = slo.validate()
        assert any("latency_p50_ms" in e for e in errors)

    def test_validate_latency_ordering(self):
        """Test validation fails when latencies are out of order."""
        # P95 < P50
        slo = SLOConfig(latency_p50_ms=500, latency_p95_ms=100)
        errors = slo.validate()
        assert any("latency_p95_ms" in e for e in errors)

        # P99 < P95
        slo = SLOConfig(latency_p95_ms=500, latency_p99_ms=100)
        errors = slo.validate()
        assert any("latency_p99_ms" in e for e in errors)

    def test_validate_memory_thresholds(self):
        """Test validation fails for invalid memory thresholds."""
        # Warning >= Critical
        slo = SLOConfig(memory_warning_threshold=0.95, memory_critical_threshold=0.80)
        errors = slo.validate()
        assert any("memory_warning_threshold" in e for e in errors)

    def test_validate_memory_threshold_out_of_range(self):
        """Test validation fails for out of range memory thresholds."""
        slo = SLOConfig(memory_warning_threshold=1.5)
        errors = slo.validate()
        assert any("memory_warning_threshold" in e for e in errors)

    def test_custom_slo_values(self):
        """Test creating SLO with custom values."""
        slo = SLOConfig(
            availability_target=0.9999,
            latency_p50_ms=50,
            latency_p95_ms=200,
            latency_p99_ms=500,
            error_rate_target=0.001,
        )

        assert slo.availability_target == 0.9999
        assert slo.latency_p50_ms == 50
        assert slo.error_rate_target == 0.001

        errors = slo.validate()
        assert errors == []


class TestHFLConfig:
    """Tests for HFLConfig."""

    def test_default_home_dir(self, temp_dir, monkeypatch):
        """Verifies the default home directory."""
        monkeypatch.delenv("HFL_HOME", raising=False)

        from hfl.config import HFLConfig

        config = HFLConfig()

        assert config.home_dir == Path.home() / ".hfl"

    def test_custom_home_dir_from_env(self, temp_dir, monkeypatch):
        """Verifies that HFL_HOME overrides the directory."""
        monkeypatch.setenv("HFL_HOME", str(temp_dir))

        from hfl.config import HFLConfig

        config = HFLConfig()

        assert config.home_dir == temp_dir

    def test_models_dir_property(self, temp_config):
        """Verifies the models_dir property."""
        assert temp_config.models_dir == temp_config.home_dir / "models"

    def test_cache_dir_property(self, temp_config):
        """Verifies the cache_dir property."""
        assert temp_config.cache_dir == temp_config.home_dir / "cache"

    def test_registry_path_property(self, temp_config):
        """Verifies the registry_path property."""
        assert temp_config.registry_path == temp_config.home_dir / "models.json"

    def test_llama_cpp_dir_property(self, temp_config):
        """Verifies the llama_cpp_dir property."""
        expected = temp_config.home_dir / "tools" / "llama.cpp"
        assert temp_config.llama_cpp_dir == expected

    def test_default_server_settings(self, temp_config):
        """Verifies default server configuration."""
        assert temp_config.host == "127.0.0.1"
        assert temp_config.port == 11434

    def test_default_inference_settings(self, temp_config):
        """Verifies default inference configuration."""
        assert temp_config.default_ctx_size == 4096
        assert temp_config.default_n_gpu_layers == -1
        assert temp_config.default_threads == 0

    def test_hf_token_from_env(self, temp_dir, monkeypatch):
        """Verifies that HF_TOKEN is read from environment."""
        monkeypatch.setenv("HF_TOKEN", "test-token-123")

        from hfl.config import HFLConfig

        config = HFLConfig()

        assert config.hf_token == "test-token-123"

    def test_hf_token_none_when_not_set(self, temp_dir, monkeypatch):
        """Verifies that hf_token is None if not configured."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        from hfl.config import HFLConfig

        config = HFLConfig()

        assert config.hf_token is None

    def test_ensure_dirs_creates_directories(self, temp_dir):
        """Verifies that ensure_dirs creates necessary directories."""
        from hfl.config import HFLConfig

        config = HFLConfig(home_dir=temp_dir / "new_home")
        config.ensure_dirs()

        assert config.models_dir.exists()
        assert config.cache_dir.exists()
        assert config.registry_path.exists()

    def test_ensure_dirs_creates_empty_registry(self, temp_dir):
        """Verifies that ensure_dirs creates an empty registry."""
        from hfl.config import HFLConfig

        config = HFLConfig(home_dir=temp_dir / "new_home")
        config.ensure_dirs()

        content = config.registry_path.read_text()
        assert content == "[]"

    def test_ensure_dirs_preserves_existing_registry(self, temp_dir):
        """Verifies that ensure_dirs does not overwrite an existing registry."""
        from hfl.config import HFLConfig

        config = HFLConfig(home_dir=temp_dir)
        config.models_dir.mkdir(parents=True)
        config.cache_dir.mkdir(parents=True)

        # Create registry with data
        config.registry_path.write_text('[{"name": "existing"}]')

        config.ensure_dirs()

        content = config.registry_path.read_text()
        assert "existing" in content

    def test_slo_config_integration(self, temp_config):
        """Verifies that HFLConfig includes SLO configuration."""
        assert hasattr(temp_config, "slo")
        assert isinstance(temp_config.slo, SLOConfig)
        # Validate default SLO
        errors = temp_config.slo.validate()
        assert errors == []

    def test_custom_slo_in_config(self, temp_dir):
        """Verifies that custom SLO can be set in HFLConfig."""
        from hfl.config import HFLConfig

        custom_slo = SLOConfig(
            availability_target=0.99,
            latency_p50_ms=200,
        )
        config = HFLConfig(home_dir=temp_dir, slo=custom_slo)

        assert config.slo.availability_target == 0.99
        assert config.slo.latency_p50_ms == 200

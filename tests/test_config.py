# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the configuration module."""

from pathlib import Path


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

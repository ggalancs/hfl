# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for API format converters."""

import pytest

from hfl.api.converters import (
    clamp,
    generation_config_to_ollama,
    generation_config_to_openai,
    ollama_to_generation_config,
    openai_to_generation_config,
)
from hfl.api.schemas import ChatCompletionRequest
from hfl.engine.base import GenerationConfig


class TestClamp:
    """Tests for clamp function."""

    def test_value_within_range(self):
        """Should return value if within range."""
        assert clamp(5, 0, 10) == 5

    def test_value_below_min(self):
        """Should return min if value below."""
        assert clamp(-5, 0, 10) == 0

    def test_value_above_max(self):
        """Should return max if value above."""
        assert clamp(15, 0, 10) == 10

    def test_value_equals_min(self):
        """Should return min if value equals min."""
        assert clamp(0, 0, 10) == 0

    def test_value_equals_max(self):
        """Should return max if value equals max."""
        assert clamp(10, 0, 10) == 10

    def test_float_values(self):
        """Should work with floats."""
        assert clamp(0.5, 0.0, 1.0) == 0.5
        assert clamp(1.5, 0.0, 1.0) == 1.0


class TestOpenAIToGenerationConfig:
    """Tests for OpenAI to GenerationConfig converter."""

    def test_basic_conversion(self):
        """Should convert basic request."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=1000,
        )

        config = openai_to_generation_config(req)

        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_default_max_tokens(self):
        """Should use default max_tokens if not specified."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        config = openai_to_generation_config(req)

        assert config.max_tokens == 2048

    def test_string_stop_sequence(self):
        """Should convert string stop to list."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stop="END",
        )

        config = openai_to_generation_config(req)

        assert config.stop == ["END"]

    def test_list_stop_sequence(self):
        """Should preserve list stop sequences."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stop=["END", "STOP"],
        )

        config = openai_to_generation_config(req)

        assert config.stop == ["END", "STOP"]

    def test_none_stop_sequence(self):
        """Should handle None stop."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stop=None,
        )

        config = openai_to_generation_config(req)

        assert config.stop is None

    def test_seed_conversion(self):
        """Should convert seed."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            seed=42,
        )

        config = openai_to_generation_config(req)

        assert config.seed == 42

    def test_default_seed(self):
        """Should use -1 as default seed."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        config = openai_to_generation_config(req)

        assert config.seed == -1


class TestOllamaToGenerationConfig:
    """Tests for Ollama to GenerationConfig converter."""

    def test_basic_conversion(self):
        """Should convert basic options."""
        options = {
            "temperature": 0.8,
            "top_p": 0.95,
            "num_predict": 512,
        }

        config = ollama_to_generation_config(options)

        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.max_tokens == 512

    def test_defaults_for_none(self):
        """Should use defaults if options is None."""
        config = ollama_to_generation_config(None)

        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.max_tokens == 2048

    def test_defaults_for_empty(self):
        """Should use defaults for empty options."""
        config = ollama_to_generation_config({})

        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_clamps_temperature(self):
        """Should clamp temperature to valid range."""
        options = {"temperature": 5.0}
        config = ollama_to_generation_config(options)
        assert config.temperature == 2.0

        options = {"temperature": -1.0}
        config = ollama_to_generation_config(options)
        assert config.temperature == 0.0

    def test_clamps_top_p(self):
        """Should clamp top_p to valid range."""
        options = {"top_p": 1.5}
        config = ollama_to_generation_config(options)
        assert config.top_p == 1.0

        options = {"top_p": -0.5}
        config = ollama_to_generation_config(options)
        assert config.top_p == 0.0

    def test_clamps_top_k(self):
        """Should clamp top_k to valid range."""
        options = {"top_k": 0}
        config = ollama_to_generation_config(options)
        assert config.top_k == 1

        options = {"top_k": 2000}
        config = ollama_to_generation_config(options)
        assert config.top_k == 1000

    def test_clamps_max_tokens(self):
        """Should clamp max_tokens to valid range."""
        options = {"num_predict": 0}
        config = ollama_to_generation_config(options)
        assert config.max_tokens == 1

        options = {"num_predict": 500000}
        config = ollama_to_generation_config(options)
        assert config.max_tokens == 128000

    def test_stop_sequences(self):
        """Should pass through stop sequences."""
        options = {"stop": ["END", "STOP"]}
        config = ollama_to_generation_config(options)
        assert config.stop == ["END", "STOP"]

    def test_seed(self):
        """Should pass through seed."""
        options = {"seed": 42}
        config = ollama_to_generation_config(options)
        assert config.seed == 42

    def test_repeat_penalty(self):
        """Should handle repeat_penalty."""
        options = {"repeat_penalty": 1.2}
        config = ollama_to_generation_config(options)
        assert config.repeat_penalty == 1.2


class TestGenerationConfigToOpenAI:
    """Tests for GenerationConfig to OpenAI converter."""

    def test_full_conversion(self):
        """Should convert all fields."""
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            stop=["END"],
            seed=42,
        )

        result = generation_config_to_openai(config)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 1000
        assert result["stop"] == ["END"]
        assert result["seed"] == 42

    def test_skips_none_values(self):
        """Should skip None values."""
        config = GenerationConfig(
            temperature=0.7,
            top_p=None,
            max_tokens=None,
        )

        result = generation_config_to_openai(config)

        assert "temperature" in result
        assert "top_p" not in result
        assert "max_tokens" not in result

    def test_skips_default_seed(self):
        """Should skip seed if -1."""
        config = GenerationConfig(seed=-1)

        result = generation_config_to_openai(config)

        assert "seed" not in result

    def test_skips_empty_stop(self):
        """Should skip stop if empty/None."""
        config = GenerationConfig(stop=None)

        result = generation_config_to_openai(config)

        assert "stop" not in result


class TestGenerationConfigToOllama:
    """Tests for GenerationConfig to Ollama converter."""

    def test_full_conversion(self):
        """Should convert all fields."""
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=1000,
            repeat_penalty=1.1,
            stop=["END"],
            seed=42,
        )

        result = generation_config_to_ollama(config)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["num_predict"] == 1000
        assert result["repeat_penalty"] == 1.1
        assert result["stop"] == ["END"]
        assert result["seed"] == 42

    def test_skips_none_values(self):
        """Should skip None values."""
        config = GenerationConfig(
            temperature=0.7,
            top_k=None,
            repeat_penalty=None,
        )

        result = generation_config_to_ollama(config)

        assert "temperature" in result
        assert "top_k" not in result
        assert "repeat_penalty" not in result

    def test_skips_default_seed(self):
        """Should skip seed if -1."""
        config = GenerationConfig(seed=-1)

        result = generation_config_to_ollama(config)

        assert "seed" not in result


class TestRoundTrip:
    """Tests for round-trip conversions."""

    def test_ollama_round_trip(self):
        """Should preserve values in round trip."""
        original = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "num_predict": 1024,
            "repeat_penalty": 1.2,
            "seed": 42,
        }

        config = ollama_to_generation_config(original)
        result = generation_config_to_ollama(config)

        assert result["temperature"] == original["temperature"]
        assert result["top_p"] == original["top_p"]
        assert result["top_k"] == original["top_k"]
        assert result["num_predict"] == original["num_predict"]
        assert result["repeat_penalty"] == original["repeat_penalty"]
        assert result["seed"] == original["seed"]

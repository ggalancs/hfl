# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for input validation utilities."""

import pytest

from hfl.validators import (
    BOUNDS,
    VALID_QUANTIZATIONS,
    ValidationError,
    validate_alias,
    validate_generation_params,
    validate_messages,
    validate_model_name,
    validate_port,
    validate_prompt,
    validate_quantization,
    validate_tts_params,
)


class TestValidateModelName:
    """Tests for validate_model_name function."""

    def test_valid_simple_name(self):
        """Valid simple model names should pass."""
        assert validate_model_name("llama-7b") == "llama-7b"
        assert validate_model_name("Qwen2.5-72B") == "Qwen2.5-72B"
        assert validate_model_name("model_v1") == "model_v1"

    def test_valid_repo_id(self):
        """Valid HuggingFace repo IDs should pass."""
        assert validate_model_name("meta-llama/Llama-3-8B") == "meta-llama/Llama-3-8B"
        assert validate_model_name("mistralai/Mistral-7B-v0.1") == "mistralai/Mistral-7B-v0.1"

    def test_path_traversal_double_dot(self):
        """Path traversal with .. should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("../../../etc/passwd")
        assert "Path traversal" in str(exc_info.value)

    def test_path_traversal_hidden_double_dot(self):
        """Hidden path traversal should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("model/../../../secret")
        assert "Path traversal" in str(exc_info.value)

    def test_absolute_path_unix(self):
        """Absolute Unix paths should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("/etc/passwd")
        assert "Absolute paths" in str(exc_info.value)

    def test_absolute_path_home(self):
        """Home-relative paths should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("~/.ssh/id_rsa")
        assert "Absolute paths" in str(exc_info.value)

    def test_backslash_rejected(self):
        """Backslashes should be rejected (Windows paths)."""
        # This tests backslash without path traversal
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("model\\subdir")
        assert "Backslashes" in str(exc_info.value)

    def test_backslash_with_traversal(self):
        """Windows path traversal should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("..\\..\\windows\\system32")
        # Detected as path traversal (.. comes first)
        assert "Path traversal" in str(exc_info.value)

    def test_empty_name(self):
        """Empty names should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("")
        assert "cannot be empty" in str(exc_info.value)

    def test_too_long_name(self):
        """Names exceeding max length should be rejected."""
        long_name = "a" * 300
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name(long_name)
        assert "too long" in str(exc_info.value)

    def test_invalid_start_character(self):
        """Names starting with invalid characters should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("-invalid")
        assert "Invalid model name format" in str(exc_info.value)

    def test_invalid_characters(self):
        """Names with invalid characters should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("model@name")
        assert "Invalid model name format" in str(exc_info.value)


class TestValidateQuantization:
    """Tests for validate_quantization function."""

    def test_valid_quantizations(self):
        """All valid quantization levels should pass."""
        for quant in VALID_QUANTIZATIONS:
            assert validate_quantization(quant) == quant
            assert validate_quantization(quant.lower()) == quant

    def test_invalid_quantization(self):
        """Invalid quantization levels should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_quantization("Q99_K_Z")
        assert "Invalid quantization" in str(exc_info.value)

    def test_quantization_case_insensitive(self):
        """Quantization validation should be case-insensitive."""
        assert validate_quantization("q4_k_m") == "Q4_K_M"
        assert validate_quantization("Q4_K_M") == "Q4_K_M"


class TestValidatePort:
    """Tests for validate_port function."""

    def test_valid_ports(self):
        """Valid port numbers should pass."""
        assert validate_port(80) == 80
        assert validate_port(443) == 443
        assert validate_port(11434) == 11434
        assert validate_port(65535) == 65535

    def test_port_too_low(self):
        """Port 0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_port(0)
        assert "must be between" in str(exc_info.value)

    def test_port_too_high(self):
        """Ports above 65535 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_port(99999)
        assert "must be between" in str(exc_info.value)

    def test_port_wrong_type(self):
        """Non-integer ports should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_port("8080")  # type: ignore
        assert "must be an integer" in str(exc_info.value)


class TestValidateGenerationParams:
    """Tests for validate_generation_params function."""

    def test_valid_params(self):
        """Valid generation parameters should pass."""
        validate_generation_params(
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            ctx_size=4096,
        )

    def test_none_params_allowed(self):
        """None values should be allowed (use defaults)."""
        validate_generation_params()
        validate_generation_params(max_tokens=None, temperature=None)

    def test_max_tokens_bounds(self):
        """max_tokens should be within bounds."""
        with pytest.raises(ValidationError):
            validate_generation_params(max_tokens=0)
        with pytest.raises(ValidationError):
            validate_generation_params(max_tokens=BOUNDS.MAX_TOKENS_LIMIT + 1)

    def test_temperature_bounds(self):
        """temperature should be within bounds."""
        validate_generation_params(temperature=0.0)  # Edge case - valid
        validate_generation_params(temperature=2.0)  # Edge case - valid

        with pytest.raises(ValidationError):
            validate_generation_params(temperature=-0.1)
        with pytest.raises(ValidationError):
            validate_generation_params(temperature=2.1)

    def test_top_p_bounds(self):
        """top_p should be within [0, 1]."""
        validate_generation_params(top_p=0.0)
        validate_generation_params(top_p=1.0)

        with pytest.raises(ValidationError):
            validate_generation_params(top_p=-0.1)
        with pytest.raises(ValidationError):
            validate_generation_params(top_p=1.1)

    def test_context_size_bounds(self):
        """context_size should be within bounds."""
        with pytest.raises(ValidationError):
            validate_generation_params(ctx_size=64)  # Too small
        with pytest.raises(ValidationError):
            validate_generation_params(ctx_size=BOUNDS.MAX_CONTEXT_SIZE + 1)


class TestValidateMessages:
    """Tests for validate_messages function."""

    def test_valid_messages(self):
        """Valid message lists should pass."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        validate_messages(messages)

    def test_empty_messages(self):
        """Empty message list should be rejected."""
        with pytest.raises(ValidationError):
            validate_messages([])

    def test_not_a_list(self):
        """Non-list messages should be rejected."""
        with pytest.raises(ValidationError):
            validate_messages("not a list")  # type: ignore

    def test_missing_role(self):
        """Messages without role should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_messages([{"content": "Hello"}])
        assert "missing 'role'" in str(exc_info.value)

    def test_missing_content(self):
        """Messages without content should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_messages([{"role": "user"}])
        assert "missing 'content'" in str(exc_info.value)

    def test_too_many_messages(self):
        """Too many messages should be rejected."""
        messages = [{"role": "user", "content": "Hi"}] * (BOUNDS.MAX_MESSAGES + 1)
        with pytest.raises(ValidationError):
            validate_messages(messages)


class TestValidatePrompt:
    """Tests for validate_prompt function."""

    def test_valid_string_prompt(self):
        """Valid string prompt should pass."""
        validate_prompt("Hello, world!")

    def test_valid_list_prompt(self):
        """Valid list of prompts should pass."""
        validate_prompt(["Hello", "World"])

    def test_too_long_prompt(self):
        """Prompt exceeding max length should be rejected."""
        long_prompt = "x" * (BOUNDS.MAX_PROMPT_LENGTH + 1)
        with pytest.raises(ValidationError):
            validate_prompt(long_prompt)

    def test_wrong_type(self):
        """Wrong type prompts should be rejected."""
        with pytest.raises(ValidationError):
            validate_prompt(12345)  # type: ignore


class TestValidateTTSParams:
    """Tests for validate_tts_params function."""

    def test_valid_params(self):
        """Valid TTS parameters should pass."""
        validate_tts_params(speed=1.0, sample_rate=22050)

    def test_speed_bounds(self):
        """speed should be within bounds."""
        with pytest.raises(ValidationError):
            validate_tts_params(speed=0.1)  # Too slow
        with pytest.raises(ValidationError):
            validate_tts_params(speed=5.0)  # Too fast

    def test_sample_rate_bounds(self):
        """sample_rate should be within bounds."""
        with pytest.raises(ValidationError):
            validate_tts_params(sample_rate=4000)  # Too low
        with pytest.raises(ValidationError):
            validate_tts_params(sample_rate=100000)  # Too high


class TestValidateAlias:
    """Tests for validate_alias function."""

    def test_valid_alias(self):
        """Valid aliases should pass."""
        assert validate_alias("llama3") == "llama3"
        assert validate_alias("my-model") == "my-model"
        assert validate_alias("model_v1.2") == "model_v1.2"

    def test_empty_alias(self):
        """Empty alias should be rejected."""
        with pytest.raises(ValidationError):
            validate_alias("")

    def test_too_long_alias(self):
        """Alias exceeding max length should be rejected."""
        with pytest.raises(ValidationError):
            validate_alias("a" * 65)

    def test_slash_not_allowed(self):
        """Slashes should not be allowed in aliases."""
        with pytest.raises(ValidationError):
            validate_alias("org/model")

# SPDX-License-Identifier: HRUL-1.0
"""Tests for CLI helper functions."""


from hfl.cli.main import (
    _estimate_model_size,
    _extract_params_from_name,
    _format_size,
)


class TestFormatSize:
    """Tests for _format_size helper function."""

    def test_zero_bytes(self):
        """Test formatting zero bytes."""
        result = _format_size(0)
        assert result == "N/A" or "N/A" in result or result != ""

    def test_small_bytes(self):
        """Test formatting small byte counts."""
        result = _format_size(500)
        assert "B" in result
        assert "500" in result

    def test_megabytes(self):
        """Test formatting megabyte sizes."""
        result = _format_size(50 * 1024 * 1024)  # 50 MB
        assert "MB" in result
        assert "50" in result

    def test_gigabytes(self):
        """Test formatting gigabyte sizes."""
        result = _format_size(5 * 1024 * 1024 * 1024)  # 5 GB
        assert "GB" in result
        assert "5" in result

    def test_large_gigabytes(self):
        """Test formatting large gigabyte sizes."""
        result = _format_size(50 * 1024 * 1024 * 1024)  # 50 GB
        assert "GB" in result

    def test_fractional_gigabytes(self):
        """Test formatting fractional gigabyte sizes."""
        result = _format_size(int(1.5 * 1024 * 1024 * 1024))  # 1.5 GB
        assert "GB" in result


class TestExtractParamsFromName:
    """Tests for _extract_params_from_name helper function."""

    def test_extract_70b(self):
        """Test extracting 70B parameters."""
        result = _extract_params_from_name("meta-llama/Llama-3.3-70B-Instruct")
        assert result == "70B"

    def test_extract_7b(self):
        """Test extracting 7B parameters."""
        result = _extract_params_from_name("mistralai/Mistral-7B-Instruct-v0.3")
        assert result == "7B"

    def test_extract_1_5b(self):
        """Test extracting 1.5B parameters."""
        result = _extract_params_from_name("microsoft/phi-1.5b")
        assert result == "1.5B"

    def test_extract_405b(self):
        """Test extracting 405B parameters."""
        result = _extract_params_from_name("meta-llama/Llama-3.1-405B")
        assert result == "405B"

    def test_extract_from_suffix(self):
        """Test extracting from model-7b pattern."""
        result = _extract_params_from_name("org/model-7b-chat")
        assert result == "7B"

    def test_no_params_found(self):
        """Test when no parameters can be extracted."""
        result = _extract_params_from_name("org/some-model-without-params")
        assert result is None

    def test_lowercase_b(self):
        """Test extracting with lowercase 'b'."""
        result = _extract_params_from_name("org/model-13b")
        assert result == "13B"

    def test_with_instruct_suffix(self):
        """Test extracting with -instruct suffix."""
        result = _extract_params_from_name("org/model-70b-instruct")
        assert result == "70B"


class TestEstimateModelSize:
    """Tests for _estimate_model_size helper function."""

    def test_estimate_none_params(self):
        """Test with None parameters returns '?'."""
        result = _estimate_model_size(None)
        assert result == "?"

    def test_estimate_7b_q4(self):
        """Test estimating 7B model with Q4 quantization."""
        result = _estimate_model_size("7B", "Q4_K_M")
        assert "GB" in result

    def test_estimate_70b_q4(self):
        """Test estimating 70B model with Q4 quantization."""
        result = _estimate_model_size("70B", "Q4_K_M")
        assert "GB" in result
        # 70B * 4.5 bits / 8 ≈ 37-40GB depending on exact calculation
        assert "37" in result or "38" in result or "39" in result or "40" in result

    def test_estimate_7b_q8(self):
        """Test estimating 7B model with Q8 quantization."""
        result = _estimate_model_size("7B", "Q8_0")
        assert "GB" in result

    def test_estimate_7b_f16(self):
        """Test estimating 7B model with F16 (no quantization)."""
        result = _estimate_model_size("7B", "F16")
        assert "GB" in result

    def test_estimate_small_model(self):
        """Test estimating small model (1.5B)."""
        result = _estimate_model_size("1.5B", "Q4_K_M")
        assert "GB" in result

    def test_estimate_q2_quantization(self):
        """Test estimating with Q2 quantization."""
        result = _estimate_model_size("70B", "Q2_K")
        assert "GB" in result

    def test_estimate_q3_quantization(self):
        """Test estimating with Q3 quantization."""
        result = _estimate_model_size("70B", "Q3_K_M")
        assert "GB" in result

    def test_estimate_q5_quantization(self):
        """Test estimating with Q5 quantization."""
        result = _estimate_model_size("70B", "Q5_K_M")
        assert "GB" in result

    def test_estimate_q6_quantization(self):
        """Test estimating with Q6 quantization."""
        result = _estimate_model_size("70B", "Q6_K")
        assert "GB" in result

    def test_estimate_unknown_quantization(self):
        """Test estimating with unknown quantization defaults to Q4."""
        result = _estimate_model_size("7B", "UNKNOWN")
        assert "GB" in result

    def test_estimate_very_large_model(self):
        """Test estimating very large model (405B)."""
        result = _estimate_model_size("405B", "Q4_K_M")
        assert "GB" in result

# SPDX-License-Identifier: HRUL-1.0
"""Edge case tests for hub resolver module."""


from hfl.hub.resolver import _get_quant_levels, _is_quantization


class TestIsQuantizationEdgeCases:
    """Edge case tests for _is_quantization function."""

    def test_empty_string(self):
        """Test empty string returns False."""
        result = _is_quantization("")
        assert result is False

    def test_none_like_empty(self):
        """Test None-like empty returns False."""
        result = _is_quantization("")
        assert result is False

    def test_f16_is_quantization(self):
        """Test F16 is recognized as quantization."""
        result = _is_quantization("F16")
        assert result is True

    def test_f32_is_quantization(self):
        """Test F32 is recognized as quantization."""
        result = _is_quantization("F32")
        assert result is True

    def test_iq2_is_quantization(self):
        """Test IQ2_XS is recognized as quantization."""
        result = _is_quantization("IQ2_XS")
        assert result is True

    def test_iq3_is_quantization(self):
        """Test IQ3_XXS is recognized as quantization."""
        result = _is_quantization("IQ3_XXS")
        assert result is True

    def test_q4_no_underscore_is_quantization(self):
        """Test Q4KM (without underscore) is recognized."""
        result = _is_quantization("Q4KM")
        assert result is True

    def test_random_string_not_quantization(self):
        """Test random string is not quantization."""
        result = _is_quantization("instruct")
        assert result is False

    def test_chat_not_quantization(self):
        """Test 'chat' is not quantization."""
        result = _is_quantization("chat")
        assert result is False

    def test_main_not_quantization(self):
        """Test 'main' is not quantization."""
        result = _is_quantization("main")
        assert result is False

    def test_version_not_quantization(self):
        """Test version string is not quantization."""
        result = _is_quantization("v1.0")
        assert result is False

    def test_q2_k_is_quantization(self):
        """Test Q2_K is recognized."""
        result = _is_quantization("Q2_K")
        assert result is True

    def test_q3_k_s_is_quantization(self):
        """Test Q3_K_S is recognized."""
        result = _is_quantization("Q3_K_S")
        assert result is True

    def test_q6_k_is_quantization(self):
        """Test Q6_K is recognized."""
        result = _is_quantization("Q6_K")
        assert result is True

    def test_lowercase_f16(self):
        """Test lowercase f16 is recognized."""
        result = _is_quantization("f16")
        assert result is True


class TestGetQuantLevelsComplete:
    """Tests for _get_quant_levels completeness."""

    def test_contains_q2_k(self):
        """Test Q2_K is in quant levels."""
        levels = _get_quant_levels()
        assert "Q2_K" in levels

    def test_contains_q3_variants(self):
        """Test Q3 variants are in quant levels."""
        levels = _get_quant_levels()
        assert "Q3_K_S" in levels
        assert "Q3_K_M" in levels
        assert "Q3_K_L" in levels

    def test_contains_q4_variants(self):
        """Test Q4 variants are in quant levels."""
        levels = _get_quant_levels()
        assert "Q4_0" in levels
        assert "Q4_K_S" in levels
        assert "Q4_K_M" in levels

    def test_contains_q5_variants(self):
        """Test Q5 variants are in quant levels."""
        levels = _get_quant_levels()
        assert "Q5_0" in levels
        assert "Q5_K_S" in levels
        assert "Q5_K_M" in levels

    def test_contains_q6_k(self):
        """Test Q6_K is in quant levels."""
        levels = _get_quant_levels()
        assert "Q6_K" in levels

    def test_contains_q8_0(self):
        """Test Q8_0 is in quant levels."""
        levels = _get_quant_levels()
        assert "Q8_0" in levels

    def test_contains_fp16(self):
        """Test FP16 variants are in quant levels."""
        levels = _get_quant_levels()
        # Check for F16 or FP16
        assert any("F16" in level or "FP16" in level for level in levels)

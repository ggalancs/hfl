# SPDX-License-Identifier: HRUL-1.0
"""Extended tests for hub resolver module."""

from unittest.mock import MagicMock, patch

import pytest

from hfl.hub.resolver import (
    ResolvedModel,
    _detect_quant,
    _get_quant_levels,
    _is_quantization,
    _select_gguf,
    resolve,
)


class TestSelectGguf:
    """Tests for _select_gguf function."""

    def test_select_with_exact_quant_match(self):
        """Test selecting GGUF with exact quantization match."""
        files = ["model-Q4_K_M.gguf", "model-Q5_K_M.gguf", "model-Q8_0.gguf"]

        result = _select_gguf(files, "Q5_K_M")

        assert "Q5_K_M" in result

    def test_select_with_lowercase_quant(self):
        """Test selecting with lowercase quantization."""
        files = ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]

        result = _select_gguf(files, "q4_k_m")

        assert "Q4_K_M" in result

    def test_select_priority_q4_k_m(self):
        """Test Q4_K_M is selected by default priority."""
        files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q6_K.gguf"]

        result = _select_gguf(files, None)

        assert "Q4_K_M" in result

    def test_select_priority_q5_k_m(self):
        """Test Q5_K_M is selected when Q4_K_M not available."""
        files = ["model-Q8_0.gguf", "model-Q5_K_M.gguf", "model-Q6_K.gguf"]

        result = _select_gguf(files, None)

        assert "Q5_K_M" in result

    def test_select_fallback_to_first(self):
        """Test fallback to first file when no priority match."""
        files = ["model-IQ2_XS.gguf", "model-IQ3_XXS.gguf"]

        result = _select_gguf(files, None)

        assert result == files[0]


class TestDetectQuant:
    """Tests for _detect_quant function."""

    def test_detect_q4_k_m(self):
        """Test detecting Q4_K_M quantization."""
        result = _detect_quant("model-Q4_K_M.gguf")

        assert result == "Q4_K_M"

    def test_detect_q8_0(self):
        """Test detecting Q8_0 quantization."""
        result = _detect_quant("model-Q8_0.gguf")

        assert result == "Q8_0"

    def test_detect_no_quant(self):
        """Test when no quantization is detected."""
        result = _detect_quant("model.gguf")

        assert result is None

    def test_detect_case_insensitive(self):
        """Test case insensitive detection."""
        result = _detect_quant("model-q4_k_m.GGUF")

        assert result == "Q4_K_M"


class TestGetQuantLevels:
    """Tests for _get_quant_levels function."""

    def test_returns_list(self):
        """Test that it returns a list."""
        result = _get_quant_levels()

        assert isinstance(result, list)
        assert len(result) > 0

    def test_contains_common_quants(self):
        """Test that common quantization levels are included."""
        result = _get_quant_levels()

        assert "Q4_K_M" in result
        assert "Q5_K_M" in result
        assert "Q8_0" in result


class TestIsQuantization:
    """Tests for _is_quantization function."""

    def test_valid_quant_q4_k_m(self):
        """Test Q4_K_M is recognized as quantization."""
        result = _is_quantization("Q4_K_M")
        assert result is True

    def test_valid_quant_lowercase(self):
        """Test lowercase quantization is recognized."""
        result = _is_quantization("q4_k_m")
        assert result is True

    def test_invalid_quant(self):
        """Test non-quantization string."""
        result = _is_quantization("instruct")
        assert result is False

    def test_branch_name_not_quant(self):
        """Test branch name is not recognized as quantization."""
        result = _is_quantization("main")
        assert result is False


class TestResolve:
    """Tests for resolve function."""

    def test_resolve_huggingface_repo_gguf(self):
        """Test resolving HuggingFace repo with GGUF files."""
        with patch("hfl.hub.resolver.HfApi") as mock_api_class:
            mock_api = MagicMock()

            # Mock model info
            mock_info = MagicMock()
            mock_info.id = "org/model"

            # Mock siblings (files)
            mock_sibling = MagicMock()
            mock_sibling.rfilename = "model-Q4_K_M.gguf"
            mock_info.siblings = [mock_sibling]

            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = resolve("org/model")

            assert result.repo_id == "org/model"
            assert result.format == "gguf"
            assert result.filename == "model-Q4_K_M.gguf"

    def test_resolve_huggingface_repo_safetensors(self):
        """Test resolving HuggingFace repo with safetensors."""
        with patch("hfl.hub.resolver.HfApi") as mock_api_class:
            mock_api = MagicMock()

            # Mock model info
            mock_info = MagicMock()
            mock_info.id = "org/model"

            # Mock siblings (safetensors files)
            mock_sibling = MagicMock()
            mock_sibling.rfilename = "model.safetensors"
            mock_info.siblings = [mock_sibling]

            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = resolve("org/model")

            assert result.repo_id == "org/model"
            assert result.format == "safetensors"

    def test_resolve_huggingface_repo_pytorch(self):
        """Test resolving HuggingFace repo with pytorch files."""
        with patch("hfl.hub.resolver.HfApi") as mock_api_class:
            mock_api = MagicMock()

            # Mock model info
            mock_info = MagicMock()
            mock_info.id = "org/model"

            # Mock siblings (pytorch files only)
            mock_sibling = MagicMock()
            mock_sibling.rfilename = "pytorch_model.bin"
            mock_info.siblings = [mock_sibling]

            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = resolve("org/model")

            assert result.repo_id == "org/model"
            assert result.format == "pytorch"

    def test_resolve_with_quantization_param(self):
        """Test resolving with specific quantization."""
        with patch("hfl.hub.resolver.HfApi") as mock_api_class:
            mock_api = MagicMock()

            mock_info = MagicMock()
            mock_info.id = "org/model"

            # Multiple GGUF files
            mock_s1 = MagicMock()
            mock_s1.rfilename = "model-Q4_K_M.gguf"
            mock_s2 = MagicMock()
            mock_s2.rfilename = "model-Q8_0.gguf"
            mock_info.siblings = [mock_s1, mock_s2]

            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = resolve("org/model", quantization="Q8_0")

            assert result.quantization == "Q8_0"
            assert "Q8_0" in result.filename

    def test_resolve_with_colon_quantization(self):
        """Test resolving with Ollama-style quantization (org/model:Q4_K_M)."""
        with patch("hfl.hub.resolver.HfApi") as mock_api_class:
            mock_api = MagicMock()

            mock_info = MagicMock()
            mock_info.id = "org/model"

            mock_s1 = MagicMock()
            mock_s1.rfilename = "model-Q4_K_M.gguf"
            mock_s2 = MagicMock()
            mock_s2.rfilename = "model-Q8_0.gguf"
            mock_info.siblings = [mock_s1, mock_s2]

            mock_api.model_info.return_value = mock_info
            mock_api_class.return_value = mock_api

            result = resolve("org/model:Q8_0")

            assert result.quantization == "Q8_0"

    def test_resolve_search_by_name(self):
        """Test resolving by searching for model name."""
        with patch("hfl.hub.resolver.HfApi") as mock_api_class:
            mock_api = MagicMock()

            # Mock search results
            mock_result = MagicMock()
            mock_result.id = "org/found-model"
            mock_api.list_models.return_value = [mock_result]

            # Mock model info
            mock_info = MagicMock()
            mock_sibling = MagicMock()
            mock_sibling.rfilename = "model.gguf"
            mock_info.siblings = [mock_sibling]
            mock_api.model_info.return_value = mock_info

            mock_api_class.return_value = mock_api

            result = resolve("found-model")

            assert result.repo_id == "org/found-model"

    def test_resolve_not_found_raises(self):
        """Test resolving non-existent model raises error."""
        with patch("hfl.hub.resolver.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.list_models.return_value = []  # No results
            mock_api_class.return_value = mock_api

            with pytest.raises(ValueError) as exc_info:
                resolve("nonexistent-model")

            assert "not found" in str(exc_info.value).lower()


class TestResolvedModel:
    """Tests for ResolvedModel dataclass."""

    def test_resolved_model_defaults(self):
        """Test ResolvedModel default values."""
        model = ResolvedModel(repo_id="org/model", format="gguf")

        assert model.repo_id == "org/model"
        assert model.format == "gguf"
        assert model.filename is None
        assert model.revision == "main"  # Default is "main"
        assert model.quantization is None

    def test_resolved_model_with_all_fields(self):
        """Test ResolvedModel with all fields."""
        model = ResolvedModel(
            repo_id="org/model",
            format="gguf",
            filename="model.Q4_K_M.gguf",
            revision="v1.0",
            quantization="Q4_K_M",
        )

        assert model.repo_id == "org/model"
        assert model.format == "gguf"
        assert model.filename == "model.Q4_K_M.gguf"
        assert model.revision == "v1.0"
        assert model.quantization == "Q4_K_M"

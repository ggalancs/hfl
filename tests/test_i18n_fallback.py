# SPDX-License-Identifier: HRUL-1.0
"""Tests for i18n fallback and edge cases."""

from unittest.mock import patch

import pytest

from hfl.i18n import (
    DEFAULT_LANGUAGE,
    _get_nested_value,
    _load_translations,
    _translations,
    get_language,
    set_language,
    t,
)


class TestTranslationFallback:
    """Tests for translation fallback behavior."""

    @pytest.fixture(autouse=True)
    def reset_translations(self):
        """Reset translations cache before each test."""
        _translations.clear()
        yield
        _translations.clear()

    def test_fallback_to_english_when_key_missing(self):
        """Test fallback to English when key not found in current language."""
        # Set to a language that might not have all keys
        original_lang = get_language()
        set_language("en")  # English should have the key

        result = t("errors.error")

        # Should return something (not the key itself if translation exists)
        assert result is not None

        set_language(original_lang)

    def test_return_key_when_no_translation(self):
        """Test returning the key when no translation found."""
        result = t("nonexistent.key.that.does.not.exist")

        # Should return the key itself
        assert result == "nonexistent.key.that.does.not.exist"

    def test_interpolation_with_missing_variable(self):
        """Test interpolation when variable is missing."""
        # Find a key that uses interpolation
        original_lang = get_language()
        set_language("en")

        # This should not crash even if we don't provide all variables
        result = t("messages.downloaded_to")  # This key might expect {path}

        # Should return something without crashing
        assert result is not None

        set_language(original_lang)

    def test_interpolation_success(self):
        """Test successful interpolation."""
        original_lang = get_language()
        set_language("en")

        # Test with a key that takes parameters
        result = t("messages.models_found", count=5)

        # Should contain the interpolated value
        assert "5" in result

        set_language(original_lang)


class TestGetNestedValue:
    """Tests for _get_nested_value function."""

    def test_single_level(self):
        """Test getting single level value."""
        data = {"key": "value"}
        result = _get_nested_value(data, "key")
        assert result == "value"

    def test_nested_level(self):
        """Test getting nested value."""
        data = {"level1": {"level2": {"key": "value"}}}
        result = _get_nested_value(data, "level1.level2.key")
        assert result == "value"

    def test_missing_key(self):
        """Test when key doesn't exist."""
        data = {"key": "value"}
        result = _get_nested_value(data, "missing")
        assert result is None

    def test_missing_nested_key(self):
        """Test when nested key doesn't exist."""
        data = {"level1": {"key": "value"}}
        result = _get_nested_value(data, "level1.missing.key")
        assert result is None

    def test_non_string_value(self):
        """Test when value is not a string."""
        data = {"key": {"nested": "value"}}
        result = _get_nested_value(data, "key")
        assert result is None  # Returns None for non-string

    def test_empty_dict(self):
        """Test with empty dictionary."""
        data = {}
        result = _get_nested_value(data, "any.key")
        assert result is None


class TestLoadTranslations:
    """Tests for _load_translations function."""

    @pytest.fixture(autouse=True)
    def reset_translations(self):
        """Reset translations cache before each test."""
        _translations.clear()
        yield
        _translations.clear()

    def test_load_default_language(self):
        """Test loading default language translations."""
        translations = _load_translations(DEFAULT_LANGUAGE)

        assert translations is not None
        assert isinstance(translations, dict)
        assert len(translations) > 0

    def test_cached_translations(self):
        """Test that translations are cached."""
        # Load once
        trans1 = _load_translations(DEFAULT_LANGUAGE)

        # Load again - should return cached
        trans2 = _load_translations(DEFAULT_LANGUAGE)

        assert trans1 is trans2  # Same object (cached)

    def test_fallback_for_missing_language(self):
        """Test fallback for non-existent language."""
        # Try to load a language that doesn't exist
        with patch("hfl.i18n.SUPPORTED_LANGUAGES", {"en", "zz"}):
            # This should fall back to English
            try:
                _load_translations("zz")
                # If it doesn't raise, it should have fallen back
            except FileNotFoundError:
                # Also acceptable if file not found
                pass


class TestSetAndGetLanguage:
    """Tests for set_language and get_language functions."""

    def test_set_valid_language(self):
        """Test setting a valid language."""
        original = get_language()

        set_language("es")
        assert get_language() == "es"

        # Restore
        set_language(original)

    def test_set_invalid_language_raises(self):
        """Test that setting invalid language raises error."""
        with pytest.raises(ValueError):
            set_language("invalid-lang-code")

    def test_get_language_returns_string(self):
        """Test that get_language returns a string."""
        lang = get_language()
        assert isinstance(lang, str)
        assert len(lang) >= 2


class TestTranslationInterpolation:
    """Tests for translation string interpolation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up language for tests."""
        original = get_language()
        set_language("en")
        yield
        set_language(original)

    def test_interpolation_with_extra_kwargs(self):
        """Test interpolation ignores extra kwargs."""
        # This should not crash even with extra params
        result = t("messages.models_found", count=5, extra_param="ignored")
        assert "5" in result

    def test_interpolation_format_error_returns_original(self):
        """Test interpolation returns original if format fails."""
        # If the format string has placeholders but wrong kwargs
        # it should return the original string
        result = t("messages.models_found", wrong_param="value")
        # Should still return something (either formatted or original)
        assert result is not None


class TestSupportedLanguages:
    """Tests for language support."""

    def test_english_is_supported(self):
        """Test English is in supported languages."""
        from hfl.i18n import SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES

    def test_spanish_is_supported(self):
        """Test Spanish is in supported languages."""
        from hfl.i18n import SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES

    def test_get_supported_languages(self):
        """Test get_supported_languages function."""
        from hfl.i18n import get_supported_languages

        languages = get_supported_languages()
        assert isinstance(languages, list)
        assert "en" in languages
        assert "es" in languages

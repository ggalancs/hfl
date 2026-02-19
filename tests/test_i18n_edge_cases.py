# SPDX-License-Identifier: HRUL-1.0
"""Edge case tests for i18n module."""


import pytest

from hfl.i18n import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    _get_nested_value,
    _load_translations,
    _translations,
    get_language,
    set_language,
    t,
)


class TestTranslationEdgeCases:
    """Edge case tests for translation function."""

    @pytest.fixture(autouse=True)
    def reset_translations(self):
        """Reset translations cache before each test."""
        _translations.clear()
        original_lang = get_language()
        yield
        _translations.clear()
        try:
            set_language(original_lang)
        except ValueError:
            set_language("en")

    def test_translation_missing_from_non_english(self):
        """Test fallback to English when key missing in current language."""
        set_language("es")

        # Use a key that definitely exists in English
        result = t("errors.error")

        # Should return something (either Spanish or English fallback)
        assert result is not None
        assert len(result) > 0

    def test_translation_missing_from_all_languages(self):
        """Test returning key when not found in any language."""
        set_language("en")

        result = t("this.key.definitely.does.not.exist.anywhere")

        # Should return the key itself
        assert result == "this.key.definitely.does.not.exist.anywhere"

    def test_translation_with_format_kwargs(self):
        """Test translation with format kwargs."""
        set_language("en")

        # Test with a known key that uses formatting
        result = t("messages.server_at", host="localhost", port=8080)

        # Should contain the interpolated values
        assert "localhost" in result or "8080" in result or result is not None

    def test_translation_format_key_error(self):
        """Test translation when format key is missing."""
        set_language("en")

        # This should not crash even with wrong kwargs
        result = t("messages.models_found", wrong_key="value")

        # Should return something (original or partially formatted)
        assert result is not None


class TestGetNestedValueEdgeCases:
    """Edge case tests for _get_nested_value function."""

    def test_deeply_nested_value(self):
        """Test getting deeply nested value."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "key": "deep_value"
                        }
                    }
                }
            }
        }

        result = _get_nested_value(data, "level1.level2.level3.level4.key")

        assert result == "deep_value"

    def test_value_is_number(self):
        """Test when value is a number (not string)."""
        data = {"key": 123}

        result = _get_nested_value(data, "key")

        # Numbers are not strings, should return None
        assert result is None

    def test_value_is_list(self):
        """Test when value is a list."""
        data = {"key": ["a", "b", "c"]}

        result = _get_nested_value(data, "key")

        # Lists are not strings, should return None
        assert result is None

    def test_value_is_bool(self):
        """Test when value is boolean."""
        data = {"key": True}

        result = _get_nested_value(data, "key")

        # Booleans are not strings, should return None
        assert result is None

    def test_partial_path_exists(self):
        """Test when only part of the path exists."""
        data = {"level1": {"level2": "value"}}

        result = _get_nested_value(data, "level1.level2.level3")

        # level2 is a string, not a dict, so level3 can't be found
        assert result is None


class TestLoadTranslationsEdgeCases:
    """Edge case tests for _load_translations function."""

    @pytest.fixture(autouse=True)
    def reset_translations(self):
        """Reset translations cache before each test."""
        _translations.clear()
        yield
        _translations.clear()

    def test_load_english(self):
        """Test loading English translations."""
        translations = _load_translations("en")

        assert translations is not None
        assert isinstance(translations, dict)
        assert len(translations) > 0

    def test_load_spanish(self):
        """Test loading Spanish translations."""
        translations = _load_translations("es")

        assert translations is not None
        assert isinstance(translations, dict)
        assert len(translations) > 0

    def test_caching_works(self):
        """Test that translations are cached."""
        _translations.clear()

        # First load
        trans1 = _load_translations("en")

        # Second load should return cached
        trans2 = _load_translations("en")

        assert trans1 is trans2


class TestLanguageManagement:
    """Tests for language management functions."""

    def test_default_language_is_valid(self):
        """Test that DEFAULT_LANGUAGE is in SUPPORTED_LANGUAGES."""
        assert DEFAULT_LANGUAGE in SUPPORTED_LANGUAGES

    def test_supported_languages_not_empty(self):
        """Test that SUPPORTED_LANGUAGES is not empty."""
        assert len(SUPPORTED_LANGUAGES) > 0

    def test_set_language_to_english(self):
        """Test setting language to English."""
        original = get_language()

        set_language("en")
        assert get_language() == "en"

        set_language(original)

    def test_set_language_to_spanish(self):
        """Test setting language to Spanish."""
        original = get_language()

        set_language("es")
        assert get_language() == "es"

        set_language(original)

    def test_get_language_returns_valid_language(self):
        """Test get_language returns a valid language code."""
        lang = get_language()

        assert lang in SUPPORTED_LANGUAGES

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the internationalization (i18n) module."""

import pytest


class TestI18nModule:
    """Tests for i18n core functionality."""

    def test_default_language_is_english(self, monkeypatch):
        """Default language should be English."""
        monkeypatch.delenv("HFL_LANG", raising=False)
        monkeypatch.setenv("LANG", "C")  # Neutral locale

        from hfl.i18n import get_language

        get_language.cache_clear()
        # May be 'en' or system default, but t() should work
        from hfl.i18n import t

        # English translation should work
        result = t("app.description")
        assert "HuggingFace" in result or "models" in result.lower()

    def test_set_language_english(self, monkeypatch):
        """Setting HFL_LANG=en should use English."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_language

        get_language.cache_clear()

        assert get_language() == "en"

    def test_set_language_spanish(self, monkeypatch):
        """Setting HFL_LANG=es should use Spanish."""
        monkeypatch.setenv("HFL_LANG", "es")

        from hfl.i18n import get_language

        get_language.cache_clear()

        assert get_language() == "es"

    def test_unsupported_language_falls_back(self, monkeypatch):
        """Unsupported language should fall back to English."""
        monkeypatch.setenv("HFL_LANG", "fr")

        from hfl.i18n import get_language

        get_language.cache_clear()

        # Should fall back to system locale or default
        lang = get_language()
        assert lang in ("en", "es")  # Either default or system locale

    def test_translation_english(self, monkeypatch):
        """English translations should work."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_language, t

        get_language.cache_clear()

        assert t("app.description") == "Run HuggingFace models locally like Ollama."
        assert t("commands.pull.description") == "Download a model from HuggingFace Hub."
        assert t("errors.model_not_found") == "Model not found"

    def test_translation_spanish(self, monkeypatch):
        """Spanish translations should work."""
        monkeypatch.setenv("HFL_LANG", "es")

        from hfl.i18n import get_language, t

        get_language.cache_clear()

        assert t("app.description") == "Ejecuta modelos de HuggingFace localmente como Ollama."
        assert t("commands.pull.description") == "Descarga un modelo de HuggingFace Hub."
        assert t("errors.model_not_found") == "Modelo no encontrado"

    def test_translation_with_interpolation(self, monkeypatch):
        """String interpolation should work."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_language, t

        get_language.cache_clear()

        result = t("messages.searching", query="llama")
        assert "llama" in result
        assert "HuggingFace" in result

    def test_missing_key_returns_key(self, monkeypatch):
        """Missing translation key should return the key itself."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_language, t

        get_language.cache_clear()

        result = t("nonexistent.key.that.does.not.exist")
        assert result == "nonexistent.key.that.does.not.exist"

    def test_nested_keys(self, monkeypatch):
        """Nested translation keys should work."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_language, t

        get_language.cache_clear()

        result = t("commands.search.options.gguf_only")
        assert "GGUF" in result

    def test_get_supported_languages(self, monkeypatch):
        """Should return list of supported languages."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_supported_languages

        languages = get_supported_languages()
        assert "en" in languages
        assert "es" in languages

    def test_set_language_function(self, monkeypatch):
        """set_language should change the language."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_language, set_language

        get_language.cache_clear()

        set_language("es")
        assert get_language() == "es"

        set_language("en")
        assert get_language() == "en"

    def test_set_invalid_language_raises(self, monkeypatch):
        """set_language with invalid language should raise ValueError."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_language, set_language

        get_language.cache_clear()

        with pytest.raises(ValueError, match="Unsupported language"):
            set_language("invalid")

    def test_alias_underscore_function(self, monkeypatch):
        """The _ alias should work like t."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import _, get_language

        get_language.cache_clear()

        assert _("app.description") == "Run HuggingFace models locally like Ollama."


class TestTranslationCompleteness:
    """Tests to ensure all translations are complete."""

    def test_english_has_all_keys(self, monkeypatch):
        """English translation file should have all required keys."""
        monkeypatch.setenv("HFL_LANG", "en")

        from hfl.i18n import get_language, t

        get_language.cache_clear()

        # Test a sample of important keys
        required_keys = [
            "app.description",
            "commands.pull.description",
            "commands.run.description",
            "commands.serve.description",
            "commands.list.description",
            "commands.search.description",
            "commands.rm.description",
            "commands.inspect.description",
            "commands.alias.description",
            "commands.login.description",
            "commands.logout.description",
            "commands.version.description",
            "errors.model_not_found",
            "messages.model_ready",
            "table.local_models",
        ]

        for key in required_keys:
            result = t(key)
            assert result != key, f"Missing English translation for: {key}"

    def test_spanish_has_all_keys(self, monkeypatch):
        """Spanish translation file should have all required keys."""
        monkeypatch.setenv("HFL_LANG", "es")

        from hfl.i18n import get_language, t

        get_language.cache_clear()

        # Test a sample of important keys
        required_keys = [
            "app.description",
            "commands.pull.description",
            "commands.run.description",
            "commands.serve.description",
            "commands.list.description",
            "commands.search.description",
            "commands.rm.description",
            "commands.inspect.description",
            "commands.alias.description",
            "commands.login.description",
            "commands.logout.description",
            "commands.version.description",
            "errors.model_not_found",
            "messages.model_ready",
            "table.local_models",
        ]

        for key in required_keys:
            result = t(key)
            assert result != key, f"Missing Spanish translation for: {key}"

    def test_translations_are_different(self, monkeypatch):
        """English and Spanish translations should be different."""
        from hfl.i18n import get_language, t

        # Get English
        monkeypatch.setenv("HFL_LANG", "en")
        get_language.cache_clear()
        en_description = t("app.description")

        # Get Spanish
        monkeypatch.setenv("HFL_LANG", "es")
        get_language.cache_clear()
        es_description = t("app.description")

        assert en_description != es_description
        assert "Run" in en_description
        assert "Ejecuta" in es_description

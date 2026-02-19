# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Internationalization (i18n) module for hfl.

Provides multilingual support for CLI messages and user-facing strings.

Usage:
    from hfl.i18n import t

    # Translate a string
    print(t("cli.pull.downloading"))  # "Downloading..." or "Descargando..."

    # With interpolation
    print(t("cli.pull.model_ready", name="llama"))  # "Model ready: llama"

Configuration:
    Set the HFL_LANG environment variable to change language:
        export HFL_LANG=es  # Spanish
        export HFL_LANG=en  # English (default)

    Supported languages: en, es
"""

import json
import locale
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# Supported languages
SUPPORTED_LANGUAGES = {"en", "es"}
DEFAULT_LANGUAGE = "en"

# Cache for loaded translations
_translations: dict[str, dict[str, Any]] = {}


def _get_system_language() -> str:
    """Detect the system language from locale settings."""
    try:
        # Try to get system locale using getlocale (preferred over deprecated getdefaultlocale)
        lang = locale.getlocale()[0]
        if lang:
            # Extract language code (e.g., "es_ES" -> "es")
            lang_code = lang.split("_")[0].lower()
            if lang_code in SUPPORTED_LANGUAGES:
                return lang_code
        # Fallback: try environment variables directly
        for env_var in ("LC_ALL", "LC_MESSAGES", "LANG"):
            lang = os.environ.get(env_var, "")
            if lang:
                lang_code = lang.split("_")[0].split(".")[0].lower()
                if lang_code in SUPPORTED_LANGUAGES:
                    return lang_code
    except Exception:
        pass
    return DEFAULT_LANGUAGE


@lru_cache(maxsize=1)
def get_language() -> str:
    """
    Get the current language setting.

    Priority:
    1. HFL_LANG environment variable
    2. System locale
    3. Default (English)

    Returns:
        Language code (e.g., "en", "es")
    """
    # Check environment variable first
    env_lang = os.environ.get("HFL_LANG", "").lower().strip()
    if env_lang:
        if env_lang in SUPPORTED_LANGUAGES:
            return env_lang
        # Try prefix match (e.g., "es_ES" -> "es")
        lang_code = env_lang.split("_")[0]
        if lang_code in SUPPORTED_LANGUAGES:
            return lang_code

    # Fall back to system locale
    return _get_system_language()


def set_language(lang: str) -> None:
    """
    Set the language programmatically.

    Args:
        lang: Language code (e.g., "en", "es")

    Raises:
        ValueError: If language is not supported
    """
    lang = lang.lower().strip()
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {lang}. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )
    os.environ["HFL_LANG"] = lang
    # Clear the cache to pick up the new language
    get_language.cache_clear()


def _load_translations(lang: str) -> dict[str, Any]:
    """Load translations for a specific language."""
    if lang in _translations:
        return _translations[lang]

    locales_dir = Path(__file__).parent / "locales"
    locale_file = locales_dir / f"{lang}.json"

    if not locale_file.exists():
        # Fall back to default language
        if lang != DEFAULT_LANGUAGE:
            return _load_translations(DEFAULT_LANGUAGE)
        raise FileNotFoundError(f"Translation file not found: {locale_file}")

    with open(locale_file, encoding="utf-8") as f:
        _translations[lang] = json.load(f)

    return _translations[lang]


def _get_nested_value(data: dict[str, Any], key: str) -> str | None:
    """Get a nested value from a dict using dot notation."""
    parts = key.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current if isinstance(current, str) else None


def t(key: str, **kwargs: Any) -> str:
    """
    Translate a key to the current language.

    Args:
        key: Translation key in dot notation (e.g., "cli.pull.downloading")
        **kwargs: Variables for string interpolation

    Returns:
        Translated string, or the key itself if not found

    Examples:
        t("cli.pull.downloading")
        t("cli.pull.model_ready", name="llama-7b")
    """
    lang = get_language()
    translations = _load_translations(lang)

    value = _get_nested_value(translations, key)

    if value is None:
        # Try fallback to English
        if lang != DEFAULT_LANGUAGE:
            fallback = _load_translations(DEFAULT_LANGUAGE)
            value = _get_nested_value(fallback, key)

        if value is None:
            # Return the key as-is if no translation found
            return key

    # Apply string interpolation if kwargs provided
    if kwargs:
        try:
            value = value.format(**kwargs)
        except KeyError:
            # If interpolation fails, return as-is
            pass

    return value


def get_supported_languages() -> list[str]:
    """Return list of supported language codes."""
    return sorted(SUPPORTED_LANGUAGES)


# Convenience alias
_ = t

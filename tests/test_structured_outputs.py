# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.api.structured_outputs`` (P0-5).

Covers:

- Ollama ``format`` normalisation: 'json', schema dict, invalid strings.
- OpenAI ``response_format`` normalisation: text / json_object /
  json_schema dispatch + inner-schema unwrap.
- Schema validator: depth cap, property-count cap, pattern-length cap.
"""

from __future__ import annotations

import pytest

from hfl.api.structured_outputs import (
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_PROPERTIES,
    normalize_ollama_format,
    normalize_openai_response_format,
    validate_json_schema,
)
from hfl.exceptions import ValidationError


class TestOllamaFormatNormalisation:
    def test_none_passes_through(self):
        assert normalize_ollama_format(None) is None

    def test_empty_string_becomes_none(self):
        assert normalize_ollama_format("") is None
        assert normalize_ollama_format("   ") is None

    def test_json_literal(self):
        assert normalize_ollama_format("json") == "json"
        # case-insensitive
        assert normalize_ollama_format("JSON") == "json"

    def test_schema_object_returned_unchanged(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        assert normalize_ollama_format(schema) == schema

    def test_invalid_string_rejected(self):
        with pytest.raises(ValidationError):
            normalize_ollama_format("yaml")

    def test_non_string_non_dict_rejected(self):
        with pytest.raises(ValidationError):
            normalize_ollama_format(42)  # type: ignore[arg-type]

    def test_gbnf_passthrough(self):
        """Raw GBNF grammars (prefix ``GBNF:``) pass through so
        advanced users can ship custom grammars."""
        result = normalize_ollama_format("GBNF:root ::= object")
        assert isinstance(result, str) and result.startswith("GBNF:")


class TestOpenAIResponseFormatNormalisation:
    def test_none_passes_through(self):
        assert normalize_openai_response_format(None) is None

    def test_type_text_returns_none(self):
        """``{'type':'text'}`` means "no constraint"."""
        assert normalize_openai_response_format({"type": "text"}) is None

    def test_type_json_object_returns_json_literal(self):
        assert normalize_openai_response_format({"type": "json_object"}) == "json"

    def test_type_json_schema_returns_inner_schema(self):
        spec = {
            "type": "json_schema",
            "json_schema": {
                "name": "MySchema",
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }
        result = normalize_openai_response_format(spec)
        assert isinstance(result, dict)
        assert result == {"type": "object", "properties": {"x": {"type": "integer"}}}

    def test_non_dict_rejected(self):
        with pytest.raises(ValidationError):
            normalize_openai_response_format("json")  # type: ignore[arg-type]

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            normalize_openai_response_format({"type": "yaml"})

    def test_json_schema_missing_inner_rejected(self):
        with pytest.raises(ValidationError):
            normalize_openai_response_format({"type": "json_schema"})

    def test_json_schema_missing_schema_field_rejected(self):
        with pytest.raises(ValidationError):
            normalize_openai_response_format({"type": "json_schema", "json_schema": {"name": "x"}})


class TestJSONSchemaValidator:
    def test_simple_schema_passes(self):
        validate_json_schema({"type": "object", "properties": {"name": {"type": "string"}}})

    def test_deep_nesting_rejected(self):
        """A schema nested deeper than MAX_SCHEMA_DEPTH is rejected."""
        schema: dict = {"type": "object"}
        cursor = schema
        for _ in range(MAX_SCHEMA_DEPTH + 2):
            cursor["properties"] = {"child": {"type": "object"}}
            cursor = cursor["properties"]["child"]
        with pytest.raises(ValidationError, match="nesting"):
            validate_json_schema(schema)

    def test_property_count_cap(self):
        """More than MAX_SCHEMA_PROPERTIES total properties → 400."""
        schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {"type": "string"} for i in range(MAX_SCHEMA_PROPERTIES + 10)
            },
        }
        with pytest.raises(ValidationError, match="properties"):
            validate_json_schema(schema)

    def test_property_count_split_across_definitions(self):
        """Property count is summed across properties + definitions."""
        schema = {
            "type": "object",
            "properties": {f"f{i}": {"type": "string"} for i in range(150)},
            "definitions": {f"D{i}": {"type": "string"} for i in range(150)},
        }
        with pytest.raises(ValidationError, match="properties"):
            validate_json_schema(schema)

    def test_oversized_pattern_rejected(self):
        schema = {
            "type": "string",
            "pattern": "a" * 2048,  # exceeds MAX_PATTERN_LENGTH (1024)
        }
        with pytest.raises(ValidationError, match="pattern"):
            validate_json_schema(schema)

    def test_normal_pattern_accepted(self):
        validate_json_schema({"type": "string", "pattern": r"^\d{3}-\d{4}$"})

    def test_anyOf_allOf_oneOf_recursed_into(self):
        """Nested schemas under anyOf/allOf/oneOf also count toward depth."""
        schema: dict = {
            "anyOf": [{"type": "object"}],
        }
        cursor = schema["anyOf"][0]
        for _ in range(MAX_SCHEMA_DEPTH + 2):
            cursor["properties"] = {"child": {"type": "object"}}
            cursor = cursor["properties"]["child"]
        with pytest.raises(ValidationError, match="nesting"):
            validate_json_schema(schema)


class TestIntegrationWithGenerationConfig:
    """Smoke-test that response_format rides on GenerationConfig
    without breaking existing fields."""

    def test_config_default_is_none(self):
        from hfl.engine.base import GenerationConfig

        assert GenerationConfig().response_format is None

    def test_config_accepts_json_literal(self):
        from hfl.engine.base import GenerationConfig

        cfg = GenerationConfig(response_format="json")
        assert cfg.response_format == "json"

    def test_config_accepts_schema_dict(self):
        from hfl.engine.base import GenerationConfig

        schema = {"type": "object"}
        cfg = GenerationConfig(response_format=schema)
        assert cfg.response_format is schema

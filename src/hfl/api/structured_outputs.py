# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Structured-output helpers (OLLAMA_PARITY_PLAN P0-5).

Parsing + validation for the Ollama ``format`` and OpenAI
``response_format`` fields. These flow into ``GenerationConfig.
response_format`` which backends then compile to their native
constrained-decoding primitive (GBNF grammar for llama-cpp,
``GuidedDecodingParams`` for vLLM, ``outlines`` FSM for
Transformers).

We validate the schema at the router boundary so a malformed or
abusive schema (deep recursion, 10K properties) fails fast with
400 instead of hanging the engine.
"""

from __future__ import annotations

from typing import Any

from hfl.exceptions import ValidationError as APIValidationError

# ----------------------------------------------------------------------
# Schema safety limits — prevent DoS via pathological JSON Schemas.
# ----------------------------------------------------------------------

# Deepest nesting allowed in a schema. JSON Schemas of real-world
# shape rarely exceed 6 levels; 10 is a generous cap that still
# bounds recursion.
MAX_SCHEMA_DEPTH = 10

# Total number of ``properties`` / ``definitions`` / ``$defs`` keys
# summed across the schema. A legitimate schema with 200 fields is
# already unusual; beyond that the grammar compilation blows up
# quadratically.
MAX_SCHEMA_PROPERTIES = 200

# Maximum length of a ``pattern`` regex inside a schema — prevents
# ReDoS against the grammar compiler.
MAX_PATTERN_LENGTH = 1024


def normalize_ollama_format(value: str | dict | None) -> str | dict | None:
    """Normalise Ollama's ``format`` field.

    Ollama accepts three shapes:
    - ``"json"`` → free-form JSON output.
    - A JSON Schema object → constrained to the schema.
    - ``None`` / omitted → unconstrained.

    Returns the normalised value suitable for
    ``GenerationConfig.response_format``. Raises
    ``APIValidationError`` on malformed input.
    """
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "json":
            return "json"
        if v == "":
            return None
        # Raw GBNF passthrough for advanced users.
        if value.startswith("GBNF:"):
            return value
        raise APIValidationError(f"format must be 'json' or a JSON Schema object, got {value!r}")
    if isinstance(value, dict):
        validate_json_schema(value)
        return value
    raise APIValidationError(
        f"format must be a string, object, or null (got {type(value).__name__})"
    )


def normalize_openai_response_format(value: dict | None) -> str | dict | None:
    """Normalise OpenAI's ``response_format`` field.

    OpenAI accepts:
    - ``{"type": "text"}`` → unconstrained (treat as None).
    - ``{"type": "json_object"}`` → free-form JSON.
    - ``{"type": "json_schema", "json_schema": {"schema": {...}, ...}}``
      → strict schema-constrained output. We unwrap to the inner
      schema.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise APIValidationError(
            f"response_format must be an object or null (got {type(value).__name__})"
        )
    rf_type = value.get("type", "text")
    if rf_type == "text":
        return None
    if rf_type == "json_object":
        return "json"
    if rf_type == "json_schema":
        spec = value.get("json_schema")
        if not isinstance(spec, dict):
            raise APIValidationError("response_format.json_schema must be an object")
        schema = spec.get("schema")
        if not isinstance(schema, dict):
            raise APIValidationError(
                "response_format.json_schema.schema must be a JSON Schema object"
            )
        validate_json_schema(schema)
        return schema
    raise APIValidationError(
        f"response_format.type must be 'text', 'json_object', or 'json_schema' (got {rf_type!r})"
    )


def validate_json_schema(schema: dict) -> None:
    """Validate a JSON Schema for depth and breadth before compilation.

    Does NOT enforce Draft 7 / 2020-12 strict compliance (schemas that
    pass this are still accepted by llama-cpp's ``LlamaGrammar.
    from_json_schema``). The goal is to reject abusive inputs — deep
    recursion, enormous property lists, pathological regex patterns
    — that could hang the grammar compiler or eat memory.

    Raises:
        APIValidationError: The schema violates one of the caps.
    """
    _validate_schema_recursive(schema, depth=0, counters={"properties": 0, "patterns": 0})
    if not isinstance(schema, dict):
        raise APIValidationError("JSON Schema must be an object at the top level")


def _validate_schema_recursive(
    node: Any,
    *,
    depth: int,
    counters: dict[str, int],
) -> None:
    """Walk the schema once, enforcing depth + count limits."""
    if depth > MAX_SCHEMA_DEPTH:
        raise APIValidationError(f"JSON Schema nesting exceeds {MAX_SCHEMA_DEPTH} levels")

    if isinstance(node, dict):
        # Regex patterns are a classic ReDoS surface on grammar
        # compilers — bound the string length.
        pattern = node.get("pattern")
        if isinstance(pattern, str):
            counters["patterns"] += 1
            if len(pattern) > MAX_PATTERN_LENGTH:
                raise APIValidationError(
                    f'JSON Schema "pattern" exceeds {MAX_PATTERN_LENGTH} chars'
                )

        # Count properties across ``properties`` / ``definitions`` /
        # ``$defs``.
        for key in ("properties", "definitions", "$defs"):
            sub = node.get(key)
            if isinstance(sub, dict):
                counters["properties"] += len(sub)
                if counters["properties"] > MAX_SCHEMA_PROPERTIES:
                    raise APIValidationError(
                        f"JSON Schema exceeds {MAX_SCHEMA_PROPERTIES} total "
                        "properties (summed across properties/definitions)"
                    )
                for child in sub.values():
                    _validate_schema_recursive(child, depth=depth + 1, counters=counters)

        # Recurse into other structural keys.
        for key in ("items", "additionalItems", "contains", "not"):
            sub = node.get(key)
            if sub is not None:
                _validate_schema_recursive(sub, depth=depth + 1, counters=counters)

        for key in ("allOf", "anyOf", "oneOf"):
            arr = node.get(key)
            if isinstance(arr, list):
                for sub in arr:
                    _validate_schema_recursive(sub, depth=depth + 1, counters=counters)

    elif isinstance(node, list):
        for item in node:
            _validate_schema_recursive(item, depth=depth + 1, counters=counters)

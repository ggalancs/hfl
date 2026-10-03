# SPDX-License-Identifier: Apache-2.0
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

import re
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

# Largest repetition bound a schema may ask for (``minItems`` /
# ``maxItems`` / ``minLength`` / ``maxLength`` and the like, and a
# ``{m,n}`` quantifier in a ``pattern``). llama-cpp-python writes every
# repetition out in the grammar: ``maxLength: 10**7`` took 638 MB and
# ``10**9`` some 60 GB. Near 1000 optional repetitions llama.cpp refuses
# the grammar anyway (where exactly depends on the rest of it), and the
# in-process engine went down on a refused one: ``llama_cpp.py`` checks
# that the grammar builds before sampling.
MAX_SCHEMA_REPEAT = 1000

# A raw ``GBNF:`` grammar goes to llama.cpp as it is. Its parser recurses
# once per nested group: 100 000 nested parentheses (200 KB) crashed the
# in-process engine — the whole server — and killed llama-server.
MAX_GRAMMAR_BYTES = 64 * 1024
MAX_GRAMMAR_DEPTH = 256

# Keywords whose value is a subschema, a list of them, or a map of them.
# All of them are walked: a limit enforced on some keywords only is a
# limit a schema nests its way around.
_SUBSCHEMA_KEYS = (
    "items", "additionalItems", "contains", "not", "additionalProperties",
    "propertyNames", "if", "then", "else", "unevaluatedItems",
    "unevaluatedProperties",
)  # fmt: skip
_SUBSCHEMA_LIST_KEYS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SUBSCHEMA_MAP_KEYS = (
    "properties", "patternProperties", "definitions", "$defs", "dependentSchemas",
)  # fmt: skip
_REPEAT_KEYS = (
    "minItems", "maxItems", "minLength", "maxLength", "minProperties",
    "maxProperties", "minContains", "maxContains",
)  # fmt: skip
# A {m}, {m,} or {m,n} quantifier: its inside captured whole (one class that
# excludes "{", so each try stops at the next brace: linear; two \s* around an empty \d* were
# polynomial backtracking — CodeQL py/polynomial-redos), then read in Python.
_QUANTIFIER_RE = re.compile(r"\{([\d\s,]+)\}")


def _quantifier_bounds(inside: str) -> list[str]:
    """The numbers of a quantifier's inside ("2", "2,", "2, 5"); [] if it is
    not one (a literal brace)."""
    parts = [part.strip() for part in inside.split(",")]
    if len(parts) > 2 or not all(part.isdigit() or part == "" for part in parts):
        return []
    return [part for part in parts if part]


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
            validate_gbnf(value[len("GBNF:") :])
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


def validate_gbnf(grammar: str) -> None:
    """Bound a raw GBNF grammar before llama.cpp parses it: its size, and
    how deep its groups nest (``(``, ``[``, ``{`` outside string literals,
    character classes and comments).

    Raises:
        APIValidationError: The grammar is too large or nests too deep.
    """
    if len(grammar.encode("utf-8")) > MAX_GRAMMAR_BYTES:
        raise APIValidationError(f"GBNF grammar exceeds {MAX_GRAMMAR_BYTES} bytes")
    depth = 0
    i, n = 0, len(grammar)
    while i < n:
        ch = grammar[i]
        if ch in '"[':
            # A literal runs to its unescaped closer; nothing inside it nests.
            closer = '"' if ch == '"' else "]"
            i += 1
            while i < n and grammar[i] != closer:
                i += 2 if grammar[i] == "\\" else 1
        elif ch == "#":
            newline = grammar.find("\n", i)
            i = n if newline == -1 else newline
        elif ch in "({":
            depth += 1
            if depth > MAX_GRAMMAR_DEPTH:
                raise APIValidationError(f"GBNF grammar nesting exceeds {MAX_GRAMMAR_DEPTH} levels")
        elif ch in ")}":
            depth = max(0, depth - 1)
        i += 1


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
            # The grammar spells a quantifier out like any other repetition.
            for match in _QUANTIFIER_RE.finditer(pattern):
                bounds = _quantifier_bounds(match.group(1))
                if any(len(b) > 4 or int(b) > MAX_SCHEMA_REPEAT for b in bounds):
                    raise APIValidationError(
                        f'JSON Schema "pattern" repeats more than {MAX_SCHEMA_REPEAT} times'
                    )

        for key in _REPEAT_KEYS:
            bound = node.get(key)
            if (
                isinstance(bound, (int, float))
                and not isinstance(bound, bool)
                and bound > MAX_SCHEMA_REPEAT
            ):
                raise APIValidationError(f'JSON Schema "{key}" exceeds {MAX_SCHEMA_REPEAT}')

        # Count properties across ``properties`` / ``definitions`` /
        # ``$defs`` and the other maps of subschemas.
        for key in _SUBSCHEMA_MAP_KEYS:
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

        # ``dependencies`` maps a property to a subschema or to a list of
        # property names (strings, which the walk passes over).
        for key in (*_SUBSCHEMA_KEYS, *_SUBSCHEMA_LIST_KEYS, "dependencies"):
            sub = node.get(key)
            if isinstance(sub, dict) and key == "dependencies":
                for child in sub.values():
                    _validate_schema_recursive(child, depth=depth + 1, counters=counters)
            elif sub is not None:
                _validate_schema_recursive(sub, depth=depth + 1, counters=counters)

    elif isinstance(node, list):
        for item in node:
            _validate_schema_recursive(item, depth=depth + 1, counters=counters)

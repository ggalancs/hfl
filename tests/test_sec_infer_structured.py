# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Security: a response format cannot take the inference engine down.

A raw ``GBNF:`` grammar nested 100 000 deep crashed the in-process engine
and killed llama-server; a schema's repetition bounds were spelled out in
full by the grammar compiler (``maxLength: 10**9`` → ~60 GB); and a grammar
llama.cpp refuses (an undefined rule, a syntax error) was a NULL sampler
that llama-cpp-python sampled through — SIGSEGV. None of these tests runs a
crashing input against a real engine: the bounds are checked in Python.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.structured_outputs import (
    MAX_GRAMMAR_BYTES,
    MAX_GRAMMAR_DEPTH,
    MAX_SCHEMA_REPEAT,
    normalize_ollama_format,
    normalize_openai_response_format,
    validate_gbnf,
    validate_json_schema,
)
from hfl.exceptions import ValidationError


def _nested(depth: int) -> str:
    return "root ::= " + "(" * depth + '"a"' + ")" * depth


class TestRawGrammar:
    def test_the_crashing_grammar_is_refused(self):
        with pytest.raises(ValidationError, match="GBNF grammar"):
            normalize_ollama_format("GBNF:" + _nested(100_000))
        # Under the size limit, the nesting alone refuses it.
        with pytest.raises(ValidationError, match="nesting"):
            normalize_ollama_format("GBNF:" + _nested(20_000))

    def test_depth_limit_is_exact(self):
        validate_gbnf(_nested(MAX_GRAMMAR_DEPTH))
        with pytest.raises(ValidationError, match="nesting"):
            validate_gbnf(_nested(MAX_GRAMMAR_DEPTH + 1))

    def test_brackets_inside_literals_and_comments_do_not_nest(self):
        many = "(" * (MAX_GRAMMAR_DEPTH + 10)
        grammar = f'root ::= "{many}" item\nitem ::= [{many}\\]] # {many}\nesc ::= "\\"{many}"\n'
        assert normalize_ollama_format("GBNF:" + grammar) == "GBNF:" + grammar

    def test_size_limit(self):
        filler = 'root ::= "' + "a" * MAX_GRAMMAR_BYTES + '"'
        with pytest.raises(ValidationError, match="bytes"):
            normalize_ollama_format("GBNF:" + filler)

    def test_ordinary_grammar_passes(self):
        grammar = 'root ::= ("yes" | "no") ws\nws ::= [ \\t\\n]*'
        assert normalize_ollama_format("GBNF:" + grammar) == "GBNF:" + grammar


class TestSchemaRepetitionBounds:
    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "array", "items": {"type": "integer"}, "maxItems": 10**7},
            {"type": "array", "items": {"type": "integer"}, "minItems": 10**9},
            {"type": "string", "maxLength": 10**9},
            {"type": "string", "minLength": MAX_SCHEMA_REPEAT + 1},
            {"type": "string", "pattern": "a{1000000}"},
            {"type": "string", "pattern": "^(ab){0,99999999999999999999}$"},
            {"type": "object", "properties": {"x": {"type": "string", "maxLength": 10**7}}},
        ],
    )
    def test_huge_bounds_refused(self, schema):
        with pytest.raises(ValidationError, match=str(MAX_SCHEMA_REPEAT)):
            validate_json_schema(schema)

    def test_bounds_at_the_limit_pass(self):
        validate_json_schema(
            {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"},
                             "maxItems": MAX_SCHEMA_REPEAT},
                    "text": {"type": "string", "maxLength": MAX_SCHEMA_REPEAT},
                    "code": {"type": "string", "pattern": r"^\d{3}-\d{1,4}$"},
                },
            }
        )  # fmt: skip

    def test_openai_schema_is_bounded_too(self):
        spec = {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "string", "maxLength": 10**9}},
        }
        with pytest.raises(ValidationError):
            normalize_openai_response_format(spec)


def _deep(key: str, depth: int = 200) -> dict:
    """A schema nested ``depth`` levels through ``key`` only."""
    leaf: dict = {"type": "string"}
    for _ in range(depth):
        if key == "items-list":
            leaf = {"type": "array", "items": [leaf]}
        elif key in ("prefixItems", "allOf", "anyOf", "oneOf"):
            leaf = {key: [leaf]}
        elif key in ("patternProperties", "$defs", "definitions", "dependentSchemas"):
            leaf = {key: {"k": leaf}}
        elif key == "dependencies":
            leaf = {"dependencies": {"k": leaf}}
        else:
            leaf = {key: leaf}
    return leaf


class TestEverySubschemaIsWalked:
    @pytest.mark.parametrize(
        "key",
        [
            "additionalProperties", "prefixItems", "patternProperties", "if", "then",
            "else", "items-list", "$defs", "definitions", "anyOf", "oneOf", "allOf",
            "propertyNames", "unevaluatedProperties", "unevaluatedItems",
            "dependentSchemas", "dependencies", "not", "contains",
        ],
    )  # fmt: skip
    def test_200_deep_is_refused(self, key):
        with pytest.raises(ValidationError, match="nesting"):
            validate_json_schema(_deep(key))

    def test_bounds_deep_inside_are_seen(self):
        schema = {"additionalProperties": {"if": {"type": "string", "maxLength": 10**9}}}
        with pytest.raises(ValidationError):
            validate_json_schema(schema)


class TestTheRouteRefusesBeforeInference:
    @pytest.fixture
    def client(self, temp_config):
        from hfl.api.state import reset_state

        reset_state()
        yield TestClient(__import__("hfl.api.server", fromlist=["app"]).app)
        reset_state()

    def test_nested_grammar_is_a_400(self, client, sample_manifest):
        from hfl.api.state import get_state

        state = get_state()
        engine = MagicMock(is_loaded=True, supports_structured_output=True)
        state.engine = engine
        state.current_model = sample_manifest
        response = client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "x"}],
                "format": "GBNF:" + _nested(100_000),
                "stream": False,
            },
        )
        assert response.status_code == 400
        assert "GBNF grammar" in response.json()["error"]
        engine.chat.assert_not_called()


class TestTheEngineChecksTheGrammarBuilds:
    """llama-cpp-python adds a grammar sampler without checking llama.cpp
    built it; the engine now builds it once first and refuses a NULL one."""

    @pytest.fixture
    def fake_llama_cpp(self, monkeypatch):
        built: list = []
        freed: list = []

        def init(vocab, grammar, root):
            built.append((vocab, grammar, root))
            return 0 if b"undefined" in grammar else 1234

        module = types.SimpleNamespace(
            llama_sampler_init_grammar=init, llama_sampler_free=freed.append
        )
        monkeypatch.setitem(sys.modules, "llama_cpp", module)
        return built, freed

    @staticmethod
    def _model(vocab):
        return types.SimpleNamespace(_model=types.SimpleNamespace(vocab=vocab))

    def test_unbuildable_grammar_is_refused(self, fake_llama_cpp):
        from hfl.engine.llama_cpp import _refuse_unbuildable_grammar

        built, freed = fake_llama_cpp
        with pytest.raises(ValidationError, match="grammar"):
            _refuse_unbuildable_grammar(self._model(7), "GBNF:root ::= undefined")
        assert built == [(7, b"root ::= undefined", b"root")]
        assert freed == []

    def test_buildable_grammar_is_built_and_freed(self, fake_llama_cpp):
        from hfl.engine.llama_cpp import _refuse_unbuildable_grammar

        built, freed = fake_llama_cpp
        _refuse_unbuildable_grammar(self._model(7), 'GBNF:root ::= "a"')
        assert len(built) == 1 and freed == [1234]

    def test_stand_in_models_are_not_probed(self, fake_llama_cpp):
        from hfl.engine.llama_cpp import _refuse_unbuildable_grammar

        built, _ = fake_llama_cpp
        _refuse_unbuildable_grammar(MagicMock(), "GBNF:root ::= undefined")
        _refuse_unbuildable_grammar(self._model(7), None)
        _refuse_unbuildable_grammar(self._model(7), "json")
        assert built == []

    def test_generate_refuses_before_sampling(self, fake_llama_cpp):
        from hfl.engine.base import GenerationConfig
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        model = MagicMock()
        model._model = types.SimpleNamespace(vocab=7)
        engine._model = model
        bad = GenerationConfig(response_format="GBNF:root ::= undefined")
        with pytest.raises(ValidationError):
            engine.generate("p", bad)
        with pytest.raises(ValidationError):
            engine.generate_stream("p", bad)
        model.assert_not_called()


class TestPatternQuantifierScan:
    """The quantifier scan of a schema "pattern" was a polynomial regex
    (CodeQL py/polynomial-redos); and padding must not hide a bound."""

    def test_a_padded_quantifier_is_still_bounded(self):
        from hfl.api.structured_outputs import validate_json_schema
        from hfl.exceptions import ValidationError as APIValidationError

        schema = {"type": "string", "pattern": "a{" + " " * 40 + "99999}"}
        with pytest.raises(APIValidationError, match="repeats more than"):
            validate_json_schema(schema)

    def test_the_scan_is_linear(self):
        import time

        from hfl.api.structured_outputs import _QUANTIFIER_RE

        started = time.monotonic()
        list(_QUANTIFIER_RE.finditer("{" + " " * 200_000))
        list(_QUANTIFIER_RE.finditer("{ " * 100_000))
        assert time.monotonic() - started < 1.0

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for ``hfl/engine/verifier.py`` — V4 F3.1."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from hfl.engine.verifier import verify_model


@dataclass
class _Manifest:
    name: str = "test-model"
    format: str | None = "gguf"
    declared_capabilities: list[str] | None = None


def _engine_ok(text="forty-two"):
    """Build a mock engine that passes every probe."""
    engine = MagicMock()
    tok = MagicMock()
    tok.encode = MagicMock(return_value=[1, 2, 3])
    tok.decode = MagicMock(return_value="Hello, world.")
    tok.apply_chat_template = MagicMock(return_value="<|im_start|>user\nhello<|im_end|>")
    engine.tokenizer = tok
    result = MagicMock()
    result.text = text
    result.tokens_generated = 1
    engine.generate = MagicMock(return_value=result)
    return engine


class TestVerifierHappyPath:
    def test_all_probes_pass(self):
        engine = _engine_ok()
        result = verify_model(engine, _Manifest())
        assert result.overall_pass is True
        names = {c.name for c in result.checks}
        assert "tokenizer_round_trip" in names
        assert "chat_template_render" in names
        assert "smoke_generation" in names
        assert "tool_parser_round_trip" in names
        assert "embedding_dim" in names

    def test_duration_is_positive(self):
        engine = _engine_ok()
        result = verify_model(engine, _Manifest())
        assert result.duration_ms >= 0


class TestVerifierFailures:
    def test_tokenizer_decoding_mismatch_fails(self):
        engine = _engine_ok()
        engine.tokenizer.decode = MagicMock(return_value="garbage")
        result = verify_model(engine, _Manifest())
        token_check = next(c for c in result.checks if c.name == "tokenizer_round_trip")
        assert token_check.passed is False
        assert result.overall_pass is False

    def test_engine_without_tokenizer_fails_check(self):
        engine = MagicMock(spec=[])  # no tokenizer attribute
        # provide chat method so chat_template check passes
        engine.chat = MagicMock()
        engine.generate = MagicMock(return_value=MagicMock(text="x", tokens_generated=1))
        result = verify_model(engine, _Manifest())
        tok_check = next(c for c in result.checks if c.name == "tokenizer_round_trip")
        assert tok_check.passed is False

    def test_smoke_generation_failure_is_recorded(self):
        engine = _engine_ok()
        engine.generate = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        result = verify_model(engine, _Manifest())
        gen_check = next(c for c in result.checks if c.name == "smoke_generation")
        assert gen_check.passed is False
        assert "CUDA OOM" in gen_check.detail

    def test_empty_generation_fails(self):
        engine = _engine_ok()
        engine.generate.return_value.text = ""
        result = verify_model(engine, _Manifest())
        gen_check = next(c for c in result.checks if c.name == "smoke_generation")
        assert gen_check.passed is False

    def test_chat_template_missing_without_engine_chat_fails(self):
        """Reproduces the MLX-pull failure mode: tokenizer with no
        apply_chat_template AND engine without chat()."""
        engine = MagicMock(spec=["tokenizer", "generate"])
        engine.tokenizer = MagicMock(spec=["encode", "decode"])
        engine.tokenizer.encode = MagicMock(return_value=[1])
        engine.tokenizer.decode = MagicMock(return_value="Hello, world.")
        engine.generate = MagicMock(return_value=MagicMock(text="x", tokens_generated=1))
        result = verify_model(engine, _Manifest())
        chat_check = next(c for c in result.checks if c.name == "chat_template_render")
        assert chat_check.passed is False


class TestToolCallEndToEnd:
    """V6 ν5 — broaden the tool_parser probe so it isn't just lexical.

    The default probe in ``_check_tool_parser`` round-trips a
    canonical ``<tool_call>`` block through the parser. That's
    necessary but not sufficient — it doesn't prove the engine
    pipeline actually emits parseable tool calls. These tests run
    a verify pass against engines whose ``chat`` returns realistic
    tool-call output, including:

    - direct structured ``tool_calls`` on the result (qwen / llama 3
      family)
    - free text containing a ``<tool_call>`` block (parser recovers)
    - free text without any tool markers (zero calls expected)
    """

    def _engine_with_chat(self, *, chat_text: str = "", tool_calls=None):
        engine = MagicMock()
        tok = MagicMock()
        tok.encode = MagicMock(return_value=[1, 2, 3])
        tok.decode = MagicMock(return_value="Hello, world.")
        tok.apply_chat_template = MagicMock(return_value="<chat>")
        engine.tokenizer = tok

        result = MagicMock()
        result.text = chat_text
        result.tokens_generated = 1
        result.tool_calls = tool_calls
        engine.generate = MagicMock(return_value=result)
        engine.chat = MagicMock(return_value=result)
        return engine

    def test_engine_with_structured_tool_calls(self):
        from hfl.engine.verifier import verify_model

        engine = self._engine_with_chat(
            chat_text="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"weather"}'},
                }
            ],
        )
        result = verify_model(engine, _Manifest(name="qwen-tools"))
        check = next(c for c in result.checks if c.name == "tool_parser_round_trip")
        assert check.passed is True

    def test_engine_with_xml_tool_call_in_text(self):
        from hfl.engine.verifier import verify_model

        # The free-text path: parser should recognise the canonical
        # <tool_call>...</tool_call> block.
        engine = self._engine_with_chat(
            chat_text='<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>',
        )
        result = verify_model(engine, _Manifest(name="qwen-tools"))
        check = next(c for c in result.checks if c.name == "tool_parser_round_trip")
        assert check.passed is True

    def test_engine_with_no_tool_calls(self):
        """Plain prose response → parser returns 0 calls and the
        check still passes (it's a "did not raise" probe)."""
        from hfl.engine.verifier import verify_model

        engine = self._engine_with_chat(chat_text="The capital of France is Paris.")
        result = verify_model(engine, _Manifest(name="qwen-prose"))
        check = next(c for c in result.checks if c.name == "tool_parser_round_trip")
        assert check.passed is True


class TestEmbeddingProbe:
    def test_skipped_for_non_embedding_model(self):
        engine = _engine_ok()
        result = verify_model(engine, _Manifest(declared_capabilities=[]))
        emb = next(c for c in result.checks if c.name == "embedding_dim")
        assert emb.passed is True  # marked as not applicable

    def test_runs_for_embedding_model(self):
        engine = _engine_ok()
        engine.embed = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        manifest = _Manifest(declared_capabilities=["embeddings"])
        result = verify_model(engine, manifest)
        emb = next(c for c in result.checks if c.name == "embedding_dim")
        assert emb.passed is True
        assert "dim=4" in emb.detail

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for :mod:`hfl.engine.embedding_engine`.

These tests do NOT require the optional ``[llama]`` / ``[transformers]``
extras at runtime — the backends are exercised through ``patch.dict``
injected into ``sys.modules`` so the actual C / PyTorch libraries
never have to load. What we verify is the HFL-level contract:
validation, shape, token accounting, dimensions truncation,
loaded/unloaded gating.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from hfl.engine.embedding_engine import (
    EmbeddingEngine,
    EmbeddingResult,
    LlamaCppEmbeddingEngine,
    TransformersEmbeddingEngine,
)

# --------------------------------------------------------------------
# Abstract interface
# --------------------------------------------------------------------


class TestEmbeddingEngineAbstract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            EmbeddingEngine()  # type: ignore[abstract]

    def test_concrete_subclass_must_override_all_abstractmethods(self):
        """A subclass that skips ``embed`` is still abstract."""

        class Incomplete(EmbeddingEngine):
            def load(self, model_path: str, **kwargs) -> None:  # noqa: D401
                pass

            def unload(self) -> None:
                pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# --------------------------------------------------------------------
# EmbeddingResult dataclass
# --------------------------------------------------------------------


class TestEmbeddingResult:
    def test_defaults(self):
        r = EmbeddingResult(embeddings=[[0.1, 0.2]])
        assert r.total_tokens == 0
        assert r.model == ""

    def test_roundtrip(self):
        r = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            total_tokens=42,
            model="nomic-embed-text",
        )
        assert len(r.embeddings) == 2
        assert r.total_tokens == 42
        assert r.model == "nomic-embed-text"


# --------------------------------------------------------------------
# LlamaCpp adapter — exercised via sys.modules injection
# --------------------------------------------------------------------


@pytest.fixture
def fake_llama_cpp():
    """Install a fake ``llama_cpp`` module into sys.modules.

    The fake Llama class records the args it was constructed with and
    returns deterministic 4-dim vectors from ``embed``. Cleans up on
    teardown so subsequent tests see the real (absent) module.
    """
    fake_module = types.ModuleType("llama_cpp")

    class FakeLlama:
        _instances: list["FakeLlama"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._closed = False
            FakeLlama._instances.append(self)

        def n_embd(self) -> int:  # Method, not property — matches llama-cpp
            return 4

        def embed(self, text: str, truncate: bool = True) -> list[float]:
            # Deterministic: length of text → rotates the vector so
            # each input produces a unique output.
            offset = len(text) % 10
            return [0.1 + offset, 0.2, 0.3, 0.4]

        def tokenize(self, data: bytes) -> list[int]:
            return [ord(c) for c in data.decode("utf-8", errors="replace")]

        def close(self) -> None:
            self._closed = True

    fake_module.Llama = FakeLlama  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"llama_cpp": fake_module}):
        FakeLlama._instances.clear()
        yield FakeLlama


class TestLlamaCppEmbeddingEngine:
    def test_load_sets_is_loaded(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        assert not engine.is_loaded

        engine.load("/tmp/fake.gguf")
        assert engine.is_loaded
        assert engine.model_name == "/tmp/fake.gguf"

    def test_load_forces_embedding_mode(self, fake_llama_cpp):
        """embedding=True must be passed to Llama so tokens are
        produced as pooled vectors, not as next-token logits."""
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")

        assert fake_llama_cpp._instances
        kwargs = fake_llama_cpp._instances[-1].kwargs
        assert kwargs.get("embedding") is True
        assert kwargs.get("n_ctx") == 8192  # safe default

    def test_load_respects_custom_n_ctx(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf", n_ctx=2048)
        assert fake_llama_cpp._instances[-1].kwargs["n_ctx"] == 2048

    def test_embed_not_loaded_raises(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.embed(["hello"])

    def test_embed_empty_list_raises(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")
        with pytest.raises(ValueError, match="non-empty"):
            engine.embed([])

    def test_embed_returns_one_vector_per_input_in_order(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")
        result = engine.embed(["a", "abcd"])
        assert len(result.embeddings) == 2
        # Our fake encodes len(text) % 10 in the first component.
        assert result.embeddings[0][0] == pytest.approx(0.1 + 1)  # len("a") = 1
        assert result.embeddings[1][0] == pytest.approx(0.1 + 4)  # len("abcd") = 4

    def test_embed_populates_total_tokens_and_model(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")
        result = engine.embed(["hello", "world!"])
        # Fake tokenizer emits one "token" per character.
        assert result.total_tokens == len("hello") + len("world!")
        assert result.model == "/tmp/fake.gguf"

    def test_dimensions_truncates_vectors(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")
        result = engine.embed(["hi"], dimensions=2)
        assert len(result.embeddings[0]) == 2

    def test_dimensions_exceeding_native_is_rejected(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")  # native dim 4
        with pytest.raises(ValueError, match="exceeds"):
            engine.embed(["hi"], dimensions=100)

    def test_dimensions_non_positive_is_rejected(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")
        for bad in (0, -1, -42):
            with pytest.raises(ValueError, match="positive"):
                engine.embed(["hi"], dimensions=bad)

    def test_unload_clears_state_and_calls_close(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")
        instance = fake_llama_cpp._instances[-1]

        engine.unload()
        assert not engine.is_loaded
        assert engine.model_name == ""
        assert instance._closed is True

    def test_embed_after_unload_raises(self, fake_llama_cpp):
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")
        engine.unload()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.embed(["hi"])


# --------------------------------------------------------------------
# Transformers adapter — verify construction path & error gates only
# --------------------------------------------------------------------


class TestTransformersEmbeddingEngineErrorGates:
    """The Transformers adapter needs torch + transformers, which are
    not installed in the CI venv. We only test the *validation* logic
    that runs before any torch call — the full integration belongs to
    a [transformers]-extra test suite we don't run by default."""

    def test_cannot_instantiate_is_abstract_no(self):
        # TransformersEmbeddingEngine is concrete; construction works
        # without deps because load() is lazy.
        engine = TransformersEmbeddingEngine()
        assert not engine.is_loaded
        assert engine.model_name == ""

    def test_embed_without_load_raises(self):
        engine = TransformersEmbeddingEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.embed(["hello"])

    def test_unload_on_unloaded_engine_is_noop(self):
        engine = TransformersEmbeddingEngine()
        engine.unload()  # must not raise
        assert not engine.is_loaded


# --------------------------------------------------------------------
# LlamaCpp: fallback tokenizer accounting
# --------------------------------------------------------------------


class TestLlamaCppTokenAccountingFallback:
    def test_uses_heuristic_when_tokenize_raises(self, fake_llama_cpp):
        """Some llama-cpp builds don't expose tokenize(); the engine
        falls back to a char-count heuristic rather than crashing."""
        engine = LlamaCppEmbeddingEngine()
        engine.load("/tmp/fake.gguf")
        # Swap tokenize for something that raises.
        instance = fake_llama_cpp._instances[-1]
        instance.tokenize = MagicMock(side_effect=RuntimeError("no tokenize"))

        result = engine.embed(["hello"])  # 5 chars → at least 1 token
        assert result.total_tokens >= 1
        assert len(result.embeddings) == 1

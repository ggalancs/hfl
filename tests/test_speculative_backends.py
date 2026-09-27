# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A Modelfile DRAFT on llama-server and MLX, not only llama-cpp-python
(plan 0.22 P1-6). Measured for real by audit E14/E15: tokens accepted from
the draft, greedy output unchanged."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from hfl.engine import llama_server


@pytest.fixture
def help_text(monkeypatch):
    def with_help(text: str) -> None:
        monkeypatch.setattr(llama_server, "_help_text", lambda exe: text)

    return with_help


def test_a_draft_gguf_is_loaded_and_used(tmp_path, help_text) -> None:
    """Builds with --spec-type default it to none: -md alone loads a draft
    that is never used (found measuring it)."""
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"GGUF")
    help_text("--spec-type none,draft-simple,ngram-simple")
    assert llama_server._speculative_args("llama-server", str(draft), 999) == [
        "-md",
        str(draft),
        "-ngld",
        "999",
        "--spec-type",
        "draft-simple",
    ]


def test_an_older_build_gets_md_only(tmp_path, help_text) -> None:
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"GGUF")
    help_text("--draft-max N")
    assert llama_server._speculative_args("llama-server", str(draft), 0) == [
        "-md",
        str(draft),
        "-ngld",
        "0",
    ]


def test_prompt_lookup_where_the_build_has_it(help_text) -> None:
    help_text("--spec-type none,ngram-simple")
    assert llama_server._speculative_args("x", "prompt-lookup", 0) == [
        "--spec-type",
        "ngram-simple",
    ]
    help_text("")
    assert llama_server._speculative_args("x", "prompt-lookup", 0) == []  # would not start


@pytest.mark.parametrize("draft", [None, "", "/models/folder-of-safetensors"])
def test_no_draft_or_not_a_gguf(draft, help_text) -> None:
    help_text("--spec-type none,draft-simple")
    assert llama_server._speculative_args("x", draft, 0) == []


# -- MLX (mlx-lm replaced by a stand-in: the logic, not the kernels) ---------


def _mlx_engine(monkeypatch, draft_vocab: int = 100):
    from hfl.engine.mlx_engine import MLXEngine

    fake = types.ModuleType("mlx_lm")
    fake.load = lambda path: (f"model:{path}", SimpleNamespace(vocab_size=draft_vocab))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    engine = MLXEngine()
    engine._tokenizer = SimpleNamespace(vocab_size=100)
    return engine


def test_mlx_takes_a_draft_with_the_models_tokenizer(monkeypatch) -> None:
    assert _mlx_engine(monkeypatch)._load_draft("/d") == "model:/d"


@pytest.mark.parametrize(
    ("spec", "vocab"),
    [("/d", 999), ("prompt-lookup", 100), (None, 100)],
    ids=["other-tokenizer", "prompt-lookup", "none"],
)
def test_mlx_refuses_what_it_cannot_use(monkeypatch, spec, vocab) -> None:
    assert _mlx_engine(monkeypatch, vocab)._load_draft(spec) is None


def test_mlx_counts_tokens_from_the_draft(monkeypatch) -> None:
    engine = _mlx_engine(monkeypatch)
    steps = [
        SimpleNamespace(from_draft=True, finish_reason=None),
        SimpleNamespace(from_draft=False, finish_reason=None),
        SimpleNamespace(from_draft=True, finish_reason=None),
        SimpleNamespace(from_draft=True, finish_reason="stop"),  # repeats the last token
    ]
    assert len(list(engine._cancellable(iter(steps)))) == 4
    assert engine.last_draft_tokens == (2, 3)


def test_mlx_passes_the_draft_to_generation(monkeypatch) -> None:
    pytest.importorskip("mlx_lm.sample_utils")
    from hfl.engine.base import GenerationConfig
    from hfl.engine.mlx_engine import MLXEngine

    engine = MLXEngine()
    assert "draft_model" not in engine._build_sampling(GenerationConfig())
    engine._draft = "draft"
    assert engine._build_sampling(GenerationConfig())["draft_model"] == "draft"

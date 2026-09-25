# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Every engine says when max_tokens cut a reply.

LlamaCppEngine.chat, the MLX and Transformers generate paths and vLLM's
synchronous path all returned stop_reason "stop" regardless, so a client
got a truncated answer reported as finished. The routes turn stop_reason
into OpenAI's finish_reason, Ollama's done_reason and Anthropic's
stop_reason.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hfl.engine.base import GenerationConfig


@pytest.mark.parametrize(("n_gen", "expected"), [(16, "length"), (5, "stop")])
def test_mlx(monkeypatch, n_gen, expected):
    from hfl.engine.mlx_engine import MLXEngine

    engine = MLXEngine()
    engine._model, engine._tokenizer = object(), object()
    monkeypatch.setattr(engine, "_prompt_store", None, raising=False)
    monkeypatch.setattr(engine, "_run_generate", lambda prompt, cfg: ("text", 7, n_gen, 1_000))
    assert engine.generate("x", GenerationConfig(max_tokens=16)).stop_reason == expected


@pytest.mark.parametrize(("n_gen", "expected"), [(16, "length"), (5, "stop")])
def test_transformers(n_gen, expected):
    torch = pytest.importorskip("torch")
    from hfl.engine.transformers_engine import TransformersEngine

    class Tokenizer:
        def __call__(self, prompt, return_tensors):
            ids = torch.zeros((1, 3), dtype=torch.long)
            return SimpleNamespace(to=lambda device: {"input_ids": ids}, input_ids=ids)

        def decode(self, tokens, skip_special_tokens):
            return "x" * len(tokens)

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            return torch.zeros((1, 3 + n_gen), dtype=torch.long)

    engine = TransformersEngine()
    engine._model, engine._tokenizer = Model(), Tokenizer()
    result = engine.generate("x", GenerationConfig(max_tokens=16, temperature=0))
    assert result.stop_reason == expected


@pytest.mark.parametrize(("finish", "expected"), [("length", "length"), ("stop", "stop")])
def test_vllm_sync(finish, expected):
    from hfl.engine.vllm_engine import VLLMEngine

    engine = VLLMEngine.__new__(VLLMEngine)
    completion = SimpleNamespace(text="t", token_ids=[1, 2], finish_reason=finish)
    engine._engine = SimpleNamespace(
        generate=lambda prompts, params: [SimpleNamespace(outputs=[completion])]
    )
    assert engine._generate_sync("x", None).stop_reason == expected

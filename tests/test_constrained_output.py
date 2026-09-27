# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Structured output on MLX and Transformers through llguidance (plan 0.22
P1-8); both used to answer a response format with a 400. Measured with
Qwen2.5-0.5B on both engines: a schema gives valid, complete JSON; a GBNF
grammar ``"yes" | "no"`` gives ``yes``."""

from __future__ import annotations

import json

import numpy as np
import pytest

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "n": {"type": "integer"}},
    "required": ["city", "n"],
    "additionalProperties": False,
}


def test_without_the_extra_both_engines_say_so(monkeypatch) -> None:
    import hfl.engine.constrained as constrained
    from hfl.engine.mlx_engine import MLXEngine
    from hfl.engine.transformers_engine import TransformersEngine

    monkeypatch.setattr(constrained, "available", lambda: False)
    assert not MLXEngine().supports_structured_output
    assert not TransformersEngine().supports_structured_output
    monkeypatch.setattr(constrained, "available", lambda: True)
    assert MLXEngine().supports_structured_output
    assert TransformersEngine().supports_structured_output


def test_the_refusal_names_the_extra() -> None:
    from types import SimpleNamespace

    from hfl.api.errors import structured_output_unsupported

    refused = structured_output_unsupported(
        SimpleNamespace(supports_structured_output=False),
        SimpleNamespace(response_format="json"),
        "/api/chat",
    )
    assert refused is not None and refused.status_code == 400
    assert "hfl[structured]" in refused.body.decode()


# -- the mask itself, over a real (tiny) tokenizer ----------------------------


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("llguidance")
    transformers = pytest.importorskip("transformers")
    from tokenizers import ByteLevelBPETokenizer

    corpus = ['{"city": "Barcelona", "n": 12}', "hello world", "yes no maybe"] * 50
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=400, special_tokens=["<eos>"])
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer)
    fast.eos_token = "<eos>"
    return fast


def _sample(tokenizer, response_format, seed: int, steps: int = 200) -> str:
    """Random logits (a model that knows nothing) through the guide: what
    comes out can only be what the format allows."""
    from hfl.engine.constrained import Guide

    eos = tokenizer.eos_token_id
    guide = Guide(tokenizer, response_format, [eos])
    rng = np.random.default_rng(seed)
    n_vocab = len(tokenizer)
    drawn: list[int] = []
    last = None
    for _ in range(steps):
        mask = guide.mask(last, n_vocab)
        logits = rng.normal(size=(1, n_vocab)).astype(np.float32)
        logits[0, eos] += 4.0  # prefer stopping as soon as the format allows it
        from llguidance.numpy import apply_token_bitmask_inplace

        apply_token_bitmask_inplace(logits, mask)
        last = int(np.argmax(logits[0]))
        if last == eos:
            break
        drawn.append(last)
    return tokenizer.decode(drawn)


@pytest.mark.parametrize("seed", range(5))
def test_a_schema_is_obeyed_even_by_noise(tokenizer, seed) -> None:
    value = json.loads(_sample(tokenizer, SCHEMA, seed))
    assert set(value) == {"city", "n"}
    assert isinstance(value["city"], str) and isinstance(value["n"], int)


@pytest.mark.parametrize("seed", range(3))
def test_json_is_an_object(tokenizer, seed) -> None:
    assert isinstance(json.loads(_sample(tokenizer, "json", seed)), dict)


@pytest.mark.parametrize("seed", range(3))
def test_a_gbnf_grammar(tokenizer, seed) -> None:
    assert _sample(tokenizer, 'GBNF:root ::= "yes" | "no"', seed) in ("yes", "no")


def test_unknown_formats_are_refused() -> None:
    pytest.importorskip("llguidance")
    from hfl.engine.constrained import grammar_for

    with pytest.raises(ValueError):
        grammar_for("xml")


def test_transformers_generate_gets_the_processor() -> None:
    pytest.importorskip("llguidance")
    pytest.importorskip("transformers")
    from hfl.engine.base import GenerationConfig
    from hfl.engine.transformers_engine import TransformersEngine

    engine = TransformersEngine()
    assert engine._constraint(GenerationConfig()) == {}
    engine._tokenizer = None
    engine._model = type("M", (), {"generation_config": type("G", (), {"eos_token_id": 2})()})()
    kwargs = engine._constraint(GenerationConfig(response_format=SCHEMA))
    assert len(kwargs["logits_processor"]) == 1


def test_mlx_adds_the_processor_and_drops_the_draft() -> None:
    pytest.importorskip("llguidance")
    pytest.importorskip("mlx_lm.sample_utils")
    from hfl.engine.base import GenerationConfig
    from hfl.engine.mlx_engine import MLXEngine

    engine = MLXEngine()
    engine._draft = "draft"
    plain = engine._build_sampling(GenerationConfig())
    formatted = engine._build_sampling(GenerationConfig(response_format="json"))
    assert plain["draft_model"] == "draft" and "draft_model" not in formatted
    assert len(formatted["logits_processors"]) == len(plain["logits_processors"]) + 1


def test_both_transformers_paths_pass_it_to_generate(monkeypatch) -> None:
    pytest.importorskip("llguidance")
    torch = pytest.importorskip("torch")
    from hfl.engine.base import GenerationConfig
    from hfl.engine.transformers_engine import TransformersEngine

    seen: list[dict] = []

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, prompt, return_tensors=None):
            return Inputs(input_ids=torch.tensor([[1, 2]]))

        def decode(self, ids, skip_special_tokens=True):
            return "{}"

    class Model:
        device = "cpu"
        generation_config = type("G", (), {"eos_token_id": 0})()

        def generate(self, **kwargs):
            seen.append(kwargs)
            streamer = kwargs.get("streamer")
            if streamer is not None:
                streamer.end()
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr("hfl.engine.constrained.torch_processor", lambda *a: "guided")
    engine = TransformersEngine()
    engine._model, engine._tokenizer = Model(), Tokenizer()
    monkeypatch.setattr(
        "transformers.TextIteratorStreamer",
        type(
            "S",
            (),
            {
                "__init__": lambda s, *a, **k: None,
                "end": lambda s: None,
                "__iter__": lambda s: iter(()),
            },
        ),
    )
    cfg = GenerationConfig(response_format="json", max_tokens=4)
    engine.generate("p", cfg)
    list(engine.generate_stream("p", cfg))
    assert [list(k["logits_processor"]) for k in seen] == [["guided"], ["guided"]]

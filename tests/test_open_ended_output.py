# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A request with no output limit generates until the model stops.

Measured on 2026-11-01 (Project Hub on Qwen3.8 through the tray): a card
request without ``max_tokens`` generated 2048 tokens of 2048, four times —
HFL's default cut the model mid-reasoning, and the answer came back cut or
empty. OpenAI and Ollama (``num_predict`` -1, its default) have no such
limit. And Ollama's own -1 / -2 / 0 were clamped to *one* token.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.converters import num_predict_tokens, ollama_to_generation_config
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import (
    OPEN_ENDED,
    GenerationConfig,
    GenerationResult,
    hit_cap,
    output_budget,
    output_cap,
)


class TestTheRule:
    def test_open_ended_is_never_cut(self):
        assert output_cap(GenerationConfig(max_tokens=OPEN_ENDED)) is None
        assert not hit_cap(GenerationConfig(max_tokens=OPEN_ENDED), 10**6)
        assert hit_cap(GenerationConfig(max_tokens=64), 64)
        assert not hit_cap(GenerationConfig(max_tokens=64), 63)

    def test_a_finite_budget_is_what_the_context_leaves(self):
        assert output_budget(GenerationConfig(max_tokens=OPEN_ENDED), 4096, 96) == 4000
        assert output_budget(GenerationConfig(max_tokens=OPEN_ENDED), 4096, 5000) == 1
        assert output_budget(GenerationConfig(max_tokens=OPEN_ENDED), None, 96) == 32768
        assert output_budget(GenerationConfig(max_tokens=64), 4096, 96) == 64

    def test_ollamas_num_predict(self):
        assert [num_predict_tokens(v) for v in (None, -1, -2, 0)] == [OPEN_ENDED] * 4
        assert num_predict_tokens(100) == 100
        assert num_predict_tokens(500000) == 128000

    def test_the_options_helper_takes_minus_one(self):
        from hfl.api.helpers import options_to_config

        assert options_to_config({"num_predict": -1}).max_tokens == OPEN_ENDED  # was a 400

    def test_a_modelfiles_num_predict(self):
        from hfl.api.modelfile_defaults import apply_parameters

        manifest = types.SimpleNamespace(default_parameters={"num_predict": "-2"})
        cfg = GenerationConfig(max_tokens=2048)
        apply_parameters(manifest, cfg, set())
        assert cfg.max_tokens == OPEN_ENDED


class TestEachEngine:
    def test_llama_server_gets_minus_one(self):
        from hfl.engine.base import ChatMessage
        from tests.test_cov_llama_server_units import _engine

        engine = _engine()
        msgs = [ChatMessage(role="user", content="hi")]
        open_cfg = GenerationConfig(max_tokens=OPEN_ENDED)
        assert engine._chat_body(msgs, open_cfg, None)["max_tokens"] == -1
        assert engine._completion_body("p", open_cfg)["n_predict"] == -1
        assert engine._chat_body(msgs, GenerationConfig(max_tokens=64), None)["max_tokens"] == 64

    def test_mlx_ends_where_the_context_does(self):
        from tests.test_cov_engines_mlx import _engine

        engine = _engine(_context=4096)  # its tokenizer reads "abc" as 3 tokens
        assert engine._output_tokens(GenerationConfig(max_tokens=OPEN_ENDED), "abc") == 4093
        assert engine._output_tokens(GenerationConfig(max_tokens=OPEN_ENDED), [1] * 96) == 4000
        assert engine._output_tokens(GenerationConfig(max_tokens=64), "abc") == 64

    def test_transformers_ends_where_the_context_does(self):
        from hfl.engine.transformers_engine import TransformersEngine

        engine = TransformersEngine()
        engine._model = types.SimpleNamespace(
            config=types.SimpleNamespace(max_position_embeddings=8192)
        )
        inputs = {"input_ids": types.SimpleNamespace(shape=(1, 192))}
        assert engine._new_tokens(GenerationConfig(max_tokens=OPEN_ENDED), inputs) == 8000
        assert engine._new_tokens(GenerationConfig(max_tokens=64), inputs) == 64
        # A vision model's language layout is nested.
        engine._model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                text_config=types.SimpleNamespace(max_position_embeddings=1024)
            )
        )
        assert engine._new_tokens(GenerationConfig(max_tokens=OPEN_ENDED), inputs) == 832

    def test_vllm_gets_none(self, monkeypatch):
        from hfl.engine.vllm_engine import VLLMEngine

        seen: dict = {}
        vllm = types.ModuleType("vllm")
        vllm.SamplingParams = lambda **kw: seen.update(kw) or kw  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        engine = VLLMEngine()
        engine._build_sampling_params(GenerationConfig(max_tokens=OPEN_ENDED))
        assert seen["max_tokens"] is None  # until max_model_len
        engine._build_sampling_params(GenerationConfig(max_tokens=64))
        assert seen["max_tokens"] == 64


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _engine(sample_manifest, generated: int):
    seen: list[GenerationConfig] = []

    def chat(messages, config=None, tools=None):
        seen.append(config)
        return GenerationResult(
            text="x " * 5, tokens_generated=generated, tokens_prompt=9, stop_reason="stop"
        )

    def chat_stream(messages, config=None, tools=None):
        seen.append(config)
        yield from ["x "] * generated

    engine = MagicMock(is_loaded=True, supports_images=False)
    engine.chat = MagicMock(side_effect=chat)
    engine.chat_stream = MagicMock(side_effect=chat_stream)
    state = get_state()
    state.engine = engine
    state.current_model = sample_manifest
    return seen


def _last_json(text: str) -> dict:
    import json

    lines = [line.removeprefix("data: ") for line in text.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines if line.startswith("{")]
    return events[-1]


class TestTheRoutes:
    def test_openai_without_max_tokens(self, client, sample_manifest):
        seen = _engine(sample_manifest, generated=5)
        body = {"model": sample_manifest.name, "messages": [{"role": "user", "content": "q"}]}
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        assert seen[0].max_tokens == OPEN_ENDED

    def test_openai_max_completion_tokens(self, client, sample_manifest):
        seen = _engine(sample_manifest, generated=5)
        body = {
            "model": sample_manifest.name,
            "messages": [{"role": "user", "content": "q"}],
            "max_completion_tokens": 10,
        }
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        assert seen[0].max_tokens == 10

    @pytest.mark.parametrize(("limit", "finish"), [(None, "stop"), (5, "length")])
    def test_openai_stream_says_why_it_ended(self, client, sample_manifest, limit, finish):
        """``max_tokens and n >= max_tokens`` called every open-ended reply
        cut (-1 is truthy)."""
        _engine(sample_manifest, generated=5)
        body = {
            "model": sample_manifest.name,
            "messages": [{"role": "user", "content": "q"}],
            "stream": True,
            **({"max_completion_tokens": limit} if limit else {}),
        }
        reply = client.post("/v1/chat/completions", json=body)
        finishes = [
            choice.get("finish_reason")
            for line in reply.text.splitlines()
            if line.startswith("data: {")
            for choice in __import__("json").loads(line[6:]).get("choices", [])
        ]
        assert finish in finishes and ({"stop", "length"} - {finish}).isdisjoint(finishes)

    @pytest.mark.parametrize("num_predict", [-1, -2, 0])
    def test_ollama_num_predict(self, client, sample_manifest, num_predict):
        seen = _engine(sample_manifest, generated=5)
        body = {
            "model": sample_manifest.name,
            "messages": [{"role": "user", "content": "q"}],
            "options": {"num_predict": num_predict},
        }
        reply = client.post("/api/chat", json=body)  # streams, as Ollama's
        assert seen[0].max_tokens == OPEN_ENDED  # was 1
        assert _last_json(reply.text)["done_reason"] == "stop"

    def test_ollama_without_options(self):
        assert ollama_to_generation_config({}).max_tokens == OPEN_ENDED


def test_back_to_openai_no_limit_is_no_field():
    from hfl.api.converters import generation_config_to_ollama, generation_config_to_openai

    assert "max_tokens" not in generation_config_to_openai(GenerationConfig(max_tokens=OPEN_ENDED))
    assert generation_config_to_openai(GenerationConfig(max_tokens=64))["max_tokens"] == 64
    assert generation_config_to_ollama(GenerationConfig(max_tokens=OPEN_ENDED))["num_predict"] == -1

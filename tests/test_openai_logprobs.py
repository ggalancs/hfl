# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Per-token logprobs and ``n`` answers.

Checked for real (2026-09-26) with the official ``openai`` SDK and
Qwen2.5-0.5B-Instruct on the default GGUF backend, llama-server and MLX: at
temperature 0 each drawn token is its best alternative, the tokens rebuild
the answer, ``n=4`` at temperature 1.2 gives different answers, and
``/api/chat`` returns them. On one llama.cpp model the values equal
llama-cpp-python's own ``logits_all`` logprobs to four decimals; before,
``/api/generate`` with ``logprobs`` answered 500 (llama-cpp-python needs
``logits_all``, ~5 GB for an 8k context).
"""

from __future__ import annotations

import importlib.util
import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state
from hfl.engine.base import GenerationConfig, GenerationResult

ENTRY = {"token": "Paris", "logprob": -0.2, "bytes": [80], "top_logprobs": []}
USER = [{"role": "user", "content": "Capital of France?"}]


@pytest.fixture
def engine():
    state = get_state()
    fake = MagicMock()
    fake.is_loaded = True
    fake.chat.side_effect = lambda messages, cfg, tools=None: GenerationResult(
        text=f"Paris {cfg.seed}",
        tokens_prompt=5,
        tokens_generated=2,
        logprobs=[ENTRY] if cfg.logprobs is not None else None,
    )
    model = MagicMock()
    model.name = "m"
    state.engine, state.current_model, state.api_key = fake, model, None
    yield fake
    state.engine = state.current_model = None


def _post(**body):
    body = {"model": "m", "messages": USER, **body}
    return TestClient(app).post("/v1/chat/completions", json=body)


class TestOpenAIRoute:
    def test_logprobs_reach_the_engine_and_come_back(self, engine):
        choice = _post(logprobs=True, top_logprobs=3).json()["choices"][0]
        assert engine.chat.call_args.args[1].logprobs == 3
        assert choice["logprobs"] == {"content": [ENTRY]}
        _post(logprobs=True)
        assert engine.chat.call_args.args[1].logprobs == 0  # the drawn token only
        assert "logprobs" not in _post().json()["choices"][0]
        assert engine.chat.call_args.args[1].logprobs is None

    def test_n_answers_and_their_usage(self, engine):
        body = _post(n=3, seed=10).json()
        assert [c["index"] for c in body["choices"]] == [0, 1, 2]
        # A seed moves on per answer, so they can differ.
        assert [c["message"]["content"] for c in body["choices"]] == [
            "Paris 10",
            "Paris 11",
            "Paris 12",
        ]
        assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}

    @pytest.mark.parametrize(
        "body",
        [{"stream": True, "logprobs": True}, {"stream": True, "n": 2}, {"top_logprobs": 2}],
    )
    def test_what_is_refused_rather_than_ignored(self, engine, body):
        response = _post(**body)
        assert response.status_code in (400, 422)
        assert not engine.chat.called and not engine.chat_stream.called

    def test_a_backend_without_logprobs_says_so(self, engine):
        engine.chat.side_effect = NotImplementedError("the vLLM backend cannot return logprobs")
        response = _post(logprobs=True)
        assert response.status_code == 400 and "logprobs need" in response.text


def test_ollama_chat_returns_them_too(engine):
    body = {"model": "m", "messages": USER, "stream": False, "options": {"logprobs": 2}}
    response = TestClient(app).post("/api/chat", json=body).json()
    assert engine.chat.call_args.args[1].logprobs == 2
    assert response["logprobs"] == [ENTRY]


def test_llama_server_entries_drop_its_end_token_and_extra_alternatives():
    from hfl.engine.llama_server import _logprob_entries

    alt = {"id": 1, "token": "x", "logprob": -1.0, "bytes": [120]}
    raw = [
        {"id": 5, "token": "Paris", "logprob": -0.1, "bytes": [80], "top_logprobs": [alt, alt]},
        {"id": 2, "token": "", "logprob": -0.5, "bytes": [], "top_logprobs": [alt]},  # EOS
    ]
    entries = _logprob_entries(raw, 0)
    assert [e["token"] for e in entries] == ["Paris"]
    assert entries[0]["top_logprobs"] == [] and "id" not in entries[0]
    assert len(_logprob_entries(raw, 1)[0]["top_logprobs"]) == 1


@pytest.mark.skipif(importlib.util.find_spec("llama_cpp") is None, reason="needs llama-cpp-python")
class TestLlamaCppSampling:
    """The logits processor sees the row each token is drawn from."""

    ROWS = [np.array([2.0, 1.0, 0.0, -5.0]), np.array([0.0, 3.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 9.0])]  # fmt: skip

    def _engine(self, monkeypatch):
        from llama_cpp import llama_cpp as lcpp

        from hfl.engine.llama_cpp import LlamaCppEngine

        rows = self.ROWS

        class Model:
            _model = MagicMock()

            def n_vocab(self):
                return 4

            def generate(self, tokens, logits_processor, **kwargs):
                for row in rows:
                    logits_processor[0](tokens, row.astype(np.float32))
                    yield int(np.argmax(row))

            def detokenize(self, tokens, prev_tokens=None, special=False):
                return "ABCD"[tokens[0]].encode()

        monkeypatch.setattr(lcpp, "llama_vocab_is_eog", lambda vocab, token: token == 3)
        engine = LlamaCppEngine()
        engine._model = Model()
        return engine

    def test_logprobs_top_and_the_end_token(self, monkeypatch):
        engine = self._engine(monkeypatch)
        cfg = GenerationConfig(logprobs=2, max_tokens=10)
        text, entries, n, finish = engine._sample_with_logprobs([0], cfg, 1.0, special=False)
        assert (text, n, finish) == ("AB", 2, "stop")  # D is the end token, not shown
        row = self.ROWS[0]
        expected = row[0] - math.log(np.exp(row).sum())
        assert entries[0]["logprob"] == pytest.approx(expected)
        assert [a["token"] for a in entries[0]["top_logprobs"]] == ["A", "B"]

    def test_stop_strings_and_max_tokens(self, monkeypatch):
        engine = self._engine(monkeypatch)
        text, _, _, finish = engine._sample_with_logprobs(
            [0], GenerationConfig(logprobs=0, stop=["B"]), 1.0, special=False
        )
        assert (text, finish) == ("A", "stop")
        text, entries, n, finish = engine._sample_with_logprobs(
            [0], GenerationConfig(logprobs=0, max_tokens=1), 1.0, special=False
        )
        assert (text, n, finish, entries[0]["top_logprobs"]) == ("A", 1, "length", [])

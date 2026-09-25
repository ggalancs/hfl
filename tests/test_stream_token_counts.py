# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Streams report the tokens they really cost.

Streamed replies used to report chunks as tokens (a chunk can carry
several: an unfinished UTF-8 character, a held-back stop sequence) and
0 prompt tokens, and ignored OpenAI's ``stream_options.include_usage``.
Engines that can count now hand back a CountedStream; an engine that
cannot keeps the old fallback, and no number is invented for OpenAI's
``usage``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.engine.base import CountedStream


class FakeLlama:
    """n_tokens behaves as measured on llama-cpp-python: prompt evaluated
    when the first chunk arrives; prompt + completion at the end, less the
    last token when max_tokens cut the reply."""

    def __init__(self, prompt: int, pieces: list[str], finish: str):
        self.n_tokens = 0
        self.prompt, self.pieces, self.finish = prompt, pieces, finish

    def create_chat_completion(self, **kwargs):
        def gen():
            self.n_tokens = self.prompt
            yield {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}
            for piece in self.pieces:
                yield {"choices": [{"delta": {"content": piece}, "finish_reason": None}]}
                self.n_tokens += 1
            if self.finish == "length":
                self.n_tokens -= 1
            yield {"choices": [{"delta": {}, "finish_reason": self.finish}]}

        return gen()


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_llama_cpp_counts_from_the_context(finish):
    from hfl.engine.base import ChatMessage
    from hfl.engine.llama_cpp import LlamaCppEngine

    engine = LlamaCppEngine()
    engine._model = FakeLlama(prompt=20, pieces=["1", ",", " 2", "🚲"], finish=finish)
    engine._architecture = "phi3"
    stream = engine.chat_stream([ChatMessage(role="user", content="x")])
    assert isinstance(stream, CountedStream)
    assert "".join(stream) == "1, 2🚲"
    assert (stream.prompt_tokens, stream.completion_tokens) == (20, 4)


def test_an_unfinished_stream_has_no_counts():
    from hfl.engine.base import ChatMessage
    from hfl.engine.llama_cpp import LlamaCppEngine

    engine = LlamaCppEngine()
    engine._model = FakeLlama(prompt=20, pieces=["a", "b"], finish="stop")
    engine._architecture = "phi3"
    stream = engine.chat_stream([ChatMessage(role="user", content="x")])
    next(stream)
    stream.close()  # the client went away
    assert (stream.prompt_tokens, stream.completion_tokens) == (None, None)


def _counted(pieces, prompt=11, completion=5):
    stream = CountedStream()

    def gen():
        yield from pieces
        stream.prompt_tokens, stream.completion_tokens = prompt, completion

    return stream.feed(gen())


@pytest.fixture
def client(temp_config, sample_manifest):
    from hfl.api.server import app
    from hfl.api.state import get_state, reset_state

    reset_state()
    engine = MagicMock(is_loaded=True)
    engine.chat_stream = MagicMock(side_effect=lambda *a, **k: _counted(["Hel", "lo"]))
    engine.generate_stream = MagicMock(side_effect=lambda *a, **k: _counted(["a", "b"]))
    state = get_state()
    state.engine = engine
    state.current_model = sample_manifest
    yield TestClient(app), engine, sample_manifest.name
    reset_state()


def _sse(text):
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_openai_include_usage(client):
    http, _, name = client
    body = {"model": name, "stream": True, "messages": [{"role": "user", "content": "x"}]}
    events = _sse(
        http.post(
            "/v1/chat/completions", json={**body, "stream_options": {"include_usage": True}}
        ).text
    )
    assert events[-1]["choices"] == []
    assert events[-1]["usage"] == {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    plain = _sse(http.post("/v1/chat/completions", json=body).text)
    assert not any("usage" in e for e in plain)


def test_openai_completions_include_usage(client):
    http, _, name = client
    events = _sse(
        http.post(
            "/v1/completions",
            json={
                "model": name,
                "prompt": "x",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ).text
    )
    assert events[-1]["usage"]["completion_tokens"] == 5


def test_no_usage_is_made_up_for_an_engine_that_cannot_count(client):
    http, engine, name = client
    engine.chat_stream = MagicMock(side_effect=lambda *a, **k: iter(["Hel", "lo"]))
    events = _sse(
        http.post(
            "/v1/chat/completions",
            json={
                "model": name,
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
                "stream_options": {"include_usage": True},
            },
        ).text
    )
    assert not any("usage" in e for e in events)


@pytest.mark.parametrize("route", ["/api/chat", "/api/generate"])
def test_ollama_counts(client, route):
    http, _, name = client
    body = {"model": name, "stream": True}
    body.update(
        {"messages": [{"role": "user", "content": "x"}]}
        if route == "/api/chat"
        else {"prompt": "x"}
    )
    last = [json.loads(line) for line in http.post(route, json=body).text.splitlines() if line][-1]
    assert (last["prompt_eval_count"], last["eval_count"]) == (11, 5)


def test_anthropic_counts(client):
    http, _, name = client
    text = http.post(
        "/v1/messages",
        json={
            "model": name,
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "x"}],
        },
    ).text
    delta = next(e for e in _sse(text) if e.get("type") == "message_delta")
    assert delta["usage"] == {"output_tokens": 5, "input_tokens": 11}


def test_responses_counts(client):
    http, _, name = client
    events = _sse(
        http.post("/v1/responses", json={"model": name, "input": "x", "stream": True}).text
    )
    usage = events[-1]["response"]["usage"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (11, 5)

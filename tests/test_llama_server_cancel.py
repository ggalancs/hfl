# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A dispatched llama-server call can be cancelled (a request past its
budget): it is sent streamed and reassembled, so the signal is checked between
events and leaving the stream makes llama-server cancel the task. Measured on
a real llama-server: with this, "stop: cancel task" at the 504 and the slot
freed at 1292 tokens; before, the task ran on to 4021 tokens, 11 s past it."""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from hfl.engine import cancel
from hfl.engine.base import ChatMessage, GenerationConfig
from hfl.engine.llama_server import LlamaServerEngine


def _sse(events: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"


def _engine(handler) -> LlamaServerEngine:
    engine = LlamaServerEngine()
    engine._client = httpx.Client(base_url="http://llama", transport=httpx.MockTransport(handler))
    return engine


CHAT_EVENTS = [
    {
        "choices": [
            {
                "delta": {"content": "Hel"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Hel",
                            "logprob": -0.1,
                            "top_logprobs": [{"token": "Hel", "logprob": -0.1}],
                        }
                    ]
                },
            }
        ]
    },
    {
        "choices": [
            {
                "delta": {
                    "content": "lo",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"ci'},
                        }
                    ],
                }
            }
        ]
    },
    {
        "choices": [
            {
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Paris"}'}}]},
                "finish_reason": "tool_calls",
            }
        ]
    },
    {
        "choices": [],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        "timings": {"prompt_ms": 10.0, "predicted_ms": 30.0},
    },
]


def test_the_streamed_chat_is_reassembled_as_its_blocking_answer() -> None:
    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse(CHAT_EVENTS))

    engine = _engine(handler)
    with cancel.scope(threading.Event()):
        result = engine.chat([ChatMessage(role="user", content="hi")], GenerationConfig(logprobs=1))
    assert seen["body"]["stream"] is True
    assert result.text == "Hello"
    assert result.tool_calls[0]["function"]["name"] == "get_weather"
    assert result.tool_calls[0]["function"]["arguments"] == {"city": "Paris"}
    assert (result.tokens_prompt, result.tokens_generated) == (7, 3)
    assert result.eval_duration == 30_000_000 and result.logprobs[0]["token"] == "Hel"


@pytest.mark.parametrize(
    ("final", "reason"),
    [
        ({"stop_type": "limit"}, "length"),  # current llama-server (b10964, Homebrew)
        ({"stop_type": "eos"}, "stop"),
        ({"stopped_limit": True}, "length"),  # older builds
    ],
)
def test_a_completion_cut_at_num_predict_says_length(final: dict, reason: str) -> None:
    """HFL read only ``stopped_limit``, which current builds no longer send:
    every completion said "stop", even one cut at num_predict."""
    events = [{"content": "1", "stop": False}, {"content": "", "stop": True, **final}]
    engine = _engine(lambda request: httpx.Response(200, content=_sse(events)))
    with cancel.scope(threading.Event()):
        assert engine.generate("0", GenerationConfig()).stop_reason == reason


def test_the_streamed_completion_is_reassembled() -> None:
    events = [
        {"content": " Paris", "stop": False, "completion_probabilities": [{"token": " Paris"}]},
        {"content": ".", "stop": False, "completion_probabilities": [{"token": "."}]},
        {
            "content": "",
            "stop": True,
            "tokens_evaluated": 5,
            "tokens_predicted": 2,
            "stopped_limit": True,
            "timings": {"predicted_ms": 20.0},
        },
    ]
    engine = _engine(lambda request: httpx.Response(200, content=_sse(events)))
    with cancel.scope(threading.Event()):
        result = engine.generate("The capital of France is", GenerationConfig())
    assert result.text == " Paris." and result.stop_reason == "length"
    assert (result.tokens_prompt, result.tokens_generated) == (5, 2)


def test_a_cancelled_request_stops_reading_and_raises() -> None:
    engine = _engine(lambda request: httpx.Response(200, content=_sse(CHAT_EVENTS)))
    signal = threading.Event()
    signal.set()
    with cancel.scope(signal), pytest.raises(cancel.GenerationCancelled):
        engine.chat([ChatMessage(role="user", content="hi")], GenerationConfig())


def test_outside_a_dispatched_call_it_stays_one_blocking_request() -> None:
    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = _engine(handler).chat([ChatMessage(role="user", content="hi")], GenerationConfig())
    assert result.text == "ok" and "stream" not in seen["body"]

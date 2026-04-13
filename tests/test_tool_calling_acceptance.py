# SPDX-License-Identifier: HRUL-1.0
"""End-to-end acceptance suite for ``hfl-tool-calling-spec.md`` §6.

These are the tests that llm-kb (the compiler agent) must see pass for
HFL to drive a tool-calling loop without client-side workarounds.

The spec's pass criteria is:

- **Blocking for llm-kb**: T1, T2, T3, T4
- **Nice-to-have**: T5 (streaming tool_calls), T6 (rate-limit headers),
  T7 (structured error envelope)

The suite drives the FastAPI TestClient with a ``FakeEngine`` that emits
canned qwen/llama3/mistral raw outputs so no real model weights are
required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.middleware import reset_rate_limiter
from hfl.api.server import app
from hfl.api.state import get_state
from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult


pytestmark = pytest.mark.acceptance


# --- Fake engine ---------------------------------------------------------------


@dataclass
class FakeEngine:
    """Programmable engine used only in acceptance tests.

    ``script`` is a list of raw text outputs the engine will return on
    successive ``chat()`` calls (one per turn). ``tools_seen`` records what
    was forwarded so tests can assert propagation.
    """

    script: list[str]
    is_loaded: bool = True
    tools_seen: list[Any] | None = None
    messages_seen: list[list[ChatMessage]] | None = None

    def __post_init__(self) -> None:
        self.tools_seen = []
        self.messages_seen = []
        self._turn = 0

    def _next(self) -> str:
        if self._turn >= len(self.script):
            return ""
        text = self.script[self._turn]
        self._turn += 1
        return text

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        self.tools_seen.append(tools)
        self.messages_seen.append(list(messages))
        return GenerationResult(text=self._next(), tokens_generated=1)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        self.tools_seen.append(tools)
        self.messages_seen.append(list(messages))
        text = self._next()
        # Emit as a single chunk — the route layer must still accumulate
        # and parse the final text.
        if text:
            yield text


@pytest.fixture
def fake_engine_factory():
    """Yield a factory that installs a FakeEngine on the server state.

    The factory takes an optional ``model_name`` so multi-model tests can
    bypass the registry lookup path (``_ensure_model_loaded`` short-
    circuits when the current model already matches by name).
    """

    installed: list[FakeEngine] = []

    def _install(
        script: list[str], model_name: str = "qwen3-32b-q4_k_m"
    ) -> FakeEngine:
        eng = FakeEngine(script=script)
        mock_model = MagicMock()
        mock_model.name = model_name
        get_state().engine = eng
        get_state().current_model = mock_model
        installed.append(eng)
        return eng

    yield _install

    get_state().engine = None
    get_state().current_model = None
    get_state().api_key = None
    reset_rate_limiter()


@pytest.fixture
def client():
    get_state().api_key = None
    reset_rate_limiter()
    yield TestClient(app)
    reset_rate_limiter()


WRITE_WIKI_TOOL = {
    "type": "function",
    "function": {
        "name": "write_wiki",
        "description": "Create or overwrite a wiki article",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}


GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


# --- T1: plain chat returns done:true -----------------------------------------


class TestT1PlainChat:
    def test_plain_chat_returns_done_true(self, client, fake_engine_factory):
        fake_engine_factory(["ok"])
        resp = client.post(
            "/api/chat",
            json={
                "model": "qwen3-32b-q4_k_m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["done"] is True
        assert body["message"]["role"] == "assistant"
        assert body["message"]["content"] == "ok"


# --- T2: tool call exposed in message.tool_calls ------------------------------


QWEN_TOOL_OUTPUT = (
    '<tool_call>{"name": "write_wiki", "arguments": '
    '{"path": "topics/hello.md", "content": "Hello world"}}</tool_call>'
)

LLAMA3_TOOL_OUTPUT = (
    '<|python_tag|>{"name": "write_wiki", '
    '"parameters": {"path": "topics/hello.md", "content": "Hello world"}}<|eom_id|>'
)

MISTRAL_TOOL_OUTPUT = (
    '[TOOL_CALLS][{"name": "write_wiki", "arguments": '
    '{"path": "topics/hello.md", "content": "Hello world"}}]'
)

NON_STANDARD_ENVELOPE_OUTPUT = (
    "<think>reasoning</think>\n\n"
    '{"tool_call": {"name": "write_wiki", "args": '
    '{"path": "topics/hello.md", "content": "Hello world"}}}'
)


@pytest.mark.parametrize(
    "model_name,raw_output",
    [
        ("qwen3-32b-q4_k_m", QWEN_TOOL_OUTPUT),
        ("llama-3.2-70b", LLAMA3_TOOL_OUTPUT),
        ("mistral-7b-instruct", MISTRAL_TOOL_OUTPUT),
        ("qwen3-32b-q4_k_m", NON_STANDARD_ENVELOPE_OUTPUT),
    ],
)
class TestT2ToolCallExposed:
    def test_tool_call_appears_in_message_tool_calls(
        self, client, fake_engine_factory, model_name, raw_output
    ):
        eng = fake_engine_factory([raw_output], model_name=model_name)
        resp = client.post(
            "/api/chat",
            json={
                "model": model_name,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You MUST call write_wiki, never respond with text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Save Hello at topics/hello.md",
                    },
                ],
                "tools": [WRITE_WIKI_TOOL],
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        # Rule C1: tool_calls is a non-empty list
        msg = body["message"]
        tcs = msg.get("tool_calls")
        assert isinstance(tcs, list) and len(tcs) >= 1, (
            f"tool_calls is empty; content was: {msg.get('content')!r}"
        )

        # Rule C2: canonical shape
        fn = tcs[0]["function"]
        assert fn["name"] == "write_wiki"

        # Rule C3: arguments is a dict
        assert isinstance(fn["arguments"], dict)
        assert fn["arguments"]["path"] == "topics/hello.md"
        assert fn["arguments"]["content"] == "Hello world"

        # Rule C4: content is "" when tool_calls present
        assert msg["content"] == ""

        # Rule C7: tool_calls is always a list
        assert isinstance(tcs, list)

        # tools forwarded to the engine
        assert eng.tools_seen[-1] is not None
        assert len(eng.tools_seen[-1]) == 1


# --- T3: multi-turn with role=tool --------------------------------------------


class TestT3MultiTurn:
    def test_role_tool_message_reaches_engine(self, client, fake_engine_factory):
        """Spec §6 T3: the ``role=tool`` message must propagate to the
        model so the next turn can consume the result."""
        eng = fake_engine_factory(["The weather in Madrid is 22C sunny."])

        resp = client.post(
            "/api/chat",
            json={
                "model": "qwen3-32b-q4_k_m",
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in Madrid?",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Madrid"},
                                }
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "get_weather",
                        "content": "22C sunny",
                    },
                ],
                "tools": [GET_WEATHER_TOOL],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        content = body["message"]["content"]
        assert "22" in content or "sunny" in content.lower()

        # Engine must have been called with the role=tool message preserved.
        sent = eng.messages_seen[-1]
        roles = [m.role for m in sent]
        assert "tool" in roles
        tool_msg = next(m for m in sent if m.role == "tool")
        assert tool_msg.name == "get_weather"
        assert tool_msg.content == "22C sunny"


# --- T4: empty tools array must not break plain chat --------------------------


class TestT4EmptyTools:
    def test_empty_tools_array(self, client, fake_engine_factory):
        fake_engine_factory(["ok"])
        resp = client.post(
            "/api/chat",
            json={
                "model": "qwen3-32b-q4_k_m",
                "stream": False,
                "tools": [],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["done"] is True
        assert body["message"]["content"] == "ok"
        # Rule C7: tool_calls is always a list
        assert body["message"]["tool_calls"] == []


# --- T5: streaming tool_calls emission ----------------------------------------


class TestT5StreamingToolCalls:
    def test_streaming_final_chunk_has_tool_calls(
        self, client, fake_engine_factory
    ):
        fake_engine_factory([QWEN_TOOL_OUTPUT])
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": "qwen3-32b-q4_k_m",
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": "Save Hello at topics/hello.md",
                    }
                ],
                "tools": [WRITE_WIKI_TOOL],
            },
        ) as resp:
            assert resp.status_code == 200
            lines = [ln for ln in resp.iter_lines() if ln]

        final = json.loads(lines[-1])
        assert final["done"] is True
        msg = final["message"]
        assert msg["tool_calls"], f"final chunk missing tool_calls: {final}"
        assert msg["tool_calls"][0]["function"]["name"] == "write_wiki"
        assert msg["tool_calls"][0]["function"]["arguments"] == {
            "path": "topics/hello.md",
            "content": "Hello world",
        }
        assert msg["content"] == ""


# --- T6: rate-limit headers present on every response ------------------------


class TestT6RateLimitHeaders:
    def test_all_four_headers_present(self, client):
        resp = client.get("/api/version")
        headers = {k.lower(): v for k, v in resp.headers.items()}
        for h in (
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
            "x-ratelimit-window",
        ):
            assert h in headers, f"missing rate-limit header: {h}"


# --- T7: structured error envelope with code/category/retryable --------------


class TestT7ErrorEnvelope:
    def test_unauthorized_envelope(self, client):
        get_state().api_key = "expected-key"
        resp = client.post(
            "/api/chat",
            headers={"Authorization": "Bearer wrong-token"},
            json={
                "model": "qwen3-32b-q4_k_m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 401
        body = resp.json()
        err = body.get("error")
        assert isinstance(err, dict), f"error must be structured dict: {err!r}"
        for k in ("error", "code", "category", "retryable"):
            assert k in err, f"error missing {k}"
        assert err["code"] == "UNAUTHORIZED"
        assert err["category"] == "auth"
        assert err["retryable"] is False
        get_state().api_key = None

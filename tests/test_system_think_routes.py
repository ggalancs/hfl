# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Integration tests for the ``system`` and ``think`` request fields.

Phase 5, P1-1. Both fields flow through the Ollama-native routes:

- ``system`` — overrides the model's default system prompt.
- ``think`` — exposes the reasoning channel; post-processed into a
  separate ``thinking`` field on the response envelope.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _mock_engine(sample_manifest, *, chat_text="ok", generate_text="ok"):
    """Install a recording mock engine on state."""
    state = get_state()
    captured = {"chat_messages": None, "generate_prompt": None, "configs": []}

    def _chat(messages, config=None, tools=None):
        captured["chat_messages"] = list(messages)
        captured["configs"].append(config)
        return MagicMock(text=chat_text, tokens_generated=1, tokens_prompt=1, stop_reason="stop")

    def _generate(prompt, config=None):
        captured["generate_prompt"] = prompt
        captured["configs"].append(config)
        return MagicMock(
            text=generate_text, tokens_generated=1, tokens_prompt=1, stop_reason="stop"
        )

    engine = MagicMock(is_loaded=True)
    engine.chat = MagicMock(side_effect=_chat)
    engine.generate = MagicMock(side_effect=_generate)
    state.engine = engine
    state.current_model = sample_manifest
    return engine, captured


# ----------------------------------------------------------------------
# system — /api/generate
# ----------------------------------------------------------------------


class TestSystemOnGenerate:
    def test_system_prepended_to_prompt(self, client, sample_manifest):
        _, captured = _mock_engine(sample_manifest)
        response = client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "what is 2+2?",
                "system": "You are a math tutor.",
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert captured["generate_prompt"] == "You are a math tutor.\n\nwhat is 2+2?"

    def test_no_system_leaves_prompt_as_is(self, client, sample_manifest):
        _, captured = _mock_engine(sample_manifest)
        client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "raw prompt",
                "stream": False,
            },
        )
        assert captured["generate_prompt"] == "raw prompt"


# ----------------------------------------------------------------------
# system — /api/chat
# ----------------------------------------------------------------------


class TestSystemOnChat:
    def test_system_inserted_as_first_message(self, client, sample_manifest):
        _, captured = _mock_engine(sample_manifest)
        client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "system": "You are Gandalf.",
                "stream": False,
            },
        )
        msgs = captured["chat_messages"]
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[0].content == "You are Gandalf."
        assert msgs[1].role == "user"

    def test_system_does_not_replace_existing_system(self, client, sample_manifest):
        """If the caller already has a system message, the override
        is prepended — Ollama allows multiple system roles."""
        _, captured = _mock_engine(sample_manifest)
        client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [
                    {"role": "system", "content": "original"},
                    {"role": "user", "content": "q"},
                ],
                "system": "prepended",
                "stream": False,
            },
        )
        msgs = captured["chat_messages"]
        # Override first, original system kept, then user
        assert [m.role for m in msgs] == ["system", "system", "user"]
        assert msgs[0].content == "prepended"
        assert msgs[1].content == "original"


# ----------------------------------------------------------------------
# think — /api/chat
# ----------------------------------------------------------------------


class TestThinkOnChat:
    def test_think_true_sets_expose_reasoning(self, client, sample_manifest):
        _, captured = _mock_engine(sample_manifest)
        client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "think": True,
                "stream": False,
            },
        )
        # Engine config saw expose_reasoning=True
        assert captured["configs"][0].expose_reasoning is True

    def test_think_false_leaves_default(self, client, sample_manifest):
        _, captured = _mock_engine(sample_manifest)
        client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "hi"}],
                "think": False,
                "stream": False,
            },
        )
        assert captured["configs"][0].expose_reasoning is False

    def test_thinking_extracted_into_separate_field(self, client, sample_manifest):
        """With think=True AND a response that carries thinking
        markers, the route splits them into ``thinking`` and
        ``content``."""
        _mock_engine(
            sample_manifest,
            chat_text="<think>pondering...</think>The answer is 42.",
        )
        response = client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "q"}],
                "think": True,
                "stream": False,
            },
        )
        body = response.json()
        assert body["message"]["content"] == "The answer is 42."
        assert body["message"]["thinking"] == "pondering..."

    def test_no_think_no_thinking_field(self, client, sample_manifest):
        """Without think=True the response omits the thinking key
        entirely (text-only envelope matches pre-Phase-5 shape)."""
        _mock_engine(sample_manifest, chat_text="<think>hidden</think>final")
        response = client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "q"}],
                "stream": False,
            },
        )
        body = response.json()
        assert "thinking" not in body["message"]

    def test_think_true_without_markers_content_unchanged(self, client, sample_manifest):
        """Model emits no reasoning markers → content unchanged,
        no ``thinking`` field added."""
        _mock_engine(sample_manifest, chat_text="plain answer")
        response = client.post(
            "/api/chat",
            json={
                "model": sample_manifest.name,
                "messages": [{"role": "user", "content": "q"}],
                "think": True,
                "stream": False,
            },
        )
        body = response.json()
        assert body["message"]["content"] == "plain answer"
        assert "thinking" not in body["message"]


# ----------------------------------------------------------------------
# think on /api/generate (just config flow, no envelope split)
# ----------------------------------------------------------------------


class TestThinkOnGenerate:
    def test_think_flag_on_generate(self, client, sample_manifest):
        """think=True on /api/generate sets expose_reasoning. The
        route doesn't split out reasoning automatically — /api/generate
        returns a raw ``response`` string, Ollama doesn't add a
        thinking field there."""
        _, captured = _mock_engine(sample_manifest)
        client.post(
            "/api/generate",
            json={
                "model": sample_manifest.name,
                "prompt": "hi",
                "think": True,
                "stream": False,
            },
        )
        assert captured["configs"][0].expose_reasoning is True

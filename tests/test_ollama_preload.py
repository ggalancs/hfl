# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Ollama's preload: a request with nothing to say loads the model.

``/api/generate`` without a prompt and ``/api/chat`` with no messages
load the model and answer ``done_reason: "load"``; with ``keep_alive: 0``
they unload it instead (``"unload"``) without loading it first. Clients
use this to warm a model up (Open WebUI, scripts, ``hfl launch``). HFL
used to pass the empty prompt to llama.cpp, which asserted: a 500.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def wired(monkeypatch, sample_manifest):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    state.engine = engine
    state.current_model = sample_manifest
    load = AsyncMock()
    unload = AsyncMock()
    monkeypatch.setattr("hfl.api.routes_native._ensure_model_loaded", load)
    monkeypatch.setattr("hfl.api.routes_native.unload_after_response", unload)
    return engine, load, unload, sample_manifest.name


@pytest.mark.parametrize("body", [{"prompt": ""}, {}])
@pytest.mark.parametrize("stream", [False, True])
def test_generate_without_a_prompt_loads(client, wired, body, stream):
    engine, load, unload, name = wired
    response = client.post("/api/generate", json={"model": name, "stream": stream, **body})
    assert response.status_code == 200
    reply = response.json()
    assert reply["done"] is True and reply["done_reason"] == "load"
    assert reply["response"] == "" and reply["model"] == name
    load.assert_awaited_once()
    unload.assert_not_awaited()
    engine.generate.assert_not_called()


@pytest.mark.parametrize("body", [{"messages": []}, {}])
def test_chat_without_messages_loads(client, wired, body):
    engine, load, unload, name = wired
    response = client.post("/api/chat", json={"model": name, "stream": False, **body})
    assert response.status_code == 200
    reply = response.json()
    assert reply["done_reason"] == "load"
    assert reply["message"] == {"role": "assistant", "content": ""}
    load.assert_awaited_once()
    engine.chat.assert_not_called()


@pytest.mark.parametrize("route", ["/api/generate", "/api/chat"])
def test_keep_alive_zero_unloads_without_loading(client, wired, route):
    engine, load, unload, name = wired
    response = client.post(route, json={"model": name, "keep_alive": 0, "stream": False})
    assert response.status_code == 200
    assert response.json()["done_reason"] == "unload"
    unload.assert_awaited_once_with(name)
    load.assert_not_awaited()


def test_a_real_prompt_still_generates(client, wired):
    engine, load, unload, name = wired
    engine.generate.return_value = MagicMock(
        text="hi", tokens_generated=1, tokens_prompt=1, stop_reason="stop"
    )
    reply = client.post("/api/generate", json={"model": name, "prompt": "x", "stream": False})
    assert reply.json()["response"] == "hi"
    engine.generate.assert_called_once()

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Images reach a model that can see them, and are refused otherwise.

Measured on 2026-10-31: Transformers dropped an image and the model
answered without it; llama.cpp without a projector failed with a bare 500;
``/api/generate`` ignored its ``images`` and Qwen3.8 called a red image
"green" (the same image through ``/api/chat``: "Red"); ``/v1/responses``
dropped ``input_image`` parts. Every route refuses before a stream starts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from tests.test_vision_routes import _png_base64, _png_data_uri


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


def _engine(sample_manifest, sees: bool):
    seen: list = []

    def chat(messages, config=None, tools=None):
        seen.append(list(messages))
        return MagicMock(text="Red", tokens_generated=1, tokens_prompt=9, stop_reason="stop")

    def chat_stream(messages, config=None, tools=None):
        seen.append(list(messages))
        yield "Red"

    engine = MagicMock(is_loaded=True, supports_images=sees)
    engine.chat = MagicMock(side_effect=chat)
    engine.chat_stream = MagicMock(side_effect=chat_stream)
    state = get_state()
    state.engine = engine
    state.current_model = sample_manifest
    return engine, seen


def _requests(model: str, stream: bool) -> dict[str, dict]:
    question = "What colour is this image?"
    return {
        "/v1/chat/completions": {
            "model": model,
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": _png_data_uri()}},
                    ],
                }
            ],
        },
        "/api/chat": {
            "model": model,
            "stream": stream,
            "messages": [{"role": "user", "content": question, "images": [_png_base64()]}],
        },
        "/api/generate": {
            "model": model,
            "stream": stream,
            "prompt": question,
            "images": [_png_base64()],
        },
        "/v1/messages": {
            "model": model,
            "stream": stream,
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _png_base64(),
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        },
        "/v1/responses": {
            "model": model,
            "stream": stream,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {"type": "input_image", "image_url": _png_data_uri()},
                    ],
                }
            ],
        },
    }


ROUTES = ["/v1/chat/completions", "/api/chat", "/api/generate", "/v1/messages", "/v1/responses"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ROUTES)
def test_a_model_that_cannot_see_refuses_them(client, sample_manifest, path, stream):
    engine, seen = _engine(sample_manifest, sees=False)
    response = client.post(path, json=_requests(sample_manifest.name, stream)[path])
    assert response.status_code == 400, response.text
    assert "cannot see images" in response.text
    assert seen == []  # the engine was never asked


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ROUTES)
def test_a_model_that_can_see_gets_them(client, sample_manifest, path, stream):
    engine, seen = _engine(sample_manifest, sees=True)
    response = client.post(path, json=_requests(sample_manifest.name, stream)[path])
    assert response.status_code == 200, response.text
    assert "Red" in response.text
    (messages,) = seen
    images = [image for m in messages for image in (m.images or [])]
    assert len(images) == 1 and images[0].startswith(b"\x89PNG")


def test_generate_with_images_goes_through_the_chat_template(client, sample_manifest):
    engine, seen = _engine(sample_manifest, sees=True)
    body = _requests(sample_manifest.name, False)["/api/generate"]
    response = client.post("/api/generate", json={**body, "system": "Be brief."})
    assert response.status_code == 200 and response.json()["response"] == "Red"
    engine.generate.assert_not_called()
    (messages,) = seen
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].content == "What colour is this image?"


def test_a_raw_prompt_cannot_carry_images(client, sample_manifest):
    _engine(sample_manifest, sees=True)
    body = _requests(sample_manifest.name, False)["/api/generate"]
    response = client.post("/api/generate", json={**body, "raw": True})
    assert response.status_code == 400 and "raw prompt" in response.text


@pytest.mark.parametrize("kind", ["input_file", "input_audio"])
def test_responses_refuses_parts_it_cannot_pass_on(client, sample_manifest, kind):
    _engine(sample_manifest, sees=True)
    body = {
        "model": sample_manifest.name,
        "input": [{"role": "user", "content": [{"type": kind, "file_id": "f"}]}],
    }
    response = client.post("/v1/responses", json=body)
    assert response.status_code == 400 and kind in response.text


def test_the_socket_takes_text_only():
    from hfl.api.routes_ws import _validate_chat_frame

    frame = {"model": "m", "messages": [{"role": "user", "content": [{"type": "image"}]}]}
    with pytest.raises(ValueError, match="takes no images"):
        _validate_chat_frame(frame)
    assert _validate_chat_frame({"model": "m", "messages": [{"role": "user", "content": "q"}]})


def test_generate_with_images_keeps_the_reasoning_out_of_the_response(client, sample_manifest):
    """Through llama-server Qwen3.8's reply came back as
    "<think>\\n\\n</think>\\n\\n Red": a chat's reasoning is in its text."""
    engine, _ = _engine(sample_manifest, sees=True)
    engine.chat.side_effect = lambda *a, **k: MagicMock(
        text="<think>\nIt is red.\n</think>\n\nRed",
        tokens_generated=9,
        tokens_prompt=9,
        stop_reason="stop",
    )
    body = _requests(sample_manifest.name, False)["/api/generate"]
    reply = client.post("/api/generate", json={**body, "think": False}).json()
    assert reply["response"].strip() == "Red" and "thinking" not in reply
    reply = client.post("/api/generate", json={**body, "think": True}).json()
    assert reply["response"].strip() == "Red" and reply["thinking"].strip() == "It is red."

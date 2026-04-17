# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""End-to-end tests for vision / multimodal routes (Phase 4, P0-6).

Exercises the full path:
    HTTP request with images
      → route handler
        → schema validation
          → vision.{decode_ollama_images | split_openai_content}
            → image_validator.validate_image
              → ChatMessage.images (bytes)
                → engine.chat(...)

The engine is mocked so we only pin the HFL-side wire contract.
"""

from __future__ import annotations

import base64
import struct
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state, reset_state

# ----------------------------------------------------------------------
# Image fixtures — minimal real images the validator will accept.
# ----------------------------------------------------------------------


def _make_png(width: int = 2, height: int = 2) -> bytes:
    """Small PNG header + empty chunks; passes magic-byte + IHDR parsing."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = (
        struct.pack(">I", 13)  # length
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"  # fake CRC
    )
    idat = struct.pack(">I", 0) + b"IDAT" + b"\x00\x00\x00\x00"
    iend = struct.pack(">I", 0) + b"IEND" + b"\x00\x00\x00\x00"
    return sig + ihdr + idat + iend


def _png_base64() -> str:
    return base64.b64encode(_make_png()).decode()


def _png_data_uri() -> str:
    return "data:image/png;base64," + _png_base64()


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app)
    reset_state()


@pytest.fixture
def vision_engine(sample_manifest):
    """Install a mocked engine that records every ChatMessage it sees."""
    state = get_state()
    captured: list[list] = []

    def _record_chat(messages, config=None, tools=None):
        captured.append(list(messages))
        return MagicMock(text="described", tokens_generated=1, tokens_prompt=1, stop_reason="stop")

    engine = MagicMock(is_loaded=True)
    engine.chat = MagicMock(side_effect=_record_chat)
    state.engine = engine
    state.current_model = sample_manifest
    return engine, captured


# ----------------------------------------------------------------------
# /api/chat — Ollama-native images[] field
# ----------------------------------------------------------------------


class TestOllamaChatImages:
    def test_image_bytes_reach_engine(self, client, vision_engine):
        engine, captured = vision_engine
        response = client.post(
            "/api/chat",
            json={
                "model": get_state().current_model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": "what is this?",
                        "images": [_png_base64()],
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 200
        engine.chat.assert_called_once()

        msgs = captured[0]
        assert msgs[0].images is not None
        assert len(msgs[0].images) == 1
        # Bytes are the decoded PNG — magic bytes must match.
        assert msgs[0].images[0].startswith(b"\x89PNG")

    def test_multiple_images_preserved_in_order(self, client, vision_engine):
        engine, captured = vision_engine
        first = _png_base64()
        second = base64.b64encode(_make_png(width=3, height=3)).decode()
        response = client.post(
            "/api/chat",
            json={
                "model": get_state().current_model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": "compare these",
                        "images": [first, second],
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 200
        imgs = captured[0][0].images
        assert len(imgs) == 2
        # Second image contains our 3x3 width marker at the IHDR offset.
        assert imgs[1][16:20] == struct.pack(">I", 3)

    def test_invalid_image_rejected_with_400(self, client, vision_engine):
        """Garbage payload → 400 before the engine is called."""
        response = client.post(
            "/api/chat",
            json={
                "model": get_state().current_model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": "broken image",
                        "images": ["this is not base64!!!"],
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 400
        engine, _ = vision_engine
        engine.chat.assert_not_called()

    def test_too_many_images_rejected_by_schema(self, client, vision_engine):
        """> 32 images → 422 (Pydantic enforces ``max_length=32``)."""
        response = client.post(
            "/api/chat",
            json={
                "model": get_state().current_model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": "too many",
                        "images": [_png_base64()] * 64,
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 422

    def test_text_only_message_has_no_images(self, client, vision_engine):
        """Ordinary chat without ``images`` leaves
        ``ChatMessage.images`` as None so old code paths don't regress."""
        engine, captured = vision_engine
        response = client.post(
            "/api/chat",
            json={
                "model": get_state().current_model.name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert captured[0][0].images is None


# ----------------------------------------------------------------------
# /v1/chat/completions — OpenAI content parts
# ----------------------------------------------------------------------


class TestOpenAIVisionContent:
    def test_text_plus_image_part_both_reach_engine(self, client, vision_engine):
        engine, captured = vision_engine
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": get_state().current_model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {"type": "image_url", "image_url": {"url": _png_data_uri()}},
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200
        engine.chat.assert_called_once()
        msg = captured[0][0]
        # Text from the text part lands in ``content``.
        assert "what is this" in msg.content
        # Image from the image_url part lands in ``images``.
        assert msg.images is not None
        assert msg.images[0].startswith(b"\x89PNG")

    def test_string_content_still_works(self, client, vision_engine):
        """Backwards compat: string ``content`` unchanged."""
        engine, captured = vision_engine
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": get_state().current_model.name,
                "messages": [{"role": "user", "content": "plain text"}],
            },
        )
        assert response.status_code == 200
        msg = captured[0][0]
        assert msg.content == "plain text"
        assert msg.images is None

    def test_http_url_rejected(self, client, vision_engine):
        """http(s) URLs in image_url.url are rejected — HFL doesn't
        fetch remote assets (SSRF guard)."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": get_state().current_model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/cat.png"},
                            }
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 400
        body = response.json()
        # The error message mentions the required data-URI form.
        msg = body.get("error") or body.get("detail") or ""
        assert "data:" in str(msg)

    def test_multiple_text_parts_concatenated(self, client, vision_engine):
        engine, captured = vision_engine
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": get_state().current_model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "alpha"},
                            {"type": "text", "text": "beta"},
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200
        assert captured[0][0].content == "alpha beta"

    def test_unknown_part_type_rejected(self, client, vision_engine):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": get_state().current_model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "input_audio", "input_audio": {}}],
                    }
                ],
            },
        )
        # Schema will reject at 422 because ``input_audio`` isn't in
        # the discriminated union.
        assert response.status_code == 422


# ----------------------------------------------------------------------
# Helper module directly (unit tests for the split/decode functions)
# ----------------------------------------------------------------------


class TestSplitOpenAIContent:
    def test_string_roundtrip(self):
        from hfl.api.vision import split_openai_content

        text, images = split_openai_content("just text")
        assert text == "just text"
        assert images is None

    def test_empty_list_path_through_schema(self):
        """Empty list would be rejected by the schema before reaching
        this helper, but directly calling it mirrors Pydantic behaviour.
        """
        from hfl.api.vision import split_openai_content

        text, images = split_openai_content([])
        assert text == ""
        assert images is None


class TestDecodeOllamaImages:
    def test_none_returns_none(self):
        from hfl.api.vision import decode_ollama_images

        assert decode_ollama_images(None) is None

    def test_empty_list_returns_none(self):
        from hfl.api.vision import decode_ollama_images

        assert decode_ollama_images([]) is None

    def test_error_includes_index(self):
        """The failing image's index is reported so clients know which
        element in a batch was the bad one."""
        from hfl.api.vision import decode_ollama_images
        from hfl.exceptions import ValidationError

        with pytest.raises(ValidationError, match=r"images\[1\]"):
            decode_ollama_images([_png_base64(), "garbage@@@"])

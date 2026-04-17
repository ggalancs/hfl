# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the LlamaCpp vision / multimodal wiring (P0-6, R29).

Verifies three things without loading a real model:

1. ``_messages_to_llama_cpp`` converts ``ChatMessage.images`` into
   llama-cpp's list-of-parts content shape with correct MIME
   detection and ordering.
2. ``_build_vision_chat_handler`` picks the right handler class
   per detected architecture, falls back gracefully when the local
   llama-cpp-python lacks multimodal support.
3. End-to-end: ``LlamaCppEngine.load`` picks up an adjacent
   ``mmproj-*.gguf`` file, instantiates the vision handler, and
   sets ``_is_multimodal=True``.

The Llama class itself is patched so no real model loads.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from hfl.engine.base import ChatMessage
from hfl.engine.llama_cpp import (
    LlamaCppEngine,
    _build_vision_chat_handler,
)

# ----------------------------------------------------------------------
# _messages_to_llama_cpp — content conversion
# ----------------------------------------------------------------------


class TestMessagesConversion:
    def test_text_only_unchanged(self):
        messages = [ChatMessage(role="user", content="hello")]
        out = LlamaCppEngine._messages_to_llama_cpp(messages)
        assert out == [{"role": "user", "content": "hello"}]

    def test_images_produce_list_of_parts(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        messages = [ChatMessage(role="user", content="what is this?", images=[png])]
        out = LlamaCppEngine._messages_to_llama_cpp(messages)

        parts = out[0]["content"]
        assert isinstance(parts, list)
        # Text part first, then image part(s)
        assert parts[0] == {"type": "text", "text": "what is this?"}
        assert parts[1]["type"] == "image_url"
        # MIME was sniffed as PNG
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_mime_sniffed_per_image(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        gif = b"GIF89a" + b"\x00" * 20
        webp = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 16
        messages = [ChatMessage(role="user", content="multi", images=[png, jpeg, gif, webp])]
        parts = LlamaCppEngine._messages_to_llama_cpp(messages)[0]["content"]

        image_parts = [p for p in parts if p["type"] == "image_url"]
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;")
        assert image_parts[1]["image_url"]["url"].startswith("data:image/jpeg;")
        assert image_parts[2]["image_url"]["url"].startswith("data:image/gif;")
        assert image_parts[3]["image_url"]["url"].startswith("data:image/webp;")

    def test_content_empty_with_images(self):
        """When there's no text but there are images, the text part
        is omitted — llama-cpp doesn't want an empty ``text``."""
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        messages = [ChatMessage(role="user", content="", images=[png])]
        out = LlamaCppEngine._messages_to_llama_cpp(messages)
        parts = out[0]["content"]
        assert all(p.get("type") != "text" or p.get("text") for p in parts)
        # All remaining entries are image parts.
        assert all(p["type"] == "image_url" for p in parts)

    def test_tool_call_fields_preserved_with_images(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        msg = ChatMessage(
            role="assistant",
            content="here you go",
            images=[png],
            tool_calls=[{"function": {"name": "describe", "arguments": {}}}],
        )
        out = LlamaCppEngine._messages_to_llama_cpp([msg])
        assert out[0]["tool_calls"][0]["function"]["name"] == "describe"

    def test_base64_payload_roundtrips(self):
        """The base64 we encode into the URI is decodable back to
        the original image bytes."""
        png = b"\x89PNG\r\n\x1a\n" + b"MAGIC"
        parts = LlamaCppEngine._messages_to_llama_cpp(
            [ChatMessage(role="user", content="x", images=[png])]
        )[0]["content"]
        url = parts[1]["image_url"]["url"]
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == png


# ----------------------------------------------------------------------
# _build_vision_chat_handler — architecture dispatch
# ----------------------------------------------------------------------


def _install_fake_handlers(monkeypatch=None):
    """Patch llama_cpp.llama_chat_format with stub handler classes
    and return them so tests can assert which was used."""
    import sys
    import types

    fake = types.ModuleType("llama_cpp.llama_chat_format")

    class _StubHandler:
        def __init__(self, *, clip_model_path, verbose=False, **_):
            self.clip_model_path = clip_model_path
            self.verbose = verbose

        def __class_getitem__(cls, item):
            return cls

    class Gemma3ChatHandler(_StubHandler):
        pass

    class Llava15ChatHandler(_StubHandler):
        pass

    class Llava16ChatHandler(_StubHandler):
        pass

    class MoondreamChatHandler(_StubHandler):
        pass

    class Qwen25VLChatHandler(_StubHandler):
        pass

    fake.Gemma3ChatHandler = Gemma3ChatHandler
    fake.Llava15ChatHandler = Llava15ChatHandler
    fake.Llava16ChatHandler = Llava16ChatHandler
    fake.MoondreamChatHandler = MoondreamChatHandler
    fake.Qwen25VLChatHandler = Qwen25VLChatHandler

    parent = sys.modules.setdefault("llama_cpp", types.ModuleType("llama_cpp"))
    parent.llama_chat_format = fake  # type: ignore[attr-defined]
    sys.modules["llama_cpp.llama_chat_format"] = fake
    return fake


class TestVisionHandlerDispatch:
    @pytest.fixture(autouse=True)
    def _patch_handlers(self):
        before = {
            "llama_cpp": __import__("sys").modules.get("llama_cpp"),
            "llama_cpp.llama_chat_format": __import__("sys").modules.get(
                "llama_cpp.llama_chat_format"
            ),
        }
        _install_fake_handlers()
        yield
        import sys

        for name, module in before.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_gemma3_gets_gemma3_handler(self):
        h = _build_vision_chat_handler(architecture="gemma3", clip_model_path="/tmp/mmproj.gguf")
        assert type(h).__name__ == "Gemma3ChatHandler"
        assert h.clip_model_path == "/tmp/mmproj.gguf"

    def test_gemma4_also_gets_gemma3_handler(self):
        """Gemma 4 shares the Gemma 3 multimodal protocol."""
        h = _build_vision_chat_handler(architecture="gemma4", clip_model_path="/tmp/mmproj.gguf")
        assert type(h).__name__ == "Gemma3ChatHandler"

    def test_qwen_vl_gets_qwen25_handler(self):
        h = _build_vision_chat_handler(architecture="qwen2-vl", clip_model_path="/tmp/mmproj.gguf")
        assert type(h).__name__ == "Qwen25VLChatHandler"

    def test_llava_16_gets_v16(self):
        h = _build_vision_chat_handler(
            architecture="llava-v1.6", clip_model_path="/tmp/mmproj.gguf"
        )
        assert type(h).__name__ == "Llava16ChatHandler"

    def test_plain_llava_gets_v15(self):
        h = _build_vision_chat_handler(architecture="llava", clip_model_path="/tmp/mmproj.gguf")
        assert type(h).__name__ == "Llava15ChatHandler"

    def test_moondream(self):
        h = _build_vision_chat_handler(architecture="moondream", clip_model_path="/tmp/mmproj.gguf")
        assert type(h).__name__ == "MoondreamChatHandler"

    def test_unknown_arch_falls_back_to_llava15(self):
        h = _build_vision_chat_handler(
            architecture="mystery-vision", clip_model_path="/tmp/mmproj.gguf"
        )
        assert type(h).__name__ == "Llava15ChatHandler"


class TestVisionHandlerImportFallback:
    def test_missing_multimodal_module_returns_none(self):
        """If llama-cpp-python is too old to ship multimodal
        handlers, we return None and skip multimodal support
        rather than crashing."""
        import sys
        import types

        before = sys.modules.get("llama_cpp.llama_chat_format")

        # Install a fake top-level llama_cpp so the import itself
        # succeeds, but omit the llama_chat_format submodule —
        # and map it to None so ``from ... import X`` raises
        # ImportError on the names.
        class BrokenModule:
            def __getattr__(self, name):
                raise ImportError(f"no {name}")

        sys.modules["llama_cpp"] = types.ModuleType("llama_cpp")
        sys.modules["llama_cpp.llama_chat_format"] = BrokenModule()  # type: ignore[assignment]
        try:
            h = _build_vision_chat_handler(architecture="gemma3", clip_model_path="/tmp/m.gguf")
            assert h is None
        finally:
            if before is None:
                sys.modules.pop("llama_cpp.llama_chat_format", None)
            else:
                sys.modules["llama_cpp.llama_chat_format"] = before


# ----------------------------------------------------------------------
# End-to-end LlamaCppEngine.load with an adjacent mmproj file
# ----------------------------------------------------------------------


class TestLoadAutodetectsCLIPProjector:
    @pytest.fixture
    def vision_model_tree(self, tmp_path):
        """Create a fake GGUF + a paired mmproj so ``load`` discovers it."""
        model = tmp_path / "gemma-3-4b.gguf"
        # Write a minimal GGUF magic header so the file passes the
        # ``.gguf`` suffix check; GGUF parsing is mocked below.
        model.write_bytes(b"GGUF" + b"\x00" * 64)
        mmproj = tmp_path / "mmproj-gemma-3-4b.gguf"
        mmproj.write_bytes(b"GGUF" + b"\x00" * 64)
        return model, mmproj

    def test_mmproj_sibling_triggers_multimodal_load(self, vision_model_tree):
        model, mmproj = vision_model_tree

        _install_fake_handlers()

        captured: dict = {}

        class FakeLlama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def close(self):
                pass

        with (
            patch("hfl.engine.llama_cpp.Llama", FakeLlama),
            patch(
                "hfl.engine.llama_cpp._read_gguf_model_info",
                return_value={"architecture": "gemma3"},
            ),
            patch(
                "hfl.engine.llama_cpp._preflight_memory_check",
                return_value=None,
            ),
        ):
            engine = LlamaCppEngine()
            engine.load(str(model))

        assert engine._is_multimodal is True
        # ``chat_handler`` was passed to Llama, ``chat_format`` was
        # stripped so the two don't fight each other.
        assert "chat_handler" in captured
        assert captured.get("chat_format") is None
        # Handler was the Gemma3 one (matches architecture).
        assert type(captured["chat_handler"]).__name__ == "Gemma3ChatHandler"
        # Handler was pointed at our mmproj file.
        assert captured["chat_handler"].clip_model_path == str(mmproj)

    def test_no_mmproj_sibling_loads_text_only(self, tmp_path):
        model = tmp_path / "text-only.gguf"
        model.write_bytes(b"GGUF" + b"\x00" * 64)

        class FakeLlama:
            def __init__(self, **kwargs):
                pass

            def close(self):
                pass

        with (
            patch("hfl.engine.llama_cpp.Llama", FakeLlama),
            patch(
                "hfl.engine.llama_cpp._read_gguf_model_info",
                return_value={"architecture": "llama3"},
            ),
            patch(
                "hfl.engine.llama_cpp._preflight_memory_check",
                return_value=None,
            ),
        ):
            engine = LlamaCppEngine()
            engine.load(str(model))
        assert engine._is_multimodal is False

    def test_explicit_clip_path_wins_over_sibling_scan(self, vision_model_tree, tmp_path):
        """When ``clip_model_path`` is explicitly passed it must NOT
        be shadowed by an auto-detected sibling."""
        _install_fake_handlers()
        model, _sibling = vision_model_tree
        explicit = tmp_path / "other-mmproj.gguf"
        explicit.write_bytes(b"GGUF" + b"\x00" * 64)

        captured: dict = {}

        class FakeLlama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def close(self):
                pass

        with (
            patch("hfl.engine.llama_cpp.Llama", FakeLlama),
            patch(
                "hfl.engine.llama_cpp._read_gguf_model_info",
                return_value={"architecture": "gemma3"},
            ),
            patch(
                "hfl.engine.llama_cpp._preflight_memory_check",
                return_value=None,
            ),
        ):
            engine = LlamaCppEngine()
            engine.load(str(model), clip_model_path=str(explicit))

        handler = captured["chat_handler"]
        assert handler.clip_model_path == str(explicit)


def _dummy_path():
    """Empty path helper — silences the ``imported but unused`` lint
    on ``pathlib.Path`` when a test shape evolves."""
    return Path("/")

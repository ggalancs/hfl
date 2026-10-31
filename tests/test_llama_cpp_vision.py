# SPDX-License-Identifier: Apache-2.0
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
import contextlib
import logging
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
        # Image part(s) first, then the text: SmolVLM named the colour of
        # three images with them first, and of none after the question.
        assert parts[1] == {"type": "text", "text": "what is this?"}
        assert parts[0]["type"] == "image_url"
        # MIME was sniffed as PNG
        assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

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
        url = parts[0]["image_url"]["url"]
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


@contextlib.contextmanager
def _fake_handlers():
    """Install the fake chat handlers and put ``sys.modules`` back after.

    The ``setdefault`` above is load-bearing in a way that bites: when the
    real ``llama_cpp`` is not currently imported it creates an EMPTY
    ``ModuleType`` and hangs only ``llama_chat_format`` on it. Left in
    place, that husk shadows the real package for the rest of the session
    — every later ``import llama_cpp`` finds a module with no ctypes
    bindings, so anything asking about its real surface raises
    AttributeError and anything merely checking the module exists is
    quietly testing the stub.

    One class restored by hand and two did not, which is why this is a
    fixture now instead of three copies of the same six lines.
    """
    import sys

    watched = ("llama_cpp", "llama_cpp.llama_chat_format")
    before = {name: sys.modules.get(name) for name in watched}
    real_parent = before["llama_cpp"]
    had_attr = hasattr(real_parent, "llama_chat_format") if real_parent else False
    original_attr = getattr(real_parent, "llama_chat_format", None) if real_parent else None
    try:
        yield _install_fake_handlers()
    finally:
        for name, module in before.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        # ``_install_fake_handlers`` also mutates the real parent in place.
        if real_parent is not None:
            if had_attr:
                real_parent.llama_chat_format = original_attr
            else:
                try:
                    del real_parent.llama_chat_format
                except AttributeError:
                    pass


class TestVisionHandlerDispatch:
    @pytest.fixture(autouse=True)
    def _patch_handlers(self):
        with _fake_handlers():
            yield

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

    def test_a_llama_with_its_own_template_never_gets_llava15s(self):
        """Without the generic handler (an older llama-cpp-python), a
        ``llama`` GGUF with a template (SmolVLM) stays text-only with it:
        LLaVA 1.5's template had it answer "1." to a question about France."""
        h = _build_vision_chat_handler(
            architecture="llama", clip_model_path="/tmp/mmproj.gguf", has_template=True
        )
        assert h is None

    def test_plain_llava_gets_v15(self):
        h = _build_vision_chat_handler(architecture="llava", clip_model_path="/tmp/mmproj.gguf")
        assert type(h).__name__ == "Llava15ChatHandler"

    def test_moondream(self):
        h = _build_vision_chat_handler(architecture="moondream", clip_model_path="/tmp/mmproj.gguf")
        assert type(h).__name__ == "MoondreamChatHandler"

    def test_a_llava15_gguf_names_its_language_model(self):
        h = _build_vision_chat_handler(architecture="llama", clip_model_path="/tmp/mmproj.gguf")
        assert type(h).__name__ == "Llava15ChatHandler"

    @pytest.mark.parametrize("arch", ["qwen35", "llama4", "mistral3", "mystery-vision"])
    def test_a_family_without_a_handler_is_text_only(self, arch, caplog):
        """LLaVA 1.5's handler was tried for these: its chat template
        replaced the model's on every request, and Qwen3.8 (``qwen35``)
        answered empty replies. Text-only keeps the model's template."""
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_cpp"):
            h = _build_vision_chat_handler(architecture=arch, clip_model_path="/tmp/mmproj.gguf")
        assert h is None
        assert "without images" in caplog.text and arch in caplog.text


class TestVisionHandlerImportFallback:
    def test_missing_multimodal_module_returns_none(self, monkeypatch):
        """If llama-cpp-python is too old to ship multimodal
        handlers, we return None and skip multimodal support
        rather than crashing.

        Both entries are restored. The ``finally`` here used to put back
        only ``llama_cpp.llama_chat_format`` and leave the bare
        ``ModuleType`` standing in for ``llama_cpp`` itself for the rest
        of the session. Two files later, a test that introspects the real
        package raised AttributeError — and, worse, any test in between
        that merely checked the backend was importable was quietly
        checking this stub.

        ``monkeypatch.setitem`` swaps the dict entry and swaps it back,
        which is the safe shape: deleting a native extension and letting
        it re-import mid-process crashes the interpreter.
        """
        import sys
        import types

        # A top-level llama_cpp that imports fine, with a submodule that
        # raises ImportError on every name — the shape of an old release
        # without multimodal support.
        class BrokenModule:
            def __getattr__(self, name):
                raise ImportError(f"no {name}")

        monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
        monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", BrokenModule())

        h = _build_vision_chat_handler(architecture="gemma3", clip_model_path="/tmp/m.gguf")
        assert h is None


# ----------------------------------------------------------------------
# End-to-end LlamaCppEngine.load with an adjacent mmproj file
# ----------------------------------------------------------------------


class TestLoadAutodetectsCLIPProjector:
    @pytest.fixture(autouse=True)
    def _patch_handlers(self):
        """Same restoring fixture as TestVisionHandlerDispatch.

        Without it, the bare ``_install_fake_handlers()`` calls below left
        a stub ``llama_cpp`` in ``sys.modules`` for the rest of the
        session. It surfaced two files later as an AttributeError on
        ``llama_model_params``, in a test that passes on its own.
        """
        with _fake_handlers():
            yield

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


# ----------------------------------------------------------------------
# The model's own template (MTMDChatHandler) for a GGUF that has one
# ----------------------------------------------------------------------


class TestTheModelsOwnTemplate:
    """A per-family handler brings a fixed template that replaced the
    model's on every request, text ones too: Qwen3.8 under LLaVA 1.5's
    answered empty replies, SmolVLM (a ``llama`` architecture) answered
    "1." to "What is the capital of France?" (measured, 2026-10-31). A GGUF
    with its own template keeps it, whatever its architecture."""

    @pytest.fixture(autouse=True)
    def fake(self):
        with _fake_handlers() as fake:

            class MTMDChatHandler:
                def __init__(self, *, clip_model_path, verbose=False, **_):
                    self.clip_model_path = clip_model_path

                def _get_chat_template(self, llama_model):
                    return llama_model.metadata["tokenizer.chat_template"]

                def __call__(self, **kwargs):
                    return kwargs

            fake.MTMDChatHandler = MTMDChatHandler
            yield fake

    @pytest.mark.parametrize("arch", ["qwen35", "llama", "gemma3", "qwen2vl", "llava", None])
    def test_any_architecture_with_a_template_keeps_it(self, fake, arch):
        h = _build_vision_chat_handler(
            architecture=arch, clip_model_path="/p/mm.gguf", has_template=True
        )
        assert isinstance(h, fake.MTMDChatHandler)
        assert h.uses_model_template is True and h.clip_model_path == "/p/mm.gguf"

    def test_without_a_template_the_family_handlers_remain(self):
        h = _build_vision_chat_handler(architecture="llama", clip_model_path="/p")
        assert type(h).__name__ == "Llava15ChatHandler"  # an old LLaVA 1.5 GGUF
        assert _build_vision_chat_handler(architecture="qwen35", clip_model_path="/p") is None

    def test_it_gets_the_request_variables_and_the_repaired_template(self, monkeypatch):
        import types

        h = _build_vision_chat_handler(
            architecture="qwen35", clip_model_path="/p", has_template=True
        )
        h.template_vars = {"enable_thinking": False}
        assert h(messages=[])["enable_thinking"] is False
        monkeypatch.setattr("hfl.models.chat_template.repair_chat_template", lambda t: t + "!")
        model = types.SimpleNamespace(metadata={"tokenizer.chat_template": "T"})
        assert h._get_chat_template(model) == "T!"

    def _load(self, tmp_path, metadata, handler_init=None):
        model = tmp_path / "qwen3.8.gguf"
        model.write_bytes(b"GGUF" + b"\x00" * 64)
        (tmp_path / "mmproj-F16.gguf").write_bytes(b"GGUF" + b"\x00" * 64)

        class FakeLlama:
            def __init__(self, **kwargs):
                self.chat_handler = kwargs.get("chat_handler")
                self.metadata = metadata

            def close(self):
                pass

        info = {"architecture": "qwen35", "has_chat_template": True}
        with (
            patch("hfl.engine.llama_cpp.Llama", FakeLlama),
            patch("hfl.engine.llama_cpp._read_gguf_model_info", return_value=info),
            patch("hfl.engine.llama_cpp._preflight_memory_check", return_value=None),
        ):
            if handler_init is not None:
                import sys

                generic = sys.modules["llama_cpp.llama_chat_format"].MTMDChatHandler
                generic._init_mtmd_context = handler_init
            engine = LlamaCppEngine()
            engine.load(str(model))
        return engine

    def test_the_engine_hands_its_template_the_tools(self, tmp_path):
        pytest.importorskip("jinja2")  # the template probe renders it
        template = "{% for t in tools %}{{ t.function.name }}{% endfor %}{{ messages[-1].content }}"
        engine = self._load(tmp_path, {"tokenizer.chat_template": template})
        assert engine.supports_images is True
        assert engine._model.chat_handler.uses_model_template is True
        assert engine._formatters == [engine._model.chat_handler]  # gets template_vars
        assert engine._template_knows_tools is True

    def test_a_projector_that_does_not_load_leaves_the_model_text_only(self, tmp_path, caplog):
        def refuse(self, model):
            raise ValueError("Failed to load mtmd context")

        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_cpp"):
            engine = self._load(tmp_path, {"tokenizer.chat_template": "x"}, refuse)
        assert engine.supports_images is False and engine._model.chat_handler is None
        assert "does not load" in caplog.text

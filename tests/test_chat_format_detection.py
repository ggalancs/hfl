# SPDX-License-Identifier: HRUL-1.0
"""Unit tests for ``hfl.engine.llama_cpp._detect_chat_format_from_gguf``.

Newer Gemma family GGUFs (released after Gemma 4) ship without an
embedded ``tokenizer.chat_template``. llama-cpp-python's auto-detection
then guesses the Llama-2 ``[INST]`` format, which silently destroys
chat quality. The detection helper reads ``general.architecture`` from
the GGUF header and explicitly maps Gemma family architectures to the
correct ``chat_format`` string, restoring the right prompt template.

These tests run in default CI (no ``gguf`` package installed) by
patching the ``gguf`` import via ``sys.modules`` injection.
"""

from __future__ import annotations

import sys
import types

import pytest

from hfl.engine.llama_cpp import (
    _ARCHITECTURE_CHAT_FORMAT,
    _detect_chat_format_from_gguf,
)


def _fake_gguf_module(arch: str | None) -> types.ModuleType:
    """Build a fake ``gguf`` module whose GGUFReader returns a single
    ``general.architecture`` field with the given value (or no field
    at all when ``arch`` is None)."""
    fake = types.ModuleType("gguf")

    class _FakeField:
        def __init__(self, value: str) -> None:
            self.parts = [value.encode("utf-8")]

    class _FakeReader:
        def __init__(self, path: str) -> None:
            self.path = path
            self.fields: dict = {}
            if arch is not None:
                self.fields["general.architecture"] = _FakeField(arch)

    fake.GGUFReader = _FakeReader  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def patched_gguf(monkeypatch):
    """Helper to inject a fake gguf module that reports a chosen arch."""

    def _install(arch: str | None) -> None:
        monkeypatch.setitem(sys.modules, "gguf", _fake_gguf_module(arch))

    return _install


# --- Architecture map sanity --------------------------------------------------


class TestArchitectureMap:
    def test_all_gemma_variants_are_mapped(self):
        for variant in ("gemma", "gemma2", "gemma3", "gemma4"):
            assert _ARCHITECTURE_CHAT_FORMAT[variant] == "gemma"


# --- Detection function -------------------------------------------------------


class TestDetectChatFormat:
    def test_gemma4_maps_to_gemma(self, patched_gguf):
        patched_gguf("gemma4")
        assert _detect_chat_format_from_gguf("/dummy/path.gguf") == "gemma"

    @pytest.mark.parametrize("arch", ["gemma", "gemma2", "gemma3", "gemma4"])
    def test_every_gemma_variant_maps_to_gemma(self, patched_gguf, arch):
        patched_gguf(arch)
        assert _detect_chat_format_from_gguf("/dummy/path.gguf") == "gemma"

    def test_unknown_architecture_returns_none(self, patched_gguf):
        """For architectures we don't override, return None so
        llama-cpp-python's own auto-detection takes over."""
        patched_gguf("qwen3")
        assert _detect_chat_format_from_gguf("/dummy/path.gguf") is None

    def test_missing_architecture_field_returns_none(self, patched_gguf):
        patched_gguf(None)
        assert _detect_chat_format_from_gguf("/dummy/path.gguf") is None

    def test_no_gguf_package_returns_none(self, monkeypatch):
        """If the optional ``gguf`` package isn't installed at all, the
        helper must return ``None`` (and not raise)."""
        # Hide gguf from sys.modules and force the import to fail.
        monkeypatch.setitem(sys.modules, "gguf", None)
        assert _detect_chat_format_from_gguf("/dummy/path.gguf") is None


# --- Integration: load() picks up the format ---------------------------------


def _install_stub_llama(monkeypatch, captured: dict) -> None:
    """Replace ``hfl.engine.llama_cpp.Llama`` with a stub that records
    the constructor kwargs. The module-level ``Llama`` symbol is the
    one ``LlamaCppEngine.load`` actually calls (it's bound at import
    time via a try/except to support installs without the optional
    ``[llama]`` extra)."""
    from hfl.engine import llama_cpp as engine_module

    class _StubLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(engine_module, "Llama", _StubLlama)


class TestLoadUsesDetectedFormat:
    def test_load_passes_chat_format_to_llama(self, monkeypatch, patched_gguf, tmp_path):
        """When ``LlamaCppEngine.load`` is called without an explicit
        ``chat_format``, the detection helper's output must be forwarded
        to the underlying ``Llama`` constructor."""
        from hfl.engine import llama_cpp as engine_module

        patched_gguf("gemma4")

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        # Create a dummy ``.gguf`` file so the path validation passes.
        dummy = tmp_path / "model.gguf"
        dummy.write_bytes(b"GGUF\x00\x00\x00\x00")

        engine = engine_module.LlamaCppEngine()
        engine.load(str(dummy), n_gpu_layers=0, verbose=True)

        assert captured.get("chat_format") == "gemma"
        assert captured.get("model_path") == str(dummy.resolve())

    def test_explicit_chat_format_overrides_detection(self, monkeypatch, patched_gguf, tmp_path):
        """A caller passing ``chat_format=`` explicitly wins over the
        auto-detection."""
        from hfl.engine import llama_cpp as engine_module

        patched_gguf("gemma4")  # would normally yield "gemma"

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        dummy = tmp_path / "model.gguf"
        dummy.write_bytes(b"GGUF\x00\x00\x00\x00")

        engine = engine_module.LlamaCppEngine()
        engine.load(
            str(dummy),
            n_gpu_layers=0,
            verbose=True,
            chat_format="chatml",
        )

        assert captured.get("chat_format") == "chatml"

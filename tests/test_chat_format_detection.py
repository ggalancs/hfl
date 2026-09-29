# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``hfl.engine.llama_cpp._detect_chat_format_from_gguf``.

Newer Gemma family GGUFs (released after Gemma 4) ship without an
embedded ``tokenizer.chat_template``. llama-cpp-python's auto-detection
then guesses the Llama-2 ``[INST]`` format, which silently destroys
chat quality. The detection helper reads ``general.architecture`` from
the GGUF header and explicitly maps Gemma family architectures to the
correct ``chat_format`` string, restoring the right prompt template.

These tests run in default CI (no ``gguf`` package installed) on real
GGUF headers written by ``tests/gguf_file.py``.
"""

from __future__ import annotations

import sys

import pytest

from hfl.engine.llama_cpp import (
    _ARCHITECTURE_CHAT_FORMAT,
    _detect_chat_format_from_gguf,
)
from tests.gguf_file import model_fields, write_gguf


@pytest.fixture
def patched_gguf(tmp_path):
    """A real GGUF header reporting ``arch`` (no architecture key when
    None); returns its path. HFL reads it with its own header reader."""

    def _install(arch: str | None) -> str:
        return str(write_gguf(tmp_path / f"{arch or 'none'}.gguf", model_fields(arch)))

    return _install


# --- Architecture map sanity --------------------------------------------------


class TestArchitectureMap:
    def test_all_gemma_variants_are_mapped(self):
        for variant in ("gemma", "gemma2", "gemma3", "gemma4"):
            assert _ARCHITECTURE_CHAT_FORMAT[variant] == "gemma"


# --- Detection function -------------------------------------------------------


class TestDetectChatFormat:
    def test_gemma4_maps_to_gemma(self, patched_gguf):
        assert _detect_chat_format_from_gguf(patched_gguf("gemma4")) == "gemma"

    @pytest.mark.parametrize("arch", ["gemma", "gemma2", "gemma3", "gemma4"])
    def test_every_gemma_variant_maps_to_gemma(self, patched_gguf, arch):
        assert _detect_chat_format_from_gguf(patched_gguf(arch)) == "gemma"

    def test_unknown_architecture_returns_none(self, patched_gguf):
        """For architectures we don't override, return None so
        llama-cpp-python's own auto-detection takes over."""
        assert _detect_chat_format_from_gguf(patched_gguf("qwen3")) is None

    def test_missing_architecture_field_returns_none(self, patched_gguf):
        assert _detect_chat_format_from_gguf(patched_gguf(None)) is None

    def test_works_without_the_gguf_package(self, monkeypatch, patched_gguf):
        """Detection no longer needs the optional ``gguf`` package: without
        it, Gemma models went back to the wrong prompt format."""
        monkeypatch.setitem(sys.modules, "gguf", None)  # importing it fails
        assert _detect_chat_format_from_gguf(patched_gguf("gemma4")) == "gemma"

    def test_a_missing_file_returns_none(self, tmp_path):
        assert _detect_chat_format_from_gguf(str(tmp_path / "absent.gguf")) is None


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
        from pathlib import Path

        from hfl.engine import llama_cpp as engine_module

        dummy = Path(patched_gguf("gemma4"))

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(str(dummy), n_gpu_layers=0, verbose=True)

        assert captured.get("chat_format") == "gemma"
        assert captured.get("model_path") == str(dummy.resolve())

    def test_explicit_chat_format_overrides_detection(self, monkeypatch, patched_gguf, tmp_path):
        """A caller passing ``chat_format=`` explicitly wins over the
        auto-detection."""
        from pathlib import Path

        from hfl.engine import llama_cpp as engine_module

        dummy = Path(patched_gguf("gemma4"))  # would normally yield "gemma"

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(
            str(dummy),
            n_gpu_layers=0,
            verbose=True,
            chat_format="chatml",
        )

        assert captured.get("chat_format") == "chatml"


class TestModelInfoIsCachedPerFile:
    """Read once per version of the file: it was read three to four times
    per load with a parser that took 2.8 s for a 0.5B model's header, so a
    model that loads in 0.2 s answered its first request in ~13 s."""

    def test_a_second_read_does_not_open_the_file(self, tmp_path, monkeypatch):
        from hfl.converter import gguf_header
        from hfl.engine.llama_cpp import _read_gguf_model_info

        path = str(write_gguf(tmp_path / "m.gguf", model_fields("qwen2", block_count=24)))
        calls: list[str] = []
        real = gguf_header.read_fields

        def counted(p, wanted=None):
            calls.append(str(p))
            return real(p, wanted)

        monkeypatch.setattr(gguf_header, "read_fields", counted)
        first = _read_gguf_model_info(path)
        opened = len(calls)
        assert _read_gguf_model_info(path) == first and len(calls) == opened

    def test_a_replaced_file_is_read_again(self, tmp_path):
        import os

        from hfl.engine.llama_cpp import _read_gguf_model_info

        path = tmp_path / "m.gguf"
        write_gguf(path, model_fields("qwen2", block_count=24))
        assert _read_gguf_model_info(str(path))["block_count"] == 24
        write_gguf(path, model_fields("qwen2", block_count=28), size_bytes=4096)
        os.utime(path, ns=(1, 2))  # a different mtime, whatever the clock's resolution
        assert _read_gguf_model_info(str(path))["block_count"] == 28

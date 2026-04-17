# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the ``tokenizer.ggml.add_bos_token`` plumbing (Phase 11 P1)."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


class _FakeLlama:
    instances: list[dict] = []

    def __init__(self, **kwargs):
        _FakeLlama.instances.append(kwargs)
        self.kwargs = kwargs

    def tokenize(self, data, add_bos=True):
        _FakeLlama.instances[-1].setdefault("_tokenize_calls", []).append(
            {"data": data, "add_bos": add_bos}
        )
        return [0, 1, 2]


@pytest.fixture
def fake_llama(monkeypatch):
    import hfl.engine.llama_cpp as lc

    _FakeLlama.instances.clear()
    monkeypatch.setattr(lc, "Llama", _FakeLlama)
    # Install fake enums for the KV-cache-type probe used in Phase 11.
    fake_c = ModuleType("llama_cpp.llama_cpp")
    fake_c.GGML_TYPE_F16 = 1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", fake_c)
    fake_parent = sys.modules.get("llama_cpp") or ModuleType("llama_cpp")
    fake_parent.llama_cpp = fake_c  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_parent)
    return _FakeLlama


def _patch_gguf(monkeypatch, add_bos):
    import hfl.engine.llama_cpp as lc

    monkeypatch.setattr(
        lc,
        "_read_gguf_model_info",
        lambda _p: {"architecture": "llama", "add_bos_token": add_bos},
    )


class TestTokenizerAddBOS:
    def test_add_bos_false_is_remembered(self, temp_config, fake_llama, monkeypatch):
        _patch_gguf(monkeypatch, add_bos=False)
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")

        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine.load(str(gguf))
        # Attribute is what later tokenize() sites read.
        assert engine._tokenizer_add_bos is False

    def test_add_bos_true_when_missing_from_gguf(self, temp_config, fake_llama, monkeypatch):
        import hfl.engine.llama_cpp as lc

        monkeypatch.setattr(
            lc,
            "_read_gguf_model_info",
            lambda _p: {"architecture": "llama"},
        )
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")

        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine.load(str(gguf))
        assert engine._tokenizer_add_bos is True

    def test_add_bos_true_when_explicit_true(self, temp_config, fake_llama, monkeypatch):
        _patch_gguf(monkeypatch, add_bos=True)
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")
        from hfl.engine.llama_cpp import LlamaCppEngine

        engine = LlamaCppEngine()
        engine.load(str(gguf))
        assert engine._tokenizer_add_bos is True

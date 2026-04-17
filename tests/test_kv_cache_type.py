# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for KV cache quantisation wiring (Phase 11 P1)."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


class _FakeLlama:
    instances: list[dict] = []

    def __init__(self, **kwargs):
        _FakeLlama.instances.append(kwargs)
        self.kwargs = kwargs

    def tokenize(self, *_a, **_k):
        return [0]


@pytest.fixture
def fake_llama(monkeypatch):
    import hfl.engine.llama_cpp as lc

    _FakeLlama.instances.clear()
    monkeypatch.setattr(lc, "Llama", _FakeLlama)
    monkeypatch.setattr(lc, "_read_gguf_model_info", lambda _p: None)

    # Fake ``llama_cpp.llama_cpp`` sub-module exposing enum ints so
    # the engine can resolve q4_0 / q8_0 to integer codes without
    # needing the real C-extension installed.
    fake_c = ModuleType("llama_cpp.llama_cpp")
    fake_c.GGML_TYPE_Q4_0 = 2  # type: ignore[attr-defined]
    fake_c.GGML_TYPE_Q8_0 = 8  # type: ignore[attr-defined]
    fake_c.GGML_TYPE_F16 = 1  # type: ignore[attr-defined]
    fake_c.GGML_TYPE_F32 = 0  # type: ignore[attr-defined]
    # Seat both ``llama_cpp`` and ``llama_cpp.llama_cpp`` in sys.modules
    # so the deferred ``from llama_cpp import llama_cpp`` resolves.
    fake_parent = sys.modules.get("llama_cpp")
    if fake_parent is None:
        fake_parent = ModuleType("llama_cpp")
        monkeypatch.setitem(sys.modules, "llama_cpp", fake_parent)
    fake_parent.llama_cpp = fake_c  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", fake_c)
    return _FakeLlama


class TestKVCacheType:
    def test_f16_leaves_type_unset(self, temp_config, fake_llama):
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")
        from hfl.engine.llama_cpp import LlamaCppEngine

        LlamaCppEngine().load(str(gguf), kv_cache_type="f16")
        assert "type_k" not in fake_llama.instances[-1]
        assert "type_v" not in fake_llama.instances[-1]

    def test_q4_0_sets_both_types(self, temp_config, fake_llama):
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")
        from hfl.engine.llama_cpp import LlamaCppEngine

        LlamaCppEngine().load(str(gguf), kv_cache_type="q4_0")
        last = fake_llama.instances[-1]
        assert last["type_k"] == 2
        assert last["type_v"] == 2

    def test_q8_0_sets_both_types(self, temp_config, fake_llama):
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")
        from hfl.engine.llama_cpp import LlamaCppEngine

        LlamaCppEngine().load(str(gguf), kv_cache_type="q8_0")
        last = fake_llama.instances[-1]
        assert last["type_k"] == 8
        assert last["type_v"] == 8

    def test_unknown_type_warns_and_falls_back(self, temp_config, fake_llama, monkeypatch):
        import hfl.engine.llama_cpp as lc

        warnings: list[str] = []
        monkeypatch.setattr(
            lc.logger,
            "warning",
            lambda msg, *a, **kw: warnings.append(msg % a if a else msg),
        )
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")
        from hfl.engine.llama_cpp import LlamaCppEngine

        LlamaCppEngine().load(str(gguf), kv_cache_type="z8_0")
        assert "type_k" not in fake_llama.instances[-1]
        assert any("kv_cache_type" in w for w in warnings)

    def test_config_default_applies(self, temp_config, fake_llama, monkeypatch):
        temp_config.kv_cache_type = "q8_0"
        gguf = temp_config.home_dir / "m.gguf"
        gguf.write_bytes(b"fake")
        from hfl.engine.llama_cpp import LlamaCppEngine

        LlamaCppEngine().load(str(gguf))
        assert fake_llama.instances[-1]["type_k"] == 8


class TestConfigField:
    def test_default_is_f16(self, temp_config):
        assert temp_config.kv_cache_type == "f16"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HFL_KV_CACHE_TYPE", "q4_0")
        from hfl.config import HFLConfig

        assert HFLConfig().kv_cache_type == "q4_0"

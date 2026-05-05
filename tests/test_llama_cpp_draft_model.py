# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for V4 F5 speculative-decoding wiring in ``LlamaCppEngine``.

The Llama constructor is replaced with a stub that records every
instantiation, so we can assert that:

- the draft model is loaded as a separate ``Llama`` when a path is
  provided,
- the target's ``draft_model`` kwarg gets the draft instance,
- the draft is freed by ``unload()``,
- a draft load failure does not abort the target load (best-effort
  speculation).

The real llama-cpp-python build is not required.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from hfl.engine import llama_cpp as engine_module

# --- Helpers (lifted from test_llama_cpp_preflight) -------------------------


def _fake_gguf_module(arch: str = "qwen") -> types.ModuleType:
    fake = types.ModuleType("gguf")

    class _Field:
        def __init__(self, value):
            if isinstance(value, str):
                self.parts = [value.encode("utf-8")]
            else:
                self.parts = [int(value).to_bytes(8, "little", signed=False)]

    class _Reader:
        def __init__(self, path):
            self.fields = {"general.architecture": _Field(arch)}

    fake.GGUFReader = _Reader  # type: ignore[attr-defined]
    return fake


def _stub_memory(monkeypatch):
    """Force the preflight to think we have 64 GB available."""
    from hfl.engine import memory as memory_module

    class _Snap:
        system_used_gb = 16.0
        system_available_gb = 64.0
        system_total_gb = 80.0
        gpu_used_gb = None
        gpu_available_gb = None
        gpu_total_gb = None
        gpu_id = None

    monkeypatch.setattr(memory_module, "HAS_PSUTIL", True)
    monkeypatch.setattr(memory_module, "get_memory_snapshot", lambda gpu_id=0: _Snap())


@pytest.fixture
def gguf_path(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF\x00\x00\x00\x00" + b"\x00" * 1024)
    return str(p)


@pytest.fixture
def draft_gguf(tmp_path):
    p = tmp_path / "draft.gguf"
    p.write_bytes(b"GGUF\x00\x00\x00\x00" + b"\x00" * 512)
    return str(p)


@pytest.fixture
def fake_gguf_for_qwen(monkeypatch):
    monkeypatch.setitem(sys.modules, "gguf", _fake_gguf_module(arch="qwen"))


@pytest.fixture
def stub_llama_capture(monkeypatch):
    """Replace ``Llama`` with a stub that records every ``__init__``
    call and supports being passed as ``draft_model=``."""
    instances: list[dict] = []

    class _StubLlama:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            instances.append(kwargs)

    monkeypatch.setattr(engine_module, "Llama", _StubLlama)
    return instances


# --- Tests ------------------------------------------------------------------


class TestDraftModelWiring:
    def test_no_draft_path_does_not_load_a_second_llama(
        self, monkeypatch, fake_gguf_for_qwen, stub_llama_capture, gguf_path
    ):
        _stub_memory(monkeypatch)

        engine = engine_module.LlamaCppEngine()
        engine.load(gguf_path, n_gpu_layers=0, verbose=True)

        # Exactly one Llama created (the target).
        assert len(stub_llama_capture) == 1
        assert "draft_model" not in stub_llama_capture[0]
        assert engine._draft_model is None

    def test_draft_path_loads_draft_then_target_with_draft_adapter(
        self,
        monkeypatch,
        fake_gguf_for_qwen,
        stub_llama_capture,
        gguf_path,
        draft_gguf,
    ):
        _stub_memory(monkeypatch)

        engine = engine_module.LlamaCppEngine()
        engine.load(
            gguf_path,
            n_gpu_layers=0,
            verbose=True,
            draft_model_path=draft_gguf,
        )

        # Two instantiations: draft first, target second.
        assert len(stub_llama_capture) == 2
        draft_kwargs, target_kwargs = stub_llama_capture
        assert draft_kwargs["model_path"] == draft_gguf
        # Target carries an adapter under ``draft_model`` — NOT the raw
        # Llama. llama-cpp-python expects ``LlamaDraftModel`` (callable
        # ndarray -> ndarray), and ``Llama.__call__`` returns a dict.
        assert "draft_model" in target_kwargs
        adapter = target_kwargs["draft_model"]
        # The adapter wraps the draft Llama instance and is callable.
        assert callable(adapter)
        # The raw draft is still tracked for ``unload`` cleanup.
        assert engine._draft_model is not None

    def test_draft_load_failure_does_not_block_target(
        self,
        monkeypatch,
        fake_gguf_for_qwen,
        gguf_path,
        draft_gguf,
    ):
        _stub_memory(monkeypatch)
        instances: list[dict] = []
        call_count = {"n": 0}

        class _StubLlama:
            def __init__(self, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    # First call is the draft — fail it.
                    raise RuntimeError("draft load: bad shape")
                instances.append(kwargs)

        monkeypatch.setattr(engine_module, "Llama", _StubLlama)

        engine = engine_module.LlamaCppEngine()
        # Must NOT raise — the draft failure logs a warning and
        # continues without speculation.
        engine.load(
            gguf_path,
            n_gpu_layers=0,
            verbose=True,
            draft_model_path=draft_gguf,
        )

        # Target still built (one successful Llama).
        assert len(instances) == 1
        # And no draft is tracked.
        assert engine._draft_model is None
        # The target's kwargs don't carry a draft_model.
        assert "draft_model" not in instances[0]

    def test_prompt_lookup_mode_does_not_load_a_second_llama(
        self,
        monkeypatch,
        fake_gguf_for_qwen,
        stub_llama_capture,
        gguf_path,
    ):
        """``draft_model_path="prompt-lookup"`` must use the
        zero-VRAM LlamaPromptLookupDecoding rather than instantiating
        another ``Llama``."""
        _stub_memory(monkeypatch)

        # The CI venv lacks the [llama] extra, so
        # ``llama_cpp.llama_speculative`` doesn't exist. Inject a fake
        # module so the engine's ``from llama_cpp.llama_speculative
        # import LlamaPromptLookupDecoding`` succeeds.
        ctor_calls = []

        class _StubLookup:
            def __init__(self, **kwargs):
                ctor_calls.append(kwargs)

        fake_pkg = types.ModuleType("llama_cpp")
        fake_spec_mod = types.ModuleType("llama_cpp.llama_speculative")
        fake_spec_mod.LlamaPromptLookupDecoding = _StubLookup  # type: ignore[attr-defined]
        fake_pkg.llama_speculative = fake_spec_mod  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "llama_cpp", fake_pkg)
        monkeypatch.setitem(sys.modules, "llama_cpp.llama_speculative", fake_spec_mod)

        engine = engine_module.LlamaCppEngine()
        engine.load(
            gguf_path,
            n_gpu_layers=0,
            verbose=True,
            draft_model_path="prompt-lookup",
        )

        # Only ONE Llama (the target) — prompt-lookup has no model.
        assert len(stub_llama_capture) == 1
        # The target's draft_model is the lookup decoder instance.
        assert "draft_model" in stub_llama_capture[0]
        assert ctor_calls, "LlamaPromptLookupDecoding was never constructed"
        # And no per-model draft is tracked (nothing to unload).
        assert engine._draft_model is None

    def test_unload_releases_both_target_and_draft(
        self,
        monkeypatch,
        fake_gguf_for_qwen,
        stub_llama_capture,
        gguf_path,
        draft_gguf,
    ):
        _stub_memory(monkeypatch)

        engine = engine_module.LlamaCppEngine()
        engine.load(
            gguf_path,
            n_gpu_layers=0,
            verbose=True,
            draft_model_path=draft_gguf,
        )

        assert engine._draft_model is not None
        engine.unload()
        assert engine._model is None
        assert engine._draft_model is None


class TestLlamaModelDraftAdapter:
    """Direct tests for the ``_LlamaModelDraftAdapter`` callable that
    bridges a small Llama into the LlamaDraftModel protocol."""

    def _draft(self, *, sample_returns):
        """Build a fake Llama whose ``sample`` returns the next id
        from a fixed sequence each time it is called."""
        seq = iter(sample_returns)
        d = MagicMock(spec=["reset", "eval", "sample"])
        d.reset = MagicMock()
        d.eval = MagicMock()
        d.sample = MagicMock(side_effect=lambda **kwargs: next(seq))
        return d

    def test_returns_intc_array_of_predicted_tokens(self):
        import numpy as np

        from hfl.engine.llama_cpp import _LlamaModelDraftAdapter

        draft = self._draft(sample_returns=[101, 102, 103])
        adapter = _LlamaModelDraftAdapter(draft, num_pred_tokens=3)
        out = adapter(np.array([1, 2, 3], dtype=np.intc))

        assert out.dtype == np.intc
        assert list(out) == [101, 102, 103]
        # First call has nothing in the cache → no reset, just a
        # full prefill via eval(input_ids).
        draft.reset.assert_not_called()
        # eval(prefill) + 3 × eval([tok]) one per generated token.
        assert draft.eval.call_count == 4
        assert draft.sample.call_count == 3

    def test_incremental_call_reuses_kv_cache(self):
        """Second call with an extended-prefix input only evaluates
        the suffix — no reset, no full prefill."""
        import numpy as np

        from hfl.engine.llama_cpp import _LlamaModelDraftAdapter

        draft = self._draft(sample_returns=[101, 102, 201, 202])
        adapter = _LlamaModelDraftAdapter(draft, num_pred_tokens=2)

        adapter(np.array([1, 2, 3], dtype=np.intc))
        # _processed = [1, 2, 3, 101, 102] (3 prompt + 2 predictions).
        eval_calls_before = draft.eval.call_count
        # Target accepted both predictions and now asks for the next
        # round. The new input_ids extends the previous _processed
        # by one token (the target sampled "999").
        adapter(np.array([1, 2, 3, 101, 102, 999], dtype=np.intc))

        # Only the new suffix [999] should have been eval'd, plus
        # the two new sampled tokens.
        new_eval_calls = draft.eval.call_count - eval_calls_before
        assert new_eval_calls == 3  # 1 suffix + 2 sampled
        # No reset — the cache is intact.
        draft.reset.assert_not_called()

    def test_divergent_input_resets_draft(self):
        """When the new input_ids diverges from the cached state
        (cancelled previous request, fresh prompt), the adapter must
        reset and replay."""
        import numpy as np

        from hfl.engine.llama_cpp import _LlamaModelDraftAdapter

        draft = self._draft(sample_returns=[101, 102, 201, 202])
        adapter = _LlamaModelDraftAdapter(draft, num_pred_tokens=2)

        adapter(np.array([1, 2, 3], dtype=np.intc))
        # Now ask for an entirely different sequence.
        adapter(np.array([99, 99, 99], dtype=np.intc))

        # Reset must have been called on the divergent second pass.
        draft.reset.assert_called_once()

    def test_empty_input_returns_empty_array(self):
        import numpy as np

        from hfl.engine.llama_cpp import _LlamaModelDraftAdapter

        draft = self._draft(sample_returns=[])
        adapter = _LlamaModelDraftAdapter(draft, num_pred_tokens=5)
        out = adapter(np.array([], dtype=np.intc))

        assert out.dtype == np.intc
        assert len(out) == 0

    def test_sample_failure_returns_partial_array_not_raise(self):
        import numpy as np

        from hfl.engine.llama_cpp import _LlamaModelDraftAdapter

        draft = MagicMock(spec=["reset", "eval", "sample"])
        draft.reset = MagicMock()
        draft.eval = MagicMock()
        # First sample succeeds, second raises — adapter should not
        # propagate so the target keeps running.
        draft.sample = MagicMock(side_effect=[42, RuntimeError("kv cache busy")])

        adapter = _LlamaModelDraftAdapter(draft, num_pred_tokens=5)
        out = adapter(np.array([1, 2, 3], dtype=np.intc))

        assert out.dtype == np.intc
        # Only the one successful prediction is returned.
        assert list(out) == [42]

    def test_eval_failure_returns_empty(self):
        """``eval()`` raising during the alignment step → no
        candidates this round (target keeps decoding plain)."""
        import numpy as np

        from hfl.engine.llama_cpp import _LlamaModelDraftAdapter

        draft = MagicMock(spec=["reset", "eval", "sample"])
        draft.reset = MagicMock()
        draft.eval = MagicMock(side_effect=RuntimeError("kv cache busy"))
        draft.sample = MagicMock(return_value=42)

        adapter = _LlamaModelDraftAdapter(draft, num_pred_tokens=5)
        out = adapter(np.array([1, 2, 3], dtype=np.intc))

        # Empty result tells llama-cpp "no candidates this round" and
        # the target falls back to plain decoding.
        assert out.dtype == np.intc
        assert len(out) == 0

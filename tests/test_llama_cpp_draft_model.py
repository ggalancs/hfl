# SPDX-License-Identifier: Apache-2.0
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
from tests.gguf_file import model_fields, write_gguf

# --- Helpers (lifted from test_llama_cpp_preflight) -------------------------


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
    """A real qwen GGUF header (no tensors), padded to 1 KB."""
    return str(write_gguf(tmp_path / "model.gguf", model_fields("qwen"), size_bytes=1032))


@pytest.fixture
def draft_gguf(tmp_path):
    return str(write_gguf(tmp_path / "draft.gguf", model_fields("qwen"), size_bytes=520))


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
        self, monkeypatch, stub_llama_capture, gguf_path
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


class TestThreadsOnTheCpu:
    """llama-cpp-python's default is half the cores: 44.8 tok/s on a 4-core
    CPU where all four gave 89.0. On the CPU alone HFL asks for every
    physical core and the model takes turns with others; with a GPU the
    default stays."""

    def test_cpu_only_uses_every_physical_core_and_takes_turns(
        self, monkeypatch, stub_llama_capture, gguf_path
    ):
        _stub_memory(monkeypatch)
        monkeypatch.setattr(engine_module, "_offloads_to_gpu", lambda layers: False)
        monkeypatch.setattr(engine_module, "_physical_cores", lambda: 6)
        engine = engine_module.LlamaCppEngine()
        engine.load(gguf_path, verbose=True)
        assert stub_llama_capture[-1]["n_threads"] == 6
        assert engine.generates_on_all_cpu_cores is True

    def test_with_a_gpu_the_default_stays(self, monkeypatch, stub_llama_capture, gguf_path):
        _stub_memory(monkeypatch)
        monkeypatch.setattr(engine_module, "_offloads_to_gpu", lambda layers: True)
        engine = engine_module.LlamaCppEngine()
        engine.load(gguf_path, verbose=True)
        assert stub_llama_capture[-1]["n_threads"] is None
        assert engine.generates_on_all_cpu_cores is False

    def test_an_explicit_thread_count_wins(self, monkeypatch, stub_llama_capture, gguf_path):
        _stub_memory(monkeypatch)
        monkeypatch.setattr(engine_module, "_offloads_to_gpu", lambda layers: False)
        engine = engine_module.LlamaCppEngine()
        engine.load(gguf_path, n_threads=3, verbose=True)
        assert stub_llama_capture[-1]["n_threads"] == 3

    def test_zero_gpu_layers_is_the_cpu(self):
        assert engine_module._offloads_to_gpu(0) is False

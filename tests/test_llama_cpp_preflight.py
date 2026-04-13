# SPDX-License-Identifier: HRUL-1.0
"""Regression tests for ``LlamaCppEngine`` memory-safety guards.

These tests exist because of a real incident: loading a Gemma 4 GGUF
kernel-panicked a macOS host twice in a row. Root cause was a
combination of:

- ``n_ctx`` defaulting to ``0`` (let llama-cpp auto-detect from the
  GGUF header), which for Gemma 3/4 means 131072 tokens and a KV
  cache in the tens-to-hundreds of GB;
- ``n_gpu_layers=-1`` + Metal unified memory on macOS, so the KV
  cache allocation pins the entire system RAM pool;
- ``flash_attn=True`` by default on an architecture where llama-cpp-
  python's flash-attention path is not yet safe.

The fix adds three layers of defence inside ``LlamaCppEngine.load``:

1. **Arch-based ``n_ctx`` cap** — Gemma 3/4 cap to 8192 unless the
   caller explicitly passes ``n_ctx=``.
2. **Arch-based ``flash_attn`` disable** — Gemma 4 forces
   ``flash_attn=False`` unless the caller explicitly passes
   ``flash_attn=True``.
3. **Memory preflight** — estimates (weights + KV cache) against
   available system RAM and raises ``OutOfMemoryError`` when the
   model obviously can't fit.

Each of the guards below must stay green. If you need to loosen any
of them, first verify on the *exact* model that caused the incident.
"""

from __future__ import annotations

import sys
import types

import pytest

from hfl.engine import llama_cpp as engine_module
from hfl.exceptions import OutOfMemoryError

# --- Helpers -----------------------------------------------------------------


def _fake_gguf_module(
    arch: str | None = None,
    block_count: int | None = None,
    embedding_length: int | None = None,
    context_length: int | None = None,
    head_count: int | None = None,
    head_count_kv: int | None = None,
    chat_template: str | None = None,
) -> types.ModuleType:
    """Build a fake ``gguf`` module whose ``GGUFReader`` exposes the
    requested metadata fields. String fields are stored as UTF-8
    bytes; int fields are stored as a little-endian byte blob so the
    production ``_read_int`` helper can decode them without numpy.
    """
    fake = types.ModuleType("gguf")

    class _FakeStrField:
        def __init__(self, value: str) -> None:
            self.parts = [value.encode("utf-8")]

    class _FakeIntField:
        def __init__(self, value: int) -> None:
            # 8 bytes little-endian — matches uint64 which is what
            # GGUF uses for most layout integers.
            self.parts = [value.to_bytes(8, "little", signed=False)]

    class _FakeReader:
        def __init__(self, path: str) -> None:
            self.path = path
            self.fields: dict = {}
            if arch is not None:
                self.fields["general.architecture"] = _FakeStrField(arch)
            if arch is not None and block_count is not None:
                self.fields[f"{arch}.block_count"] = _FakeIntField(block_count)
            if arch is not None and embedding_length is not None:
                self.fields[f"{arch}.embedding_length"] = _FakeIntField(embedding_length)
            if arch is not None and context_length is not None:
                self.fields[f"{arch}.context_length"] = _FakeIntField(context_length)
            if arch is not None and head_count is not None:
                self.fields[f"{arch}.attention.head_count"] = _FakeIntField(head_count)
            if arch is not None and head_count_kv is not None:
                self.fields[f"{arch}.attention.head_count_kv"] = _FakeIntField(head_count_kv)
            if chat_template is not None:
                self.fields["tokenizer.chat_template"] = _FakeStrField(chat_template)

    fake.GGUFReader = _FakeReader  # type: ignore[attr-defined]
    return fake


def _install_stub_llama(monkeypatch, captured: dict) -> None:
    """Replace ``hfl.engine.llama_cpp.Llama`` with a stub that records
    constructor kwargs, so tests can assert on them without needing
    the real ``[llama]`` extra installed."""

    class _StubLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(engine_module, "Llama", _StubLlama)


def _stub_memory_snapshot(
    monkeypatch, *, available_gb: float, total_gb: float | None = None
) -> None:
    """Force ``get_memory_snapshot`` to return a known-sized budget so
    the preflight check is deterministic. Also flips ``HAS_PSUTIL`` to
    True in case the test host doesn't have psutil installed.
    """
    from hfl.engine import memory as memory_module

    class _Snapshot:
        def __init__(self, available: float, total: float) -> None:
            self.system_used_gb = total - available
            self.system_available_gb = available
            self.system_total_gb = total
            self.gpu_used_gb = None
            self.gpu_available_gb = None
            self.gpu_total_gb = None
            self.gpu_id = None

    def _fake_snapshot(gpu_id: int = 0) -> _Snapshot:
        return _Snapshot(available_gb, total_gb if total_gb is not None else available_gb)

    monkeypatch.setattr(memory_module, "HAS_PSUTIL", True)
    monkeypatch.setattr(memory_module, "get_memory_snapshot", _fake_snapshot)


@pytest.fixture
def dummy_gguf(tmp_path):
    """A tiny on-disk ``.gguf`` file whose path validation passes.

    ``size_mb`` can be overridden by calling ``dummy_gguf(size_mb=...)``
    — some tests need the file to *look* large so the preflight has
    something to reject.
    """

    def _make(size_mb: float = 0.001) -> str:
        path = tmp_path / "model.gguf"
        path.write_bytes(b"GGUF\x00\x00\x00\x00" + b"\x00" * int(size_mb * 1024 * 1024))
        return str(path)

    return _make


@pytest.fixture
def patched_gguf(monkeypatch):
    """Install a fake ``gguf`` module in ``sys.modules`` with a chosen
    set of layout fields."""

    def _install(**kwargs) -> None:
        monkeypatch.setitem(sys.modules, "gguf", _fake_gguf_module(**kwargs))

    return _install


# --- Architecture-based n_ctx cap --------------------------------------------


class TestGemmaContextCap:
    """When the caller doesn't pass an explicit ``n_ctx``, Gemma 3/4
    GGUFs must be capped to 8192 regardless of what the header
    advertises. The advertised 131072-token context is what crashed
    the host in the original incident.
    """

    @pytest.mark.parametrize("arch", ["gemma3", "gemma4"])
    def test_auto_ctx_is_capped_to_8192(self, monkeypatch, patched_gguf, dummy_gguf, arch):
        patched_gguf(
            arch=arch,
            block_count=42,
            embedding_length=3584,
            context_length=131072,
        )
        _stub_memory_snapshot(monkeypatch, available_gb=64.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True)

        assert captured["n_ctx"] == 8192, (
            f"{arch}: auto-detected n_ctx must be capped to the safe "
            f"default, got {captured.get('n_ctx')}"
        )

    def test_explicit_n_ctx_wins_over_cap(self, monkeypatch, patched_gguf, dummy_gguf):
        """If the caller explicitly passes ``n_ctx=``, we respect it —
        the cap is a safe *default*, not a hard limit. Users who know
        their hardware can fit a larger context window.
        """
        patched_gguf(
            arch="gemma4",
            block_count=42,
            embedding_length=3584,
            context_length=131072,
        )
        # 32k context on a 42-layer 3584-dim model ≈ 36 GB KV cache.
        # Pretend the host has plenty of room so the preflight lets
        # this through — we're only testing the cap bypass here.
        _stub_memory_snapshot(monkeypatch, available_gb=512.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_ctx=32768, n_gpu_layers=0, verbose=True)

        assert captured["n_ctx"] == 32768

    def test_non_gemma_arch_is_not_capped(self, monkeypatch, patched_gguf, dummy_gguf):
        """The cap is targeted at Gemma 3/4 specifically. Other
        architectures keep their existing auto-detect behaviour.
        """
        patched_gguf(arch="qwen3", block_count=32, embedding_length=2048)
        _stub_memory_snapshot(monkeypatch, available_gb=64.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True)

        # Non-Gemma default falls through to hfl_config.default_ctx_size
        # which is 0 (let llama-cpp-python decide).
        assert captured["n_ctx"] == 0


# --- Architecture-based flash_attn disable -----------------------------------


class TestGemmaFlashAttnDisable:
    """Gemma 4 must have ``flash_attn`` forced off unless the caller
    explicitly opts in. llama-cpp-python's flash-attention path for
    new arches has historically been crash-prone.
    """

    def test_gemma4_defaults_flash_attn_off(self, monkeypatch, patched_gguf, dummy_gguf):
        patched_gguf(arch="gemma4")
        _stub_memory_snapshot(monkeypatch, available_gb=64.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True)

        assert captured["flash_attn"] is False

    def test_gemma4_explicit_flash_attn_true_wins(self, monkeypatch, patched_gguf, dummy_gguf):
        patched_gguf(arch="gemma4")
        _stub_memory_snapshot(monkeypatch, available_gb=64.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True, flash_attn=True)

        assert captured["flash_attn"] is True

    def test_non_gemma_keeps_flash_attn_on(self, monkeypatch, patched_gguf, dummy_gguf):
        patched_gguf(arch="llama")
        _stub_memory_snapshot(monkeypatch, available_gb=64.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True)

        assert captured["flash_attn"] is True


# --- Memory preflight --------------------------------------------------------


class TestMemoryPreflight:
    """The preflight refuses to instantiate ``Llama`` when the
    estimated footprint exceeds ``_MEMORY_SAFETY_FRACTION`` of
    available system RAM. The raise must happen *before* we call the
    (real) Llama constructor — otherwise we're back to square one and
    the host crashes.
    """

    def test_oversized_gemma4_raises_before_loading(self, monkeypatch, patched_gguf, dummy_gguf):
        # 62-layer, 4608-dim Gemma 4 27B with a 32k user-set context.
        # KV cache ≈ 2 * 62 * 32768 * 4608 * 2 bytes ≈ 35 GB.
        # Plus a ~10 GB file. Against 16 GB of "available" RAM, this
        # has to be refused.
        patched_gguf(
            arch="gemma4",
            block_count=62,
            embedding_length=4608,
            context_length=131072,
        )
        _stub_memory_snapshot(monkeypatch, available_gb=16.0)

        # Track whether Llama was called — it must NOT be.
        constructed = {"count": 0}

        class _ExplodingStubLlama:
            def __init__(self, **kwargs):
                constructed["count"] += 1
                raise AssertionError(
                    "Llama constructor must not be reached when the preflight refuses the load"
                )

        monkeypatch.setattr(engine_module, "Llama", _ExplodingStubLlama)

        engine = engine_module.LlamaCppEngine()
        with pytest.raises(OutOfMemoryError) as exc_info:
            engine.load(
                dummy_gguf(size_mb=10.0),
                n_ctx=32768,
                n_gpu_layers=0,
                verbose=True,
            )

        assert constructed["count"] == 0
        # Mensaje específico para la familia Gemma.
        assert "gemma4" in exc_info.value.details

    def test_preflight_disabled_by_env_var(self, monkeypatch, patched_gguf, dummy_gguf):
        """``HFL_DISABLE_MEMORY_PREFLIGHT=1`` lets power users bypass
        the check (e.g. for a discrete GPU whose VRAM we don't read).
        """
        patched_gguf(
            arch="gemma4",
            block_count=62,
            embedding_length=4608,
        )
        _stub_memory_snapshot(monkeypatch, available_gb=1.0)
        monkeypatch.setenv("HFL_DISABLE_MEMORY_PREFLIGHT", "1")

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        # Would otherwise raise OutOfMemoryError.
        engine.load(
            dummy_gguf(size_mb=10.0),
            n_ctx=32768,
            n_gpu_layers=0,
            verbose=True,
        )
        assert captured.get("model_path")  # Llama was reached

    def test_no_psutil_skips_preflight(self, monkeypatch, patched_gguf, dummy_gguf):
        """If psutil isn't installed we can't measure anything, so
        the preflight must fall back to a no-op (with a warning) and
        let the load proceed — the arch-based caps still apply."""
        from hfl.engine import memory as memory_module

        monkeypatch.setattr(memory_module, "HAS_PSUTIL", False)

        patched_gguf(
            arch="gemma4",
            block_count=62,
            embedding_length=4608,
            context_length=131072,
        )

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True)

        # Arch cap still applied even without psutil.
        assert captured["n_ctx"] == 8192
        assert captured["flash_attn"] is False

    def test_preflight_passes_when_model_fits(self, monkeypatch, patched_gguf, dummy_gguf):
        """Baseline sanity: a small model with plenty of headroom
        must load cleanly through the new code path."""
        patched_gguf(arch="llama", block_count=32, embedding_length=2048)
        _stub_memory_snapshot(monkeypatch, available_gb=64.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(
            dummy_gguf(size_mb=2.0),
            n_ctx=4096,
            n_gpu_layers=0,
            verbose=True,
        )

        assert captured["n_ctx"] == 4096
        assert captured["flash_attn"] is True


# --- Estimation helper sanity -----------------------------------------------


class TestEstimator:
    """Direct unit tests for the estimator so we notice if somebody
    changes the formula without updating the preflight budget."""

    def test_weights_only_when_no_info(self, dummy_gguf):
        path = dummy_gguf(size_mb=1.0)
        gb = engine_module._estimate_memory_required_gb(path, None, 8192)
        # ~1 MB = ~0.001 GB
        assert 0.0005 < gb < 0.01

    def test_kv_cache_scales_linearly_with_ctx(self, dummy_gguf):
        path = dummy_gguf(size_mb=0.001)
        info = {
            "architecture": "gemma4",
            "block_count": 62,
            "embedding_length": 4608,
            "max_context": 131072,
        }
        small = engine_module._estimate_memory_required_gb(path, info, 8192)
        big = engine_module._estimate_memory_required_gb(path, info, 32768)
        # 4x n_ctx → ~4x KV cache (weights are negligible on a 1KB file).
        ratio = big / small if small > 0 else 0
        assert 3.5 < ratio < 4.5

    def test_missing_layout_zeroes_kv_component(self, dummy_gguf):
        path = dummy_gguf(size_mb=0.001)
        info = {
            "architecture": "gemma4",
            "block_count": None,
            "embedding_length": None,
            "max_context": None,
            "head_count": None,
            "head_count_kv": None,
        }
        gb = engine_module._estimate_memory_required_gb(path, info, 32768)
        # File is ~1 KB, no KV contribution possible.
        assert gb < 0.001

    def test_gqa_aware_estimate_is_tighter_than_naive(self, dummy_gguf):
        """Real Gemma 4 31B numbers: 60 layers, 5376 embed dim, 32
        heads, 4 KV heads (GQA 8:1). At ``n_ctx=8192`` the GQA-aware
        estimate should be ~8× smaller than the fallback that uses
        ``embedding_length`` directly. This is the fix that stops the
        preflight from falsely rejecting large GQA models that would
        actually fit in memory.
        """
        path = dummy_gguf(size_mb=0.001)
        gqa_info = {
            "architecture": "gemma4",
            "block_count": 60,
            "embedding_length": 5376,
            "head_count": 32,
            "head_count_kv": 4,
            "max_context": 262144,
        }
        fallback_info = {
            "architecture": "gemma4",
            "block_count": 60,
            "embedding_length": 5376,
            "head_count": None,
            "head_count_kv": None,
            "max_context": 262144,
        }
        gqa = engine_module._estimate_memory_required_gb(path, gqa_info, 8192)
        naive = engine_module._estimate_memory_required_gb(path, fallback_info, 8192)
        # With GQA 32/4 = 8:1, the GQA branch must be ~8× smaller
        # (tiny file size is negligible at this scale).
        ratio = naive / gqa if gqa > 0 else 0
        assert 7.0 < ratio < 9.0, (
            f"expected ~8× smaller GQA estimate, got ratio {ratio:.2f} "
            f"(gqa={gqa:.3f} GB, naive={naive:.3f} GB)"
        )

    def test_gqa_aware_estimate_matches_real_gemma4_31b(self, dummy_gguf):
        """Sanity check against the numbers we measured from the real
        NoxStrix/gemma-4-31B-it-Q4_K_M-GGUF file that kernel-panicked
        the host. At n_ctx=8192 the KV cache is ~1.23 GB; at the
        GGUF-advertised max of 262144 it's ~39.4 GB.
        """
        path = dummy_gguf(size_mb=0.001)
        info = {
            "architecture": "gemma4",
            "block_count": 60,
            "embedding_length": 5376,
            "head_count": 32,
            "head_count_kv": 4,
            "max_context": 262144,
        }
        at_8k = engine_module._estimate_memory_required_gb(path, info, 8192)
        at_max = engine_module._estimate_memory_required_gb(path, info, 262144)
        # ±10 % tolerance to absorb the negligible file-size term.
        assert 1.10 < at_8k < 1.35, f"at 8192: {at_8k:.2f} GB"
        assert 37.0 < at_max < 41.0, f"at 262144: {at_max:.2f} GB"


# --- Gemma 4 channel marker filter -------------------------------------------


class TestGemma4ChannelFilter:
    """Gemma 4 fine-tunes that ship without an embedded chat template
    make the model emit its training-format split-pipe reasoning
    markers (``<|channel>thought``, ``<channel|>``, etc.) as literal
    text in the chat output. The post-filter strips these so the user
    sees a clean answer.
    """

    def test_user_reported_incident_output_is_cleaned(self):
        """This is the exact string the user reported on 2026-04-13
        after the Gemma 4 load stopped kernel-panicking. It must come
        out as just the answer, with the thought wrapper gone.
        """
        raw = "<|channel>thought\n<channel|>Hello! How can I help you today?"
        cleaned = engine_module._strip_gemma4_channel_markers(raw)
        assert cleaned == "Hello! How can I help you today?", (
            f"unexpected cleaned output: {cleaned!r}"
        )

    def test_thought_block_with_content_is_fully_suppressed(self):
        raw = (
            "<|channel>thought\n"
            "The user is asking a simple greeting. I should respond politely.\n"
            "<channel|>"
            "Hi there!"
        )
        cleaned = engine_module._strip_gemma4_channel_markers(raw)
        assert cleaned == "Hi there!"

    def test_final_channel_wrappers_stripped_content_kept(self):
        raw = "<|channel>final\nThe capital of France is Paris.\n<channel|>"
        cleaned = engine_module._strip_gemma4_channel_markers(raw)
        assert cleaned.strip() == "The capital of France is Paris."

    def test_turn_markers_stripped(self):
        raw = "<|turn>assistant\nHello world<turn|>"
        cleaned = engine_module._strip_gemma4_channel_markers(raw)
        assert cleaned.strip() == "Hello world"

    def test_text_without_markers_is_passthrough(self):
        raw = "Plain text with no markers at all."
        assert engine_module._strip_gemma4_channel_markers(raw) == raw

    def test_multiple_thought_blocks_all_stripped(self):
        raw = (
            "<|channel>thought\nfirst thought<channel|>"
            "answer part one "
            "<|channel>thought\nsecond thought<channel|>"
            "answer part two"
        )
        cleaned = engine_module._strip_gemma4_channel_markers(raw)
        assert cleaned == "answer part one answer part two"


class TestGemma4StreamFilter:
    """Streaming filter must handle markers split across chunks
    because llama-cpp-python's stream iterator yields one token at
    a time and a marker token like ``<|channel>`` can share a chunk
    boundary with the preceding/following text.
    """

    def test_markers_split_across_chunks_are_still_stripped(self):
        chunks = [
            "<|channel>",
            "thought\n",
            "<channel|>",
            "Hello",
            "! How ",
            "can I help you today?",
        ]
        out = "".join(engine_module._filter_gemma4_stream(iter(chunks)))
        assert out == "Hello! How can I help you today?"

    def test_single_chunk_full_answer(self):
        chunks = ["<|channel>thought\n<channel|>Clean answer here."]
        out = "".join(engine_module._filter_gemma4_stream(iter(chunks)))
        assert out == "Clean answer here."

    def test_no_markers_passthrough(self):
        chunks = ["Hello", " ", "world", "!"]
        out = "".join(engine_module._filter_gemma4_stream(iter(chunks)))
        assert out == "Hello world!"

    def test_empty_stream(self):
        out = list(engine_module._filter_gemma4_stream(iter([])))
        assert out == []

    def test_trailing_partial_marker_is_dropped(self):
        """If the stream ends mid-marker (truncated generation), the
        flush step must still run the regex on the buffer so the
        orphan ``<|`` fragment doesn't leak."""
        chunks = ["The answer is: ", "<|channel>", "final\n", "42"]
        out = "".join(engine_module._filter_gemma4_stream(iter(chunks)))
        assert out == "The answer is: 42"


class TestGemma4EngineIntegration:
    """End-to-end check that ``LlamaCppEngine.chat`` and ``chat_stream``
    apply the filter when the loaded architecture is ``gemma4`` and
    leave other architectures untouched."""

    def _stub_chat(self, monkeypatch, content: str):
        """Replace ``LlamaCppEngine._model.create_chat_completion``
        with a stub returning ``content`` so we can test the filter
        without a real model."""

        class _StubModel:
            def create_chat_completion(self, **kwargs):
                if kwargs.get("stream"):

                    def _gen():
                        for ch in content:
                            yield {"choices": [{"delta": {"content": ch}}]}

                    return _gen()
                return {
                    "choices": [
                        {
                            "message": {"content": content, "tool_calls": None},
                        }
                    ],
                    "usage": {"completion_tokens": 0, "prompt_tokens": 0},
                }

        engine = engine_module.LlamaCppEngine()
        engine._model = _StubModel()
        return engine

    def test_chat_strips_markers_for_gemma4(self, monkeypatch):
        engine = self._stub_chat(
            monkeypatch,
            "<|channel>thought\n<channel|>Hello there!",
        )
        engine._architecture = "gemma4"

        from hfl.engine.base import ChatMessage

        result = engine.chat([ChatMessage(role="user", content="hi")])
        assert result.text == "Hello there!"

    def test_chat_does_not_strip_for_non_gemma4(self, monkeypatch):
        """The filter must be a no-op for other architectures, so we
        don't accidentally swallow legitimate ``<|...|>`` content
        from other formats (e.g. ChatML)."""
        raw = "<|channel>thought\n<channel|>Something"
        engine = self._stub_chat(monkeypatch, raw)
        engine._architecture = "llama"

        from hfl.engine.base import ChatMessage

        result = engine.chat([ChatMessage(role="user", content="hi")])
        assert result.text == raw

    def test_chat_stream_strips_markers_for_gemma4(self, monkeypatch):
        engine = self._stub_chat(
            monkeypatch,
            "<|channel>thought\n<channel|>Streamed clean answer.",
        )
        engine._architecture = "gemma4"

        from hfl.engine.base import ChatMessage

        out = "".join(engine.chat_stream([ChatMessage(role="user", content="hi")]))
        assert out == "Streamed clean answer."


# --- Embedded chat_template deference ---------------------------------------


class TestEmbeddedChatTemplateDeference:
    """Gemma 4 GGUFs from proper converters (bartowski, unsloth,
    lmstudio-community, official Google exports) ship a
    ``tokenizer.chat_template`` that already encodes the correct
    Gemma 4 prompt format with ``<|turn>`` / ``<|channel>``
    delimiters. Overriding ``chat_format`` with our static map
    (which only knows Gemma 2's ``<start_of_turn>`` format)
    silently corrupts the prompt. When a template is embedded we
    must leave ``chat_format=None`` so llama-cpp-python uses the
    Jinja template.

    This regression test exists because the incident workflow was:
      1. Load NoxStrix GGUF (no template) → arch override fires,
         prompt builds with Gemma 2 format, model emits channel
         markers. Fixed by post-filter.
      2. Replace with bartowski GGUF (has template) → arch override
         would STILL fire without this fix, re-breaking the prompt.
    """

    def test_gemma4_with_embedded_template_does_not_set_chat_format(
        self, monkeypatch, patched_gguf, dummy_gguf
    ):
        """The canonical bartowski / unsloth path: template is
        present, so ``chat_format`` must be left as ``None`` for
        llama-cpp-python to use the embedded Jinja."""
        patched_gguf(
            arch="gemma4",
            block_count=60,
            embedding_length=5376,
            head_count=32,
            head_count_kv=4,
            chat_template=(
                "{% for msg in messages %}<|turn>{{msg.role}}\n{{msg.content}}<turn|>{% endfor %}"
            ),
        )
        _stub_memory_snapshot(monkeypatch, available_gb=128.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True)

        assert captured.get("chat_format") is None, (
            f"expected chat_format=None (defer to embedded template), "
            f"got {captured.get('chat_format')!r}"
        )

    def test_gemma4_without_embedded_template_still_uses_static_override(
        self, monkeypatch, patched_gguf, dummy_gguf
    ):
        """The NoxStrix-style path: no template embedded, so the
        override must still kick in to avoid llama-cpp-python's
        ``[INST]`` fallback. This keeps the pre-incident fix intact."""
        patched_gguf(arch="gemma4")  # no chat_template
        _stub_memory_snapshot(monkeypatch, available_gb=128.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True)

        assert captured.get("chat_format") == "gemma"

    def test_explicit_chat_format_overrides_embedded_template(
        self, monkeypatch, patched_gguf, dummy_gguf
    ):
        """Caller-supplied ``chat_format=`` always wins, even when a
        template is embedded. Advanced users know what they're doing."""
        patched_gguf(
            arch="gemma4",
            chat_template="{% if true %}x{% endif %}",
        )
        _stub_memory_snapshot(monkeypatch, available_gb=128.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(
            dummy_gguf(),
            n_gpu_layers=0,
            verbose=True,
            chat_format="chatml",
        )

        assert captured.get("chat_format") == "chatml"

    def test_non_gemma_with_embedded_template_is_untouched(
        self, monkeypatch, patched_gguf, dummy_gguf
    ):
        """Non-Gemma architectures were never in the override map, so
        this is mostly a sanity check — both before and after the
        fix, llama gets ``chat_format=None`` for a llama GGUF with
        its own template."""
        patched_gguf(
            arch="llama",
            chat_template="{% for m in messages %}{{m.content}}{% endfor %}",
        )
        _stub_memory_snapshot(monkeypatch, available_gb=128.0)

        captured: dict = {}
        _install_stub_llama(monkeypatch, captured)

        engine = engine_module.LlamaCppEngine()
        engine.load(dummy_gguf(), n_gpu_layers=0, verbose=True)

        assert captured.get("chat_format") is None

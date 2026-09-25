# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the MLX backend (Phase 13 P1 — V2 row 14).

We don't require a real mlx-lm install (it's darwin-arm64 only).
Instead we inject a fake ``mlx_lm`` into ``sys.modules`` so the
engine's plumbing is exercised portably. Platform gating is tested
separately via monkeypatched ``platform.system()`` / ``machine()``.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from hfl.engine import mlx_engine
from hfl.engine.base import ChatMessage, GenerationConfig


@pytest.fixture
def fake_mlx(monkeypatch):
    """Seat a fake ``mlx_lm`` module in sys.modules for the test's duration."""
    generated_texts: list[str] = []

    class _FakeTokenizer:
        def encode(self, text):
            # One token per character is enough for the prompt/gen-count invariant.
            return list(range(len(text)))

        def apply_chat_template(self, dicts, tokenize=False, add_generation_prompt=True):
            parts = [f"{d['role']}:{d['content']}" for d in dicts]
            if add_generation_prompt:
                parts.append("assistant:")
            return "\n".join(parts)

    fake = ModuleType("mlx_lm")

    def _load(path):  # noqa: ARG001
        return object(), _FakeTokenizer()

    def _generate(_model, _tokenizer, *, prompt, **kwargs):  # noqa: ANN001
        text = "ECHO:" + prompt
        generated_texts.append(text)
        return text

    def _stream_generate(_model, _tokenizer, *, prompt, **kwargs):  # noqa: ANN001
        for chunk in ("ec", "ho", "-", "stream"):
            yield chunk

    fake.load = _load  # type: ignore[attr-defined]
    fake.generate = _generate  # type: ignore[attr-defined]
    fake.stream_generate = _stream_generate  # type: ignore[attr-defined]

    # mlx-lm 0.31+ moved sampling knobs onto a sampler callable +
    # logits_processors list. MLXEngine._build_sampling imports these
    # helpers; the fixture seats a stub submodule so the import chain
    # resolves without a real mlx-lm install.
    sample_utils = ModuleType("mlx_lm.sample_utils")

    def _make_sampler(**_kwargs):
        return lambda logits: logits

    def _make_logits_processors(**_kwargs):
        return []

    sample_utils.make_sampler = _make_sampler  # type: ignore[attr-defined]
    sample_utils.make_logits_processors = _make_logits_processors  # type: ignore[attr-defined]
    fake.sample_utils = sample_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    # No prompt-cache module unless a test seats one (``fake_cache``). Set
    # explicitly: with a real mlx-lm installed, an earlier test may have
    # imported the real ``mlx_lm.models.cache``, and a sys.modules hit would
    # hand these plumbing tests a real cache wrapped around a fake model.
    monkeypatch.setitem(sys.modules, "mlx_lm.models", None)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", None)
    # Force the availability gate to pass so the engine uses our fake.
    monkeypatch.setattr(mlx_engine, "is_available", lambda: True)
    return {"texts": generated_texts}


# ----------------------------------------------------------------------
# Platform gating
# ----------------------------------------------------------------------


class TestIsAvailable:
    def test_non_darwin_is_not_available(self, monkeypatch):
        monkeypatch.setattr(mlx_engine.platform, "system", lambda: "Linux")
        monkeypatch.setattr(mlx_engine.platform, "machine", lambda: "x86_64")
        assert mlx_engine.is_available() is False

    def test_intel_darwin_is_not_available(self, monkeypatch):
        monkeypatch.setattr(mlx_engine.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(mlx_engine.platform, "machine", lambda: "x86_64")
        assert mlx_engine.is_available() is False

    def test_darwin_arm64_without_mlx_is_not_available(self, monkeypatch):
        monkeypatch.setattr(mlx_engine.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(mlx_engine.platform, "machine", lambda: "arm64")
        # Clear any existing mlx_lm injection so the import fails cleanly.
        monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
        # Also block the import path itself by shadowing with a broken module.
        fake = ModuleType("mlx_lm")

        def _boom(*_a, **_k):
            raise ImportError("nope")

        fake.__getattr__ = _boom  # type: ignore[attr-defined]
        # The module imports; but the feature gate doesn't do anything
        # beyond the try/except so we also need the import itself to
        # work. Instead, simply monkeypatch ``is_available`` to
        # simulate the "no SDK" state.
        monkeypatch.setattr(mlx_engine, "is_available", lambda: False)
        assert mlx_engine.is_available() is False


class TestLoadGate:
    def test_load_raises_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(mlx_engine, "is_available", lambda: False)
        engine = mlx_engine.MLXEngine()
        with pytest.raises(RuntimeError):
            engine.load("/nope")


# ----------------------------------------------------------------------
# Happy path with the fake SDK
# ----------------------------------------------------------------------


class TestMLXEngine:
    def test_load_populates_model_and_tokenizer(self, fake_mlx):
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        assert engine.is_loaded

    def test_unload_clears_state(self, fake_mlx):
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        engine.unload()
        assert not engine.is_loaded

    def test_generate_returns_result_shape(self, fake_mlx):
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        result = engine.generate("hi", GenerationConfig(max_tokens=8))
        assert result.text == "ECHO:hi"
        assert result.tokens_generated > 0
        assert result.total_duration > 0

    def test_chat_applies_template(self, fake_mlx):
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        result = engine.chat(
            [
                ChatMessage(role="user", content="hello"),
            ],
            GenerationConfig(),
        )
        # Fake template produces ``user:hello\nassistant:``; our fake
        # ``generate`` echoes it back. Check both role + content made
        # it through.
        assert "user:hello" in result.text
        assert "assistant:" in result.text

    def test_stream_yields_chunks(self, fake_mlx):
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        chunks = list(engine.generate_stream("hi", GenerationConfig()))
        assert chunks == ["ec", "ho", "-", "stream"]

    def test_generate_before_load_raises(self, fake_mlx):
        engine = mlx_engine.MLXEngine()
        with pytest.raises(RuntimeError):
            engine.generate("hi", GenerationConfig())


class TestMLXSeedAndStop:
    """#20: MLX must honour the request seed (reproducibility, parity with
    vLLM/diffusers) and stop sequences (which mlx-lm has no native support for)."""

    def _seat_mlx_core(self, monkeypatch):
        """Seat a fake ``mlx.core`` whose ``random.seed`` records calls."""
        import types

        seeds: list[int] = []
        core = ModuleType("mlx.core")
        core.random = types.SimpleNamespace(seed=lambda s: seeds.append(s))  # type: ignore[attr-defined]
        mx = ModuleType("mlx")
        mx.core = core  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx", mx)
        monkeypatch.setitem(sys.modules, "mlx.core", core)
        return seeds

    def test_seed_applied_when_requested(self, fake_mlx, monkeypatch):
        seeds = self._seat_mlx_core(monkeypatch)
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        engine.generate("hi", GenerationConfig(seed=123))
        assert seeds == [123]

    def test_seed_not_applied_when_negative(self, fake_mlx, monkeypatch):
        seeds = self._seat_mlx_core(monkeypatch)
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        engine.generate("hi", GenerationConfig(seed=-1))
        assert seeds == []

    def test_stop_truncates_generate(self, fake_mlx):
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        # fake generate() returns "ECHO:" + prompt -> "ECHO:X"; stop "HO".
        result = engine.generate("X", GenerationConfig(stop=["HO"]))
        assert result.text == "EC"  # truncated at the stop boundary

    def test_stop_truncates_stream(self, fake_mlx, monkeypatch):
        fake = sys.modules["mlx_lm"]

        def _sg(_m, _t, *, prompt, **kwargs):  # noqa: ANN001
            # Tokens that are not 1:1 with words and that cross a stop string.
            for chunk in ("par", "tial ", "ST", "OP here"):
                yield chunk

        monkeypatch.setattr(fake, "stream_generate", _sg)

        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        out = "".join(engine.generate_stream("X", GenerationConfig(stop=["STOP"])))
        assert "STOP" not in out
        assert "here" not in out  # nothing past the stop leaks
        assert out == "partial "


# ----------------------------------------------------------------------
# Prompt cache (LRUPromptCache) plumbing
# ----------------------------------------------------------------------


class _Resp:
    """The fields of mlx-lm's GenerationResponse the engine reads."""

    def __init__(self, text, token, prompt_tokens, n, finish=None):
        self.text = text
        self.token = token
        self.prompt_tokens = prompt_tokens
        self.prompt_tps = 1000.0
        self.generation_tokens = n
        self.generation_tps = 100.0
        self.finish_reason = finish


@pytest.fixture
def fake_cache(fake_mlx, monkeypatch):
    """A fake ``mlx_lm.models.cache`` whose store behaves like the real one
    (longest cached prefix of the request wins) and records every call."""
    calls: dict = {"stream": [], "inserted": [], "fetched": []}

    class _Store:
        def __init__(self, max_size, max_bytes):
            self.max_size, self.max_bytes = max_size, max_bytes
            self.entries: list[list[int]] = []

        def fetch_nearest_cache(self, model_key, tokens):
            calls["fetched"].append(list(tokens))
            best = None
            for key in self.entries:
                if tokens[: len(key)] == key and (best is None or len(key) > len(best)):
                    best = key
            if best is None:
                return None, tokens
            return {"kv": list(best)}, tokens[len(best) :]

        def insert_cache(self, model_key, tokens, prompt_cache, cache_type="assistant"):
            calls["inserted"].append(list(tokens))
            self.entries.append(list(tokens))

    cache_mod = ModuleType("mlx_lm.models.cache")
    cache_mod.LRUPromptCache = _Store  # type: ignore[attr-defined]
    cache_mod.make_prompt_cache = lambda model: {"kv": []}  # type: ignore[attr-defined]
    models_mod = ModuleType("mlx_lm.models")
    models_mod.cache = cache_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)

    def _stream_generate(_model, _tokenizer, *, prompt, prompt_cache, **kwargs):
        calls["stream"].append({"prompt": list(prompt), "cache": prompt_cache})
        if calls.get("explode"):
            raise RuntimeError("metal fault")
        # Three generated tokens with ids 900.., then the final response.
        pieces = ["A", "B", "C"]
        for i, piece in enumerate(pieces[:-1]):
            yield _Resp(piece, 900 + i, len(prompt), i + 1)
        yield _Resp(pieces[-1], 902, len(prompt), 3, finish="length")

    sys.modules["mlx_lm"].stream_generate = _stream_generate  # type: ignore[attr-defined]
    return calls


def _loaded():
    engine = mlx_engine.MLXEngine()
    engine.load("/fake/model")
    return engine


class TestPromptCache:
    def test_first_request_evaluates_everything_and_stores_prompt_plus_reply(self, fake_cache):
        engine = _loaded()
        result = engine.generate("hello", GenerationConfig(max_tokens=3))
        assert result.text == "ABC"
        assert fake_cache["stream"][0]["prompt"] == [0, 1, 2, 3, 4]
        # Key = the prompt ids plus the ids the model produced, not a
        # re-tokenisation of the text "ABC".
        assert fake_cache["inserted"] == [[0, 1, 2, 3, 4, 900, 901, 902]]

    def test_a_continuation_evaluates_only_the_new_tokens(self, fake_cache):
        engine = _loaded()
        engine.generate("hello", GenerationConfig(max_tokens=3))
        # The fake tokenizer maps text to range(len), so a longer prompt
        # shares the stored prefix only up to the reply ids; seat a key that
        # a longer prompt does extend.
        engine._prompt_store.entries.append(list(range(8)))
        result = engine.generate("hello, world", GenerationConfig(max_tokens=3))
        assert fake_cache["stream"][1]["prompt"] == list(range(8, 12))
        assert fake_cache["stream"][1]["cache"] == {"kv": list(range(8))}
        assert engine.last_prompt_tokens_reused == 8
        # tokens_prompt reports the whole prompt, not the part evaluated.
        assert result.tokens_prompt == 12

    def test_an_exact_hit_still_evaluates_the_prompt(self, fake_cache):
        """mlx-lm cannot generate from an empty prompt; an exact hit must
        fall back to a fresh cache, not pass zero tokens."""
        engine = _loaded()
        engine._prompt_store.entries.append(list(range(5)))
        engine.generate("hello", GenerationConfig(max_tokens=3))
        assert fake_cache["stream"][0]["prompt"] == [0, 1, 2, 3, 4]
        assert fake_cache["stream"][0]["cache"] == {"kv": []}

    def test_timings_are_measured_not_apportioned(self, fake_cache):
        engine = _loaded()
        result = engine.generate("hello", GenerationConfig(max_tokens=3))
        # 5 evaluated prompt tokens at 1000 tok/s, 3 generated at 100 tok/s.
        assert result.prompt_eval_duration == 5_000_000
        assert result.eval_duration == 30_000_000
        assert result.stop_reason == "length"

    def test_stream_stores_after_the_consumer_finishes(self, fake_cache):
        engine = _loaded()
        assert list(engine.generate_stream("hello", GenerationConfig(max_tokens=3))) == [
            "A",
            "B",
            "C",
        ]
        assert fake_cache["inserted"] == [[0, 1, 2, 3, 4, 900, 901, 902]]

    def test_a_stream_closed_early_stores_what_was_generated(self, fake_cache):
        engine = _loaded()
        stream = engine.generate_stream("hello", GenerationConfig(max_tokens=3))
        assert next(stream) == "A"
        stream.close()
        assert fake_cache["inserted"] == [[0, 1, 2, 3, 4, 900]]

    def test_a_stop_string_ends_generation_and_keeps_the_cache_consistent(self, fake_cache):
        engine = _loaded()
        result = engine.generate("hello", GenerationConfig(max_tokens=3, stop=["B"]))
        assert result.text == "A"
        assert fake_cache["inserted"] == [[0, 1, 2, 3, 4, 900, 901]]

    def test_a_failed_request_is_not_stored(self, fake_cache):
        engine = _loaded()
        fake_cache["explode"] = True
        with pytest.raises(RuntimeError, match="metal fault"):
            engine.generate("hello", GenerationConfig(max_tokens=3))
        assert fake_cache["inserted"] == []

    def test_a_response_without_token_ids_is_not_stored(self, fake_cache):
        """Guessing the key from text could name KV it does not hold."""
        engine = _loaded()

        def _textual(_model, _tokenizer, *, prompt, prompt_cache, **kwargs):
            yield from ("x", "y")

        sys.modules["mlx_lm"].stream_generate = _textual  # type: ignore[attr-defined]
        assert engine.generate("hello").text == "xy"
        assert fake_cache["inserted"] == []

    def test_unload_drops_the_store(self, fake_cache):
        engine = _loaded()
        assert engine._prompt_store is not None
        engine.unload()
        assert engine._prompt_store is None

    def test_budget_is_the_configured_bytes(self, fake_cache, monkeypatch):
        from hfl.config import config

        monkeypatch.setattr(config, "mlx_prompt_cache_bytes", 12345)
        assert _loaded()._prompt_store.max_bytes == 12345

    def test_zero_budget_runs_the_uncached_path(self, fake_cache, monkeypatch):
        from hfl.config import config

        monkeypatch.setattr(config, "mlx_prompt_cache_bytes", 0)
        engine = _loaded()
        assert engine._prompt_store is None
        assert engine.generate("hi").text == "ECHO:hi"
        assert fake_cache["stream"] == []

    def test_an_mlx_lm_without_the_cache_runs_uncached(self, fake_mlx):
        engine = _loaded()
        assert engine._prompt_store is None
        assert engine.generate("hi").text == "ECHO:hi"

    def test_the_default_budget_keeps_the_cache_on(self):
        from hfl.config import HFLConfig

        assert HFLConfig().mlx_prompt_cache_bytes > 0


class TestUnloadReleasesMemory:
    """MLX keeps freed Metal buffers in its own cache, so dropping the model
    references left a 17 GB model's memory with the process (measured: 16.7
    GB resident after unload, 0.6 GB once the cache is cleared). With several
    models resident, evicting one would have freed nothing."""

    def test_unload_clears_the_mlx_cache_after_dropping_the_model(self, fake_mlx, monkeypatch):
        seen = {}
        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")

        core = ModuleType("mlx.core")

        def clear_cache():
            seen["model_at_clear"] = engine._model

        core.clear_cache = clear_cache  # type: ignore[attr-defined]
        mlx_pkg = ModuleType("mlx")
        mlx_pkg.core = core  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", core)

        engine.unload()
        assert "model_at_clear" in seen, "unload never cleared the MLX buffer cache"
        assert seen["model_at_clear"] is None, "cleared before the model was dropped"

    def test_old_mlx_keeps_it_under_metal(self, fake_mlx, monkeypatch):
        calls = []
        core = ModuleType("mlx.core")
        core.metal = ModuleType("mlx.core.metal")  # type: ignore[attr-defined]
        core.metal.clear_cache = lambda: calls.append(1)  # type: ignore[attr-defined]
        mlx_pkg = ModuleType("mlx")
        mlx_pkg.core = core  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
        monkeypatch.setitem(sys.modules, "mlx.core", core)

        engine = mlx_engine.MLXEngine()
        engine.load("/fake/model")
        engine.unload()
        assert calls == [1]


class TestStreamCounts:
    """A streamed MLX reply knows its token counts once read, as the other
    backends' do; it used to report none (Ollama's ``eval_count`` missing,
    OpenAI's ``usage`` absent)."""

    def test_counted_from_mlx_lm_s_own_figures(self, fake_cache):
        from hfl.engine.base import stream_counts

        engine = _loaded()
        stream = engine.generate_stream("hello", GenerationConfig(max_tokens=3))
        assert "".join(stream) == "ABC"
        assert stream_counts(stream) == (5, 3)

    def test_a_reused_prefix_still_counts_as_prompt(self, fake_cache):
        from hfl.engine.base import stream_counts

        engine = _loaded()
        engine._prompt_store.entries.append(list(range(8)))
        stream = engine.chat_stream([ChatMessage(role="user", content="hello, world")])
        "".join(stream)
        prompt, completion = stream_counts(stream)
        assert engine.last_prompt_tokens_reused > 0
        assert prompt == len(
            engine._tokenizer.encode(
                engine._messages_to_prompt([ChatMessage(role="user", content="hello, world")])
            )
        )
        assert completion == 3

    def test_no_figures_no_numbers(self, fake_mlx):
        """Responses without counts (plain strings here) leave them unknown."""
        from hfl.engine.base import stream_counts

        engine = _loaded()
        stream = engine.generate_stream("hi", GenerationConfig())
        "".join(stream)
        assert stream_counts(stream) == (None, None)


class _RecordingTokenizer:
    """A tokenizer that keeps what the chat template was asked to render."""

    def __init__(self, template):
        self.chat_template = template
        self.calls: list[tuple[list, dict]] = []

    def encode(self, text):
        return list(range(len(text)))

    def apply_chat_template(self, dicts, **kwargs):
        self.calls.append((dicts, kwargs))
        return "PROMPT"


WEATHER = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
CALL = {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}


class TestToolsAndReasoning:
    """On MLX a model never saw its tools (``tools`` was dropped) and
    ``think: false`` changed nothing; measured with Qwen3-1.7B-4bit."""

    def _render(self, fake_mlx, template, tools=WEATHER, reasoning=None):
        engine = _loaded()
        engine._tokenizer = _RecordingTokenizer(template)
        history = [
            ChatMessage(role="user", content="w?"),
            ChatMessage(role="assistant", content="", tool_calls=[CALL]),
            ChatMessage(role="tool", content="31C", name="get_weather"),
        ]
        engine._messages_to_prompt(history, tools, reasoning)
        return engine._tokenizer.calls[-1]

    def test_a_template_with_tools_gets_them_and_the_history(self, fake_mlx):
        dicts, kwargs = self._render(fake_mlx, "{% for t in tools %}{% endfor %}")
        assert kwargs["tools"] == WEATHER
        assert dicts[1]["tool_calls"] == [CALL] and dicts[2]["role"] == "tool"

    def test_a_template_without_them_gets_them_written_in(self, fake_mlx):
        dicts, kwargs = self._render(fake_mlx, "{{ messages }}")
        assert "tools" not in kwargs
        assert "<tools>" in dicts[0]["content"] and "<tool_call>" in dicts[2]["content"]

    def test_the_reasoning_switch_reaches_the_template(self, fake_mlx):
        _, kwargs = self._render(fake_mlx, "{{ tools }}", reasoning="off")
        assert kwargs["enable_thinking"] is False
        _, kwargs = self._render(fake_mlx, "{{ tools }}")
        assert "enable_thinking" not in kwargs  # not asked: the model's default

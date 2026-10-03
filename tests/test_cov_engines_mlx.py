# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The MLX engine and its batch scheduler, against stand-in ``mlx`` /
``mlx_lm`` modules: loading with an adapter, a draft or batching, the
batched paths of generate / stream / logprobs, the logprobs path itself,
prompt rendering fallbacks, stop strings and the scheduler's edge cases
(failed inserts, a late request after unload, a store that fails).

Neither ``mlx`` nor ``mlx_lm`` is installed in CI, so every module the
engine imports is a fake seated in ``sys.modules`` for the test only.
"""

from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

from hfl.engine import cancel, mlx_batch, mlx_engine
from hfl.engine.base import ChatMessage, GenerationConfig


class _Resp:
    """The fields of mlx-lm's GenerationResponse the engine reads."""

    def __init__(
        self,
        text,
        token,
        *,
        prompt_tokens=4,
        n=1,
        finish=None,
        logprobs=None,
        reused=0,
        from_draft=False,
    ):
        self.text = text
        self.token = token
        self.prompt_tokens = prompt_tokens
        self.prompt_tps = 1000.0
        self.generation_tokens = n
        self.generation_tps = 100.0
        self.finish_reason = finish
        self.logprobs = logprobs
        self.reused = reused
        self.from_draft = from_draft


class _L:
    """Logits that record what was done to them."""

    def __init__(self, v):
        self.v = v

    def __mul__(self, factor):
        return ("scaled", self.v, factor)

    def __eq__(self, other):
        return isinstance(other, _L) and other.v == self.v

    def __repr__(self):
        return f"_L({self.v!r})"


class _Tok:
    vocab_size = 10
    eos_token_ids = [0]
    chat_template = None

    def encode(self, text):
        return list(range(1, len(text) + 1))

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


class _FakeMx(types.ModuleType):
    """``mlx.core`` with what the engine and the sampler call."""

    float32 = "float32"

    def __init__(self):
        super().__init__("mlx.core")
        self.cleared = 0
        self.evaluated: list = []
        draws: list = []
        self.draws = draws

        class _Random:
            @staticmethod
            def key(seed):
                return ("key", seed)

            @staticmethod
            def split(key):
                return ("next", key), ("draw", key)

            @staticmethod
            def categorical(logits, key):
                draws.append((logits, key))
                return 7

        self.random = _Random()

    def clear_cache(self):
        self.cleared += 1

    def eval(self, value):
        self.evaluated.append(value)


@pytest.fixture
def mlx(monkeypatch):
    """Fake ``mlx_lm`` (load, stream_generate, sample_utils) and ``mlx.core``."""
    state: dict = {"loads": [], "stream": [], "responses": None, "fail_load": set()}

    fake = types.ModuleType("mlx_lm")

    def _load(path, **kwargs):
        state["loads"].append((path, kwargs))
        if path in state["fail_load"]:
            raise OSError(f"cannot read {path}")
        return f"model:{path}", _Tok()

    def _stream_generate(_model, _tokenizer, *, prompt, **kwargs):
        state["stream"].append({"prompt": prompt, **kwargs})
        responses = state["responses"]
        if responses is None:
            responses = [_Resp("A", 5), _Resp("B", 6), _Resp("C", 7, finish="stop")]
        yield from responses

    fake.load = _load  # type: ignore[attr-defined]
    fake.stream_generate = _stream_generate  # type: ignore[attr-defined]
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kw: ("greedy", kw)  # type: ignore[attr-defined]
    sample_utils.make_logits_processors = lambda **kw: []  # type: ignore[attr-defined]
    sample_utils.apply_top_k = lambda lp, k: _L(("top_k", k, lp))  # type: ignore[attr-defined]
    sample_utils.apply_top_p = lambda lp, p: _L(("top_p", p, lp))  # type: ignore[attr-defined]
    fake.sample_utils = sample_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", None)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", None)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", None)

    core = _FakeMx()
    pkg = types.ModuleType("mlx")
    pkg.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(mlx_engine, "is_available", lambda: True)
    state["mx"] = core
    return state


def _engine(**attrs) -> mlx_engine.MLXEngine:
    engine = mlx_engine.MLXEngine()
    engine._model, engine._tokenizer, engine._model_path = "model", _Tok(), "/m"
    for k, v in attrs.items():
        setattr(engine, k, v)
    return engine


class _FakeBatch:
    """The engine's view of ``BatchScheduler``: ``stream`` and ``close``."""

    def __init__(self, steps):
        self.steps = steps
        self.calls: list[dict] = []
        self.closed = False

    def stream(self, tokens, **kwargs):
        self.calls.append({"tokens": tokens, **kwargs})
        return iter(self.steps)

    def close(self):
        self.closed = True


# ----------------------------------------------------------------------
# Availability and small helpers
# ----------------------------------------------------------------------


def test_available_on_apple_silicon_with_mlx_lm(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setattr(mlx_engine.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_engine.platform, "machine", lambda: "ARM64")
    assert mlx_engine.is_available() is True


def test_as_float32_converts_only_what_numpy_cannot_read(mlx):
    plain = [1.0, 2.0]
    assert mlx_engine._as_float32(plain) is plain  # no dtype: as is

    class _Arr:
        def __init__(self, dtype):
            self.dtype = dtype

        def astype(self, dtype):
            return _Arr(dtype)

    f32 = _Arr("float32")
    assert mlx_engine._as_float32(f32) is f32
    bf16 = _Arr("bfloat16")
    converted = mlx_engine._as_float32(bf16)
    assert converted is not bf16 and converted.dtype == "float32"


def test_model_name_falls_back_when_nothing_is_loaded():
    assert mlx_engine.MLXEngine().model_name == "mlx-engine"
    assert _engine().model_name == "/m"


# ----------------------------------------------------------------------
# Loading: adapter, draft, batching
# ----------------------------------------------------------------------


def test_an_mlx_adapter_folder_reaches_mlx_lm(mlx, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    engine = mlx_engine.MLXEngine()
    engine.load("/models/q", lora_paths=[str(adapter)])
    assert mlx["loads"][0] == ("/models/q", {"adapter_path": str(adapter)})
    assert engine.is_loaded


def test_a_draft_that_fails_to_load_is_ignored(mlx, caplog):
    mlx["fail_load"].add("/drafts/broken")
    engine = mlx_engine.MLXEngine()
    with caplog.at_level("WARNING", logger="hfl.engine.mlx_engine"):
        engine.load("/models/q", draft_model_path="/drafts/broken")
    assert engine._draft is None
    assert "could not be loaded" in caplog.text


def test_a_draft_with_the_same_tokenizer_drives_speculative_decoding(mlx, caplog):
    engine = mlx_engine.MLXEngine()
    engine.load("/models/q", draft_model_path="/drafts/small")
    assert engine._draft == "model:/drafts/small"
    assert engine._prompt_store is None  # a draft runs without the prompt cache
    mlx["responses"] = [
        _Resp("A", 5, from_draft=True),
        _Resp("B", 6),
        _Resp("", 6, finish="stop"),
    ]
    with caplog.at_level("INFO", logger="hfl.engine.mlx_engine"):
        result = engine.generate("hi", GenerationConfig(max_tokens=8))
    assert result.text == "AB"
    assert mlx["stream"][-1]["draft_model"] == "model:/drafts/small"
    assert engine.last_draft_tokens == (1, 2)  # the final response repeats a token
    assert "speculative: 1 of 2 tokens from the draft" in caplog.text


def test_a_batchable_model_gets_a_scheduler_and_unload_closes_it(mlx, monkeypatch):
    made: list[dict] = []

    class _Scheduler(_FakeBatch):
        def __init__(self, model, tokenizer, *, slots, store, model_key):
            super().__init__([])
            made.append({"model": model, "slots": slots, "store": store, "key": model_key})

    monkeypatch.setattr(mlx_batch, "configured_slots", lambda: 3)
    monkeypatch.setattr(mlx_batch, "batchable", lambda model: True)
    monkeypatch.setattr(mlx_batch, "BatchScheduler", _Scheduler)
    engine = mlx_engine.MLXEngine()
    engine.load("/models/q")
    assert made == [{"model": "model:/models/q", "slots": 3, "store": None, "key": "/models/q"}]
    assert engine.supports_concurrent_inference and engine.parallel_slots == 3
    scheduler = engine._batch
    engine.unload()
    assert scheduler.closed and engine._batch is None and engine.parallel_slots == 0
    assert mlx["mx"].cleared == 1  # the freed Metal buffers go back too


def test_unload_without_any_cache_api_still_drops_the_model(mlx, monkeypatch):
    bare = types.ModuleType("mlx.core")  # neither clear_cache nor metal
    monkeypatch.setitem(sys.modules, "mlx.core", bare)
    sys.modules["mlx"].core = bare  # type: ignore[attr-defined]
    engine = _engine()
    engine.unload()
    assert not engine.is_loaded


# ----------------------------------------------------------------------
# The batched paths
# ----------------------------------------------------------------------


def test_batched_generate_reads_its_own_stream_with_its_own_signal(mlx):
    steps = [
        _Resp("Hel", 5, prompt_tokens=3, n=1, reused=2),
        _Resp("lo", 6, prompt_tokens=3, n=2, reused=2, finish="length"),
    ]
    batch = _FakeBatch(steps)
    engine = _engine(_batch=batch)
    signal = threading.Event()
    with cancel.scope(signal):
        result = engine.generate("abc", GenerationConfig(max_tokens=0, temperature=0))
    call = batch.calls[0]
    assert call["tokens"] == [1, 2, 3] and call["max_tokens"] == 2048  # 0 means the default
    assert call["logprobs"] is False and call["cancelled"]() is False
    signal.set()
    assert call["cancelled"]() is True  # its own signal, not the engine's
    assert result.text == "Hello" and result.stop_reason == "length"
    assert engine.last_prompt_tokens_reused == 2 and result.tokens_prompt == 5


def test_batched_generate_without_a_request_signal_is_never_cancelled(mlx):
    batch = _FakeBatch([_Resp("x", 5)])
    _engine(_batch=batch).generate("a", GenerationConfig(max_tokens=4))
    assert batch.calls[0]["cancelled"]() is False


def test_batched_stream_counts_the_reused_prefix(mlx):
    steps = [
        _Resp("ab", 5, prompt_tokens=4, n=1, reused=6),
        _Resp("c", 6, prompt_tokens=4, n=2, reused=6),
    ]
    engine = _engine(_batch=_FakeBatch(steps))
    counted = engine.generate_stream("q", GenerationConfig(max_tokens=4))
    assert "".join(counted) == "abc"
    assert counted.prompt_tokens == 10 and counted.completion_tokens == 2
    assert engine.last_prompt_tokens_reused == 6
    assert not engine._native.locked()  # batched streams take no engine lock


def test_batched_logprobs_read_the_batch(mlx):
    lp = np.log(np.array([0.1, 0.2, 0.7]))
    batch = _FakeBatch([_Resp("x", 2, logprobs=lp, finish="stop")])
    engine = _engine(_batch=batch)
    result = engine.generate("q", GenerationConfig(max_tokens=4, logprobs=1))
    assert batch.calls[0]["logprobs"] is True
    assert result.logprobs[0]["logprob"] == pytest.approx(np.log(0.7), rel=1e-5)
    assert [t["token"] for t in result.logprobs[0]["top_logprobs"]] == ["<2>"]


# ----------------------------------------------------------------------
# Logprobs one at a time
# ----------------------------------------------------------------------


def test_logprobs_on_the_plain_path_normalise_and_rank(mlx):
    mlx["responses"] = [
        _Resp("Hi", 1, logprobs=np.array([2.0, 1.0, 0.0]), prompt_tokens=5),
        _Resp(" there", 0, logprobs=np.array([0.0, 0.0, 0.0]), prompt_tokens=5),  # EOS
    ]
    engine = _engine()
    result = engine.generate("q", GenerationConfig(max_tokens=8, logprobs=2))
    assert result.text == "Hi" and result.stop_reason == "stop"
    assert result.tokens_generated == 1 and result.tokens_prompt == 5
    entry = result.logprobs[0]
    probs = np.exp([2.0, 1.0, 0.0]) / np.exp([2.0, 1.0, 0.0]).sum()
    assert entry["token"] == "<1>" and entry["bytes"] == list(b"<1>")
    assert entry["logprob"] == pytest.approx(np.log(probs[1]))
    assert [t["token"] for t in entry["top_logprobs"]] == ["<0>", "<1>"]


def test_logprobs_stop_at_a_stop_string(mlx):
    mlx["responses"] = [
        _Resp("ab", 3, logprobs=np.zeros(4)),
        _Resp("cEND", 2, logprobs=np.zeros(4)),
        _Resp("never", 1, logprobs=np.zeros(4)),
    ]
    result = _engine().generate("q", GenerationConfig(max_tokens=8, logprobs=0, stop="END"))
    assert result.text == "abc" and result.stop_reason == "stop"
    assert all(e["top_logprobs"] == [] for e in result.logprobs)


def test_logprobs_run_to_the_length_limit(mlx):
    mlx["responses"] = [_Resp("a", 3, logprobs=np.zeros(4)), _Resp("b", 2, logprobs=np.zeros(4))]
    result = _engine().generate("q", GenerationConfig(max_tokens=2, logprobs=1))
    assert result.stop_reason == "length" and result.tokens_generated == 2


def test_logprobs_through_the_prompt_cache(mlx, monkeypatch):
    class _Store:
        def __init__(self):
            self.inserted = []

        def fetch_nearest_cache(self, key, tokens):
            return None, tokens

        def insert_cache(self, key, tokens, cache):
            self.inserted.append(tokens)

    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model: "KV"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm.models", types.ModuleType("mlx_lm.models"))
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    store = _Store()
    mlx["responses"] = [_Resp("a", 3, logprobs=np.zeros(4), finish="stop")]
    result = _engine(_prompt_store=store).generate("xy", GenerationConfig(logprobs=1))
    assert result.text == "a" and store.inserted == [[1, 2, 3]]
    assert mlx["stream"][-1]["prompt_cache"] == "KV"


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def test_the_keyed_sampler_applies_top_p_top_k_and_its_own_seed(mlx):
    cfg = GenerationConfig(temperature=0.5, top_k=3, top_p=0.9, seed=11)
    sample = mlx_engine.MLXEngine._keyed_sampler(cfg)
    assert sample(_L("logits")) == 7
    assert sample(_L("logits")) == 7
    (first, key1), (_second, key2) = mlx["mx"].draws
    # top_p first, then top_k, then scaled by 1/temperature.
    assert first == ("scaled", ("top_k", 3, _L(("top_p", 0.9, _L("logits")))), 2.0)
    assert key1 == ("draw", ("key", 11)) and key2 == ("draw", ("next", ("key", 11)))


def test_the_keyed_sampler_skips_filters_that_are_off(mlx):
    sample = mlx_engine.MLXEngine._keyed_sampler(
        GenerationConfig(temperature=1.0, top_k=0, top_p=1.0)
    )
    sample(_L("L"))
    assert mlx["mx"].draws[0][0] == ("scaled", "L", 1.0)


def test_a_response_format_adds_the_constraint_processor(mlx, monkeypatch):
    import hfl.engine.constrained as constrained

    monkeypatch.setattr(constrained, "mlx_processor", lambda tok, fmt: ("constraint", fmt))
    fmt = {"type": "json_object"}
    kwargs = _engine()._build_sampling(GenerationConfig(response_format=fmt))
    assert kwargs["logits_processors"] == [("constraint", fmt)]


# ----------------------------------------------------------------------
# Plain generate: failures and counts
# ----------------------------------------------------------------------


def test_a_failing_stream_is_logged_and_raised(mlx, caplog):
    def _boom():
        raise RuntimeError("metal fault")
        yield

    mlx["responses"] = _boom()
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="metal fault"):
        _engine().generate("q", GenerationConfig(max_tokens=4))
    assert "MLX generate failed" in caplog.text


def test_token_counts_are_zero_when_the_tokenizer_cannot_count(mlx):
    class _Broken(_Tok):
        def encode(self, text):
            raise ValueError("no")

    engine = _engine(_tokenizer=_Broken())
    result = engine.generate("q", GenerationConfig(max_tokens=10, stop=["zzz"]))
    assert result.text == "ABC"  # the stop string never occurs: nothing cut
    assert result.tokens_prompt == 0 and result.tokens_generated == 0


def test_generate_and_count_refuse_without_a_model():
    engine = mlx_engine.MLXEngine()
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.generate_stream("q")
    with pytest.raises(RuntimeError, match="no model loaded"):
        engine.count_prompt_tokens([ChatMessage(role="user", content="hi")])


def test_count_prompt_tokens_renders_as_chat_does(mlx):
    engine = _engine()
    n = engine.count_prompt_tokens([ChatMessage(role="user", content="hi")])
    assert n == len("user: hi\nassistant:")  # the manual render, one token per char


# ----------------------------------------------------------------------
# The cached path's edges
# ----------------------------------------------------------------------


class _Plain:
    """An iterator without ``close`` (the cached path must not need one)."""

    def __init__(self, items):
        self._it = iter(items)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


def test_cached_generate_with_a_closeless_stream_and_no_stop_hit(mlx):
    class _Batch(_FakeBatch):
        def stream(self, tokens, **kwargs):
            self.calls.append(kwargs)
            return _Plain(self.steps)

    engine = _engine(_batch=_Batch([_Resp("ab", 5, n=1), _Resp("c", 6, n=2, finish="stop")]))
    result = engine.generate("q", GenerationConfig(max_tokens=4, stop=["zz"]))
    assert result.text == "abc" and result.stop_reason == "stop"


def test_a_cached_stream_with_a_closeless_iterator_and_no_counts(mlx):
    class _Batch(_FakeBatch):
        def stream(self, tokens, **kwargs):
            return _Plain(["raw", "text"])  # neither ``text`` nor counts

    engine = _engine(_batch=_Batch([]))
    counted = engine.generate_stream("q")
    assert list(counted) == ["raw", "text"]
    assert counted.prompt_tokens is None  # nothing measured: no number made up


# ----------------------------------------------------------------------
# Prompt rendering
# ----------------------------------------------------------------------


class _TemplTok(_Tok):
    def __init__(self, behaviour):
        self.chat_template = "{{ messages }}"
        self.behaviour = behaviour
        self.calls: list = []

    def apply_chat_template(self, dicts, **kwargs):
        self.calls.append((dicts, kwargs))
        return self.behaviour(dicts, kwargs)


@pytest.fixture
def templates(monkeypatch):
    """Decide what the template probe says, without jinja2."""
    import hfl.engine.llama_cpp as llama_cpp

    probe = {"tools": True, "system": True}
    monkeypatch.setattr(llama_cpp, "_template_renders_tools", lambda t, f: probe["tools"])
    monkeypatch.setattr(llama_cpp, "_template_takes_system", lambda t: probe["system"])
    return probe


def test_rendering_needs_a_tokenizer():
    with pytest.raises(RuntimeError, match="tokenizer not loaded"):
        mlx_engine.MLXEngine()._messages_to_prompt([])


def test_names_ids_and_calls_reach_the_template(templates):
    tok = _TemplTok(lambda d, k: "OK")
    engine = _engine(_tokenizer=tok)
    msgs = [
        ChatMessage(role="system", content="be brief"),
        ChatMessage(role="assistant", content="", tool_calls=[{"function": {"name": "f"}}]),
        ChatMessage(role="tool", content="42", name="f", tool_call_id="call_1"),
    ]
    assert engine._messages_to_prompt(msgs, tools=None) == "OK"
    dicts, kwargs = tok.calls[0]
    assert "tools" not in kwargs  # a template that renders tools, but none asked
    assert dicts[1]["tool_calls"] == [{"function": {"name": "f"}}]
    assert dicts[2]["name"] == "f" and dicts[2]["tool_call_id"] == "call_1"
    assert dicts[0]["role"] == "system"


def test_a_template_without_a_system_role_gets_it_folded(templates):
    templates["system"] = False
    tok = _TemplTok(lambda d, k: "OK")
    engine = _engine(_tokenizer=tok)
    engine._messages_to_prompt(
        [ChatMessage(role="system", content="S"), ChatMessage(role="user", content="U")]
    )
    dicts, _ = tok.calls[0]
    assert dicts == [{"role": "user", "content": "S\n\nU"}]


def test_an_old_tokenizer_without_extra_variables_still_renders(templates):
    def behaviour(dicts, kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword")
        return "OLD"

    tok = _TemplTok(behaviour)
    engine = _engine(_tokenizer=tok)
    out = engine._messages_to_prompt([ChatMessage(role="user", content="u")], reasoning="off")
    assert out == "OLD" and len(tok.calls) == 2


def test_a_template_that_fails_twice_falls_back_to_role_tags(templates):
    def behaviour(dicts, kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword")
        raise ValueError("broken template")

    engine = _engine(_tokenizer=_TemplTok(behaviour))
    out = engine._messages_to_prompt([ChatMessage(role="user", content="u")], reasoning="off")
    assert out == "user: u\nassistant:"


def test_a_template_that_raises_falls_back_to_role_tags(templates):
    def behaviour(dicts, kwargs):
        raise ValueError("broken template")

    engine = _engine(_tokenizer=_TemplTok(behaviour))
    assert engine._messages_to_prompt([ChatMessage(role="user", content="u")]) == (
        "user: u\nassistant:"
    )


def test_a_tokenizer_without_apply_renders_role_tags(templates):
    templates["tools"] = False
    engine = _engine()  # _Tok has no apply_chat_template
    assert engine._messages_to_prompt([ChatMessage(role="user", content="u")]) == (
        "user: u\nassistant:"
    )


# ----------------------------------------------------------------------
# Stop strings while streaming
# ----------------------------------------------------------------------


def test_stream_until_stop_holds_back_a_possible_stop_and_flushes_the_tail():
    engine = mlx_engine.MLXEngine()
    pieces = list(engine._stream_until_stop(iter(["ab", "cd", "e"]), ["XYZ"], lambda t: t))
    assert "".join(pieces) == "abcde"
    assert pieces[-1] == "de"  # the held-back tail, flushed at the end


def test_stream_until_stop_at_a_boundary_yields_nothing_more():
    engine = mlx_engine.MLXEngine()
    # "ab" goes out first (safe), then the stop starts exactly there.
    pieces = list(engine._stream_until_stop(iter(["abcd", "X"]), ["cdX"], lambda t: t))
    assert pieces == ["ab"]


# ======================================================================
# mlx_batch
# ======================================================================


EOS = 0


@dataclass
class _BResp:
    uid: int
    token: int
    logprobs: object = None
    finish_reason: str | None = None
    prompt_cache: object = None
    all_tokens: list[int] = field(default_factory=list)


class _Detok:
    def __init__(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = f"<{token}>"

    def finalize(self):
        self.last_segment = ""


class _BTok:
    eos_token_ids = [EOS]

    @property
    def detokenizer(self):
        return _Detok()


class _Gen:
    """A scripted BatchGenerator: ``script`` is a list of response lists."""

    instances: list[_Gen] = []
    created = threading.Event()
    gate: threading.Event | None = None
    fail_insert = False
    fail_close = False

    def __init__(self, model, stop_tokens=None, completion_batch_size=4, prefill_batch_size=4):
        if _Gen.gate is not None:
            _Gen.gate.wait(10)
        self.stop_tokens = stop_tokens
        self.script: list[list[_BResp]] = []
        self.removed: list[int] = []
        self.inserted: list[dict] = []
        self.uid = 0
        _Gen.instances.append(self)
        _Gen.created.set()

    def insert(self, prompts, max_tokens, caches=None, all_tokens=None, samplers=None,
               logits_processors=None):  # fmt: skip
        if _Gen.fail_insert:
            raise ValueError("prompt too long")
        self.inserted.append({"prompts": prompts, "caches": caches, "all_tokens": all_tokens})
        uid, self.uid = self.uid, self.uid + 1
        return [uid]

    def next(self):
        return [], (self.script.pop(0) if self.script else [])

    def remove(self, uids):
        self.removed.extend(uids)

    def close(self):
        if _Gen.fail_close:
            raise RuntimeError("close failed")


@pytest.fixture
def batch_mlx(monkeypatch):
    _Gen.instances.clear()
    _Gen.created = threading.Event()
    _Gen.gate = None
    _Gen.fail_insert = False
    _Gen.fail_close = False
    gen_mod = types.ModuleType("mlx_lm.generate")
    gen_mod.BatchGenerator = _Gen  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", gen_mod)
    core = _FakeMx()
    pkg = types.ModuleType("mlx")
    pkg.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    yield core
    _Gen.gate = None


def _cache_module(monkeypatch, make):
    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm.models", types.ModuleType("mlx_lm.models"))
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)


def test_batchable_when_every_cache_layer_merges(batch_mlx, monkeypatch):
    class _Layer:
        def merge(self):
            pass

    _cache_module(monkeypatch, lambda model: [_Layer(), _Layer()])
    assert mlx_batch.batchable("model") is True
    _cache_module(monkeypatch, lambda model: [_Layer(), object()])
    assert mlx_batch.batchable("model") is False


def test_batchable_is_false_when_the_cache_cannot_be_built(batch_mlx, monkeypatch):
    def _boom(model):
        raise RuntimeError("unsupported architecture")

    _cache_module(monkeypatch, _boom)
    assert mlx_batch.batchable("model") is False


def test_batchable_is_false_without_batching_in_mlx_lm(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", None)
    assert mlx_batch.batchable("model") is False


def test_float32_converts_and_evaluates_on_the_batch_thread(batch_mlx):
    class _Arr:
        def astype(self, dtype):
            return ("converted", dtype)

    assert mlx_batch._float32([1.0]) == [1.0]  # nothing to convert
    assert mlx_batch._float32(_Arr()) == ("converted", "float32")
    assert batch_mlx.evaluated == [("converted", "float32")]


def _sched(store=None) -> mlx_batch.BatchScheduler:
    return mlx_batch.BatchScheduler("model", _BTok(), slots=2, store=store)


def test_a_failed_insert_reaches_the_caller_and_the_scheduler_lives_on(batch_mlx):
    scheduler = _sched()
    _Gen.fail_insert = True
    with pytest.raises(ValueError, match="prompt too long"):
        next(scheduler.stream([1], max_tokens=2, sampler=None, processors=[]))
    _Gen.fail_insert = False
    gen = _Gen.instances[0]
    gen.script = [[_BResp(0, 5)], [_BResp(0, 6, finish_reason="length")]]
    steps = list(scheduler.stream([1], max_tokens=2, sampler=None, processors=[]))
    scheduler.close()
    assert [s.token for s in steps] == [5, 6]
    assert gen.stop_tokens == [[EOS]]


def test_close_twice_is_harmless_and_a_failing_close_is_tolerated(batch_mlx):
    _Gen.fail_close = True
    scheduler = _sched()
    scheduler.close()
    assert scheduler._stopped.is_set()
    scheduler.close()  # already stopped: returns at once


def test_responses_for_removed_requests_and_unknown_commands_are_ignored(batch_mlx):
    scheduler = _sched()
    scheduler._commands.put(("noop",))  # neither an insert nor a remove
    scheduler._commands.put(("remove", 99))  # not active
    gen_ready = scheduler.stream([1, 2], max_tokens=4, sampler=None, processors=[])
    # The generator answers for a uid nobody holds, then ends the real one.
    assert _Gen.created.wait(10)
    gen = _Gen.instances[0]
    gen.script = [
        [_BResp(42, 9), _BResp(0, 5, logprobs="LP")],
        [_BResp(0, 6, finish_reason="stop")],
    ]
    steps = list(gen_ready)
    scheduler.close()
    assert [s.token for s in steps] == [5, 6] and gen.removed == []
    assert steps[0].logprobs is None  # this request did not ask for them


def test_logprobs_requests_get_float32_distributions(batch_mlx):
    class _Arr:
        def astype(self, dtype):
            return "f32-array"

    scheduler = _sched()
    stream = scheduler.stream([1], max_tokens=4, sampler=None, processors=[], logprobs=True)
    assert _Gen.created.wait(10)
    _Gen.instances[0].script = [[_BResp(0, 5, logprobs=_Arr(), finish_reason="stop")]]
    steps = list(stream)
    scheduler.close()
    assert steps[0].logprobs == "f32-array"


def test_an_exact_cache_hit_evaluates_the_prompt_afresh(batch_mlx):
    class _Store:
        def fetch_nearest_cache(self, key, tokens):
            return "CACHE", []  # the whole prompt cached: nothing left

        def insert_cache(self, key, tokens, cache):
            raise RuntimeError("store full")  # a failed insert is only logged

    scheduler = _sched(_Store())
    stream = scheduler.stream([1, 2, 3], max_tokens=2, sampler=None, processors=[])
    assert _Gen.created.wait(10)
    gen = _Gen.instances[0]
    gen.script = [[_BResp(0, 5, finish_reason="stop", prompt_cache="KV", all_tokens=[1, 2, 3])]]
    steps = list(stream)
    scheduler.close()
    assert gen.inserted[0] == {"prompts": [[1, 2, 3]], "caches": None, "all_tokens": None}
    assert steps[-1].reused == 0 and steps[-1].prompt_tokens == 3


def test_a_request_that_arrives_after_the_stop_is_refused(batch_mlx):
    _Gen.gate = threading.Event()  # the batch thread waits in its generator
    scheduler = _sched()
    late = mlx_batch._Insert([1], 2, None, [], False)
    scheduler._commands.put(("stop",))
    scheduler._commands.put(late)
    _Gen.gate.set()
    scheduler._thread.join(10)
    reply = late.reply.get(timeout=5)
    assert isinstance(reply, RuntimeError) and "unloaded" in str(reply)


def test_a_cancelled_request_leaves_while_waiting_for_its_first_token(batch_mlx):
    scheduler = _sched()
    calls = {"n": 0}

    def cancelled():
        calls["n"] += 1
        return calls["n"] >= 2  # waits once, then gives up

    steps = list(
        scheduler.stream([1], max_tokens=4, sampler=None, processors=[], cancelled=cancelled)
    )
    scheduler.close()
    assert steps == [] and calls["n"] == 2
    assert _Gen.instances[0].removed == [0]

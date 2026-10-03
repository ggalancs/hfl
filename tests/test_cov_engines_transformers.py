# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The Transformers engine against a stand-in ``transformers`` whose
``StoppingCriteria`` is a real class, so the engine's own criteria — the
cancel watch and the first-token clock — are built and called: the
blocking generate with measured timings, cancellation, structured output,
the streaming worker, quantization without CUDA, a load that fails half
way, the MPS release on unload and the prompt fallbacks.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import types

import pytest

from hfl.engine.base import ChatMessage, GenerationConfig


class _Ids:
    def __init__(self, n):
        self.shape = (1, n)


class _Inputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class _Tokenizer:
    """Whitespace tokens; decode joins them back."""

    def __call__(self, prompt, return_tensors=None):
        n = len(prompt.split())
        return _Inputs(input_ids=_Ids(n) if return_tensors else list(range(n)))

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(f"t{t}" for t in tokens)


def _transformers(monkeypatch, *, cuda=False, mps=lambda: False):
    seen: dict = {}
    tf = types.ModuleType("transformers")

    class StoppingCriteria:
        pass

    class StoppingCriteriaList(list):
        pass

    class LogitsProcessorList(list):
        pass

    class TextIteratorStreamer:
        def __init__(self, tokenizer, skip_prompt, skip_special_tokens):
            self.q: queue.Queue = queue.Queue()

        def end(self):
            self.q.put(None)

        def __iter__(self):
            while (item := self.q.get(timeout=10)) is not None:
                yield item

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            return _Tokenizer()

    class AutoModelForCausalLM:
        model = None

        @staticmethod
        def from_pretrained(path, **kwargs):
            seen["load_kwargs"] = kwargs
            return AutoModelForCausalLM.model or types.SimpleNamespace(device="cpu")

    for cls in (
        StoppingCriteria,
        StoppingCriteriaList,
        LogitsProcessorList,
        TextIteratorStreamer,
        BitsAndBytesConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
    ):
        setattr(tf, cls.__name__, cls)

    calls: list[str] = []

    def mps_available():
        return mps()

    torch = types.ModuleType("torch")
    torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda,
        empty_cache=lambda: calls.append("cuda.empty_cache"),
        synchronize=lambda: calls.append("cuda.synchronize"),
    )
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=mps_available)
    )
    torch.mps = types.SimpleNamespace(  # type: ignore[attr-defined]
        empty_cache=lambda: calls.append("mps.empty_cache")
    )
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", torch)
    seen["torch_calls"] = calls
    seen["tf"] = tf
    return seen


class _Model:
    """``generate`` that consults the stopping criteria like transformers."""

    device = "cpu"

    def __init__(self, new_tokens=3, on_step=None):
        self.new_tokens = new_tokens
        self.on_step = on_step
        self.kwargs: dict = {}
        self.stopped: list[bool] = []
        self.generation_config = types.SimpleNamespace(eos_token_id=[2, 3])

    def generate(self, **kwargs):
        self.kwargs = kwargs
        prompt_n = kwargs["input_ids"].shape[1] if "input_ids" in kwargs else 0
        criteria = kwargs.get("stopping_criteria") or []
        produced = []
        for i in range(self.new_tokens):
            if self.on_step is not None:
                self.on_step(i)
            produced.append(100 + i)
            if any(c(None, None) for c in criteria):
                self.stopped.append(True)
                break
        streamer = kwargs.get("streamer")
        if streamer is not None:
            for piece in ["Hel", "", "lo"]:
                streamer.q.put(piece)
            streamer.end()
            return None
        return [list(range(prompt_n)) + produced]


def _engine(model):
    from hfl.engine.transformers_engine import TransformersEngine

    engine = TransformersEngine()
    engine._model, engine._tokenizer, engine._model_id = model, _Tokenizer(), "org/m"
    return engine


# ----------------------------------------------------------------------
# Load / unload
# ----------------------------------------------------------------------


def test_quantization_without_cuda_loads_in_full_precision(monkeypatch, caplog):
    from hfl.engine.transformers_engine import TransformersEngine

    seen = _transformers(monkeypatch, cuda=False)
    engine = TransformersEngine()
    with caplog.at_level("WARNING", logger="hfl.engine.transformers_engine"):
        engine.load("org/m", quantization="4bit")
    assert "quantization_config" not in seen["load_kwargs"]
    assert "needs an NVIDIA GPU" in caplog.text and engine.is_loaded


def test_a_load_that_fails_after_the_model_exists_leaves_nothing(monkeypatch):
    from hfl.engine.transformers_engine import TransformersEngine

    seen = _transformers(monkeypatch, cuda=True)

    class _Broken:
        @property
        def device(self):
            raise RuntimeError("device lost")

    seen["tf"].AutoModelForCausalLM.model = _Broken()
    engine = TransformersEngine()
    with pytest.raises(RuntimeError, match="device lost"):
        engine.load("org/m", quantization="8bit")
    assert seen["torch_calls"] == ["cuda.empty_cache"]  # the cleanup ran
    assert seen["load_kwargs"]["quantization_config"].kwargs == {"load_in_8bit": True}


def test_a_failed_load_leaves_the_engine_unloaded(monkeypatch):
    from hfl.engine.transformers_engine import TransformersEngine

    seen = _transformers(monkeypatch, cuda=True)

    class _Broken:
        @property
        def device(self):
            raise RuntimeError("device lost")

    seen["tf"].AutoModelForCausalLM.model = _Broken()
    engine = TransformersEngine()
    with pytest.raises(RuntimeError, match="device lost"):
        engine.load("org/m")
    assert not engine.is_loaded


def test_unload_releases_the_mps_cache_only_when_mps_is_there(monkeypatch):
    seen = _transformers(monkeypatch, cuda=False, mps=lambda: False)
    engine = _engine(_Model())
    engine.unload()
    assert seen["torch_calls"] == [] and not engine.is_loaded


def test_an_mps_probe_that_fails_does_not_fail_unload(monkeypatch):
    def broken():
        raise AttributeError("no mps in this build")

    _transformers(monkeypatch, mps=broken)
    engine = _engine(_Model())
    engine.unload()
    assert not engine.is_loaded


# ----------------------------------------------------------------------
# Blocking generation
# ----------------------------------------------------------------------


def test_generate_measures_prefill_and_decodes_only_new_tokens(monkeypatch):
    _transformers(monkeypatch)
    model = _Model(new_tokens=3)
    engine = _engine(model)
    result = engine.generate("one two", GenerationConfig(max_tokens=3, temperature=0.0))
    assert result.text == "t100 t101 t102"
    assert result.tokens_prompt == 2 and result.tokens_generated == 3
    assert result.stop_reason == "length"  # every max_new_tokens used
    assert model.kwargs["do_sample"] is False and model.kwargs["temperature"] is None
    assert model.kwargs["max_new_tokens"] == 3
    # Prefill is measured up to the first token, not apportioned.
    assert 0 <= result.prompt_eval_duration <= result.total_duration
    assert result.eval_duration >= 0


def test_cancel_stops_a_blocking_generation_at_the_next_token(monkeypatch):
    _transformers(monkeypatch)
    holder: dict = {}

    def on_step(i):
        if i == 1 and not holder.get("done"):
            holder["done"] = True
            holder["engine"].cancel()

    model = _Model(new_tokens=10, on_step=on_step)
    engine = _engine(model)
    holder["engine"] = engine
    result = engine.generate("p", GenerationConfig(max_tokens=10, temperature=0.7))
    assert result.tokens_generated == 2 and result.stop_reason == "stop"
    assert model.stopped == [True] and model.kwargs["do_sample"] is True
    # The next request starts clean.
    assert engine.generate("p", GenerationConfig(max_tokens=10)).tokens_generated == 10


def test_a_response_format_constrains_generation(monkeypatch):
    import hfl.engine.constrained as constrained

    _transformers(monkeypatch)
    made: list = []
    monkeypatch.setattr(
        constrained,
        "torch_processor",
        lambda tok, fmt, eos: made.append((fmt, eos)) or "PROCESSOR",
    )
    model = _Model(new_tokens=1)
    engine = _engine(model)
    fmt = {"type": "json_object"}
    engine.generate("p", GenerationConfig(max_tokens=4, response_format=fmt))
    assert made == [(fmt, [2, 3])]
    assert list(model.kwargs["logits_processor"]) == ["PROCESSOR"]
    model.generation_config.eos_token_id = 7  # a single id
    assert engine._constraint(GenerationConfig(response_format=fmt))
    assert made[-1] == (fmt, [7])
    assert engine._constraint(GenerationConfig()) == {}


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------


def test_stream_skips_empty_pieces_and_offers_a_cancel_criterion(monkeypatch):
    _transformers(monkeypatch)
    model = _Model(new_tokens=1)
    engine = _engine(model)
    pieces = list(
        engine.chat_stream([ChatMessage(role="user", content="hi")], GenerationConfig(max_tokens=2))
    )
    assert pieces == ["Hel", "lo"]
    (criterion,) = model.kwargs["stopping_criteria"]
    assert criterion(None, None) is True  # the stream is over: generation must stop
    assert model.kwargs["top_k"] == GenerationConfig().top_k


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------


class _TemplateTokenizer(_Tokenizer):
    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, msgs, **kwargs):
        self.calls.append(kwargs)
        if "tools" in kwargs or "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument")
        return f"PROMPT({len(msgs)}):{msgs[-1].get('tool_call_id')}"


def test_an_old_template_without_tools_or_reasoning_still_renders(monkeypatch):
    _transformers(monkeypatch)
    engine = _engine(_Model())
    engine._tokenizer = _TemplateTokenizer()
    msgs = [
        ChatMessage(role="user", content="w?"),
        ChatMessage(role="tool", content="31C", name="weather", tool_call_id="call_9"),
    ]
    tools = [{"type": "function", "function": {"name": "weather"}}]
    prompt = engine._build_prompt(msgs, tools=tools, reasoning="off")
    assert prompt == "PROMPT(2):call_9"
    first, second = engine._tokenizer.calls
    assert first["tools"] == tools and "enable_thinking" in first
    assert "tools" not in second and "enable_thinking" not in second


def test_the_manual_prompt_keeps_tool_results_and_skips_unknown_roles(monkeypatch):
    _transformers(monkeypatch)
    engine = _engine(_Model())  # _Tokenizer has no chat template
    msgs = [
        ChatMessage(role="user", content="w?"),
        ChatMessage(role="tool", content="31C", name="weather"),
        ChatMessage(role="developer", content="ignored"),
    ]
    assert engine._build_prompt(msgs) == "[INST] w? [/INST]\n[TOOL weather] 31C"


def test_count_prompt_tokens_needs_a_model_and_counts_the_rendered_prompt(monkeypatch):
    from hfl.engine.transformers_engine import TransformersEngine

    with pytest.raises(RuntimeError, match="no model loaded"):
        TransformersEngine().count_prompt_tokens([ChatMessage(role="user", content="hi")])
    _transformers(monkeypatch)
    engine = _engine(_Model())
    n = engine.count_prompt_tokens(
        [ChatMessage(role="user", content="a b c")], GenerationConfig(reasoning="off")
    )
    assert n == len("[INST] a b c [/INST]".split())

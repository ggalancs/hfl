# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The KV prefix reuse that makes multi-turn chat cheap, and its guard.

This file exists because of a near-miss. The roadmap called for building
a prefix cache, on the reasoning that "every chat turn re-processes the
system prompt". Measuring first showed that is false for the llama.cpp
backend: ``llama-cpp-python`` matches the prompt against the KV cache
inside ``Llama.generate()``, and HFL never resets or rebuilds the model
between requests, so the reuse is already live. Measured through HFL's
own engine on SmolLM2-135M with a ~1 500-token system prompt::

    turn   prompt tok   prefill ms
       1         1522         74.8
       5         1626          7.8

Turn 5 sends *more* tokens and pays 9.6x less, because only the new
suffix is evaluated. So there was nothing to build — but there was
something to protect. A 10x win nobody asserts is a 10x win that a
future refactor deletes in silence: one ``self._model.reset()``, or
constructing ``Llama`` per request, and every turn pays full prefill
again with no test going red.

Hence two guards, deliberately different in kind:

* a **structural** one that runs everywhere, including the CI venv that
  has no ``llama_cpp`` and no model files, and
* a **measured** one that runs only where a real model exists, and
  proves the effect rather than the code shape.

Neither subsumes the other. The structural guard would pass if
llama-cpp-python changed its own behaviour upstream; the measured guard
would not. The measured guard cannot run in CI; the structural one can.

**Known gap, measured not assumed:** the MLX engine has no cache handling
at all — ``mlx_lm.generate()`` is called with the whole prompt every turn.
On Apple Silicon, where MLX is the faster backend, multi-turn chat pays
full prefill each time. ``mlx_lm.models.cache`` ships ``LRUPromptCache``,
so this is wirable; it is a separate piece of work, tracked in the
roadmap rather than pretended away here.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "hfl" / "engine" / "llama_cpp.py"

SMOL = Path.home() / ".hfl/models/bartowski--SmolLM2-135M-Instruct-GGUF"
SMOL_GGUF = SMOL / "SmolLM2-135M-Instruct-Q4_K_M.gguf"

# Both preconditions, not one. ``~/.hfl`` belongs to the machine, so the
# CI venv sees the model file while having no ``[llama]`` extra — and the
# engine answers a missing backend with RuntimeError, not ImportError, so
# ``importorskip`` inside the fixture does not catch it either.
HAVE_LLAMA_CPP = importlib.util.find_spec("llama_cpp") is not None
CAN_MEASURE = SMOL_GGUF.exists() and HAVE_LLAMA_CPP


class TestStructuralGuard:
    """Runs anywhere. Catches the realistic ways the reuse gets deleted."""

    @staticmethod
    def _tree() -> ast.Module:
        return ast.parse(ENGINE.read_text(encoding="utf-8"))

    def test_the_primary_model_is_never_reset(self):
        """``Llama.reset()`` clears ``n_tokens``, which is the state the
        prefix match is made against. Calling it on ``self._model`` turns
        every turn back into a cold prefill.

        The draft model is a separate object with its own KV and IS reset
        deliberately (speculative decoding rewinds it), so the assertion
        is about the receiver, not about the method name.
        """
        offenders = []
        for node in ast.walk(self._tree()):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "reset":
                continue
            recv = func.value
            # self._model.reset() / self._model._model.reset()
            if isinstance(recv, ast.Attribute) and recv.attr in {"_model", "model"}:
                inner = recv.value
                if isinstance(inner, ast.Name) and inner.id == "self":
                    offenders.append(node.lineno)

        assert not offenders, (
            "self._model.reset() at line(s) "
            f"{offenders} — that drops the KV cache the prompt is matched "
            "against, so every chat turn pays full prefill again. Measured "
            "cost of losing this: 7.8 ms -> 74.8 ms on a 1500-token prompt."
        )

    def test_the_model_is_built_once_per_load_not_per_request(self):
        """A ``Llama(...)`` constructed inside a request handler would have
        an empty KV every time, which is the same regression wearing a
        different shape."""
        tree = self._tree()
        request_methods = {"chat", "chat_stream", "generate", "generate_stream"}
        offenders = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in request_methods:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    name = getattr(f, "id", None) or getattr(f, "attr", None)
                    if name == "Llama":
                        offenders.append(f"{node.name}:{sub.lineno}")

        assert not offenders, (
            f"Llama(...) constructed inside a request path: {offenders}. "
            "A per-request model starts with an empty KV cache, so prefix "
            "reuse can never hit."
        )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not CAN_MEASURE, reason="needs both the SmolLM2-135M GGUF in ~/.hfl and llama-cpp-python"
)
class TestMeasuredReuse:
    """Runs where a real model exists. Proves the effect, not the shape.

    Skipped in the CI venv by design — it has no ``[llama]`` extra and no
    model files — which is exactly why the structural guard above is not
    optional.
    """

    @staticmethod
    def _engine():
        pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")
        from hfl.engine.llama_cpp import LlamaCppEngine

        eng = LlamaCppEngine()
        eng.load(str(SMOL_GGUF), n_ctx=4096)
        return eng

    def test_a_shared_prefix_is_not_re_evaluated(self):
        from hfl.engine.base import ChatMessage, GenerationConfig

        eng = self._engine()
        cfg = GenerationConfig(max_tokens=8, temperature=0.0)
        system = ("You are a meticulous assistant. " * 250).strip()

        history = [ChatMessage(role="system", content=system)]
        prefills: list[float] = []
        prompt_tokens: list[int] = []

        for turn in range(1, 4):
            history.append(ChatMessage(role="user", content=f"Reply with the number {turn}."))
            result = eng.chat(history, cfg)
            prefills.append(result.prompt_eval_duration / 1e6)
            prompt_tokens.append(result.tokens_prompt)
            history.append(ChatMessage(role="assistant", content=(result.text or "")[:30]))

        assert prompt_tokens[-1] > prompt_tokens[0], (
            "the conversation must actually grow, or this proves nothing"
        )
        # The headline: a longer prompt costing far less than the first one
        # is only possible if the shared prefix was not re-evaluated.
        assert prefills[-1] < prefills[0] * 0.5, (
            f"prefill went {prefills[0]:.1f} ms -> {prefills[-1]:.1f} ms while the "
            f"prompt grew {prompt_tokens[0]} -> {prompt_tokens[-1]} tokens. "
            "Prefix reuse appears to be gone: the KV cache is being dropped "
            "between requests."
        )

    def test_a_diverging_prompt_pays_full_prefill(self):
        """The other direction, so the first test cannot pass by accident.

        If prefills were always small — because the measurement is broken,
        or the model is too tiny to measure — this would pass too. A
        genuinely unrelated prompt must cost what a cold prompt costs.
        """
        from hfl.engine.base import ChatMessage, GenerationConfig

        eng = self._engine()
        cfg = GenerationConfig(max_tokens=8, temperature=0.0)

        first = [ChatMessage(role="system", content=("Alpha context. " * 250).strip())]
        first.append(ChatMessage(role="user", content="Hello."))
        cold = eng.chat(first, cfg).prompt_eval_duration / 1e6

        # Same length, no shared prefix at all.
        second = [ChatMessage(role="system", content=("Omega context. " * 250).strip())]
        second.append(ChatMessage(role="user", content="Hello."))
        diverged = eng.chat(second, cfg).prompt_eval_duration / 1e6

        assert diverged > cold * 0.5, (
            f"a prompt sharing no prefix cost {diverged:.1f} ms against a cold "
            f"{cold:.1f} ms — suspiciously cheap, which means the prefill "
            "measurement is not measuring prefill."
        )


class TestMLXGapIsRecorded:
    """The gap is a fact about the code, so it is asserted like one.

    When somebody wires ``LRUPromptCache`` into the MLX engine this test
    fails, which is the intended signal: come back and replace it with a
    measured guard like the llama.cpp one above.
    """

    def test_mlx_engine_still_has_no_prompt_cache(self):
        mlx = ENGINE.parent / "mlx_engine.py"
        text = mlx.read_text(encoding="utf-8")
        has_cache = any(
            marker in text for marker in ("prompt_cache", "make_prompt_cache", "LRUPromptCache")
        )
        assert not has_cache, (
            "The MLX engine now mentions a prompt cache. Good — but this "
            "test was the record that it did not. Replace it with a measured "
            "reuse test like TestMeasuredReuse, and update the roadmap."
        )


def test_env_does_not_disable_the_reuse():
    """No HFL knob should be able to turn this off by accident.

    Recorded because the reuse is invisible: nothing logs a hit, so a
    switch that silently disabled it would cost 10x with no symptom.
    """
    from hfl.config import HFLConfig

    suspicious = [
        name
        for name in dir(HFLConfig)
        if not name.startswith("_") and ("cache" in name.lower() and "kv" in name.lower())
    ]
    # There is no such knob today; if one appears, it needs a test proving
    # the default keeps the reuse on.
    assert suspicious == [], (
        f"New KV-cache configuration appeared: {suspicious}. Add a test that "
        "the default value preserves prefix reuse before shipping it."
    )


def test_measured_guard_is_not_silently_skipped_on_this_machine():
    """A skip is not a pass.

    On a developer machine that has the model, the measured guard must
    actually run. This surfaces the difference instead of letting a green
    run hide it.
    """
    if os.environ.get("HFL_REQUIRE_MEASURED_REUSE") != "1":
        pytest.skip("set HFL_REQUIRE_MEASURED_REUSE=1 to enforce locally")
    assert SMOL_GGUF.exists(), (
        f"{SMOL_GGUF} is missing, so the measured prefix-reuse guard did not run"
    )
    assert HAVE_LLAMA_CPP, "llama-cpp-python is absent, so the measured guard did not run"

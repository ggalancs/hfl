# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Structured output for engines that sample in Python: MLX, Transformers.

llama.cpp and llama-server constrain sampling with a grammar of their own;
MLX and Transformers had none, so a ``format`` / ``response_format`` got a
400 (it used to be ignored — prose where JSON was asked). With the
``[structured]`` extra, llguidance (the library llama.cpp's own constrained
decoding builds on) turns the request's format into a token mask applied
before every sampling step:

- ``"json"``: a JSON object (as llama.cpp's ``json_object``);
- a dict: that JSON schema;
- ``"GBNF:<grammar>"``: that grammar.

Each engine wraps :class:`Guide` in its own logits-processor shape. A guide
is one request's: it follows the tokens as they are drawn.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

_TOKENIZERS: dict[tuple[int, int], tuple[Any, Any]] = {}
_LOCK = threading.Lock()


# JSON on one line, with a space after ":" and ",": whitespace left free
# (llguidance's default), Qwen2.5-0.5B wrote `"population": 1,` and then
# spaces until the token limit (measured). A bounded whitespace_pattern fixed
# that on llguidance 1.8 only — 1.7, the line vLLM 0.30 requires, ignores it
# (measured, the same runaway) — so no free whitespace at all, which every
# version honours (measured on 1.7.0, 1.7.6, 1.8.0: valid JSON, stopped).
_JSON_OPTIONS: Any = {"whitespace_flexible": False, "item_separator": ", ", "key_separator": ": "}


def available() -> bool:
    """Whether llguidance is installed (the ``[structured]`` extra)."""
    return importlib.util.find_spec("llguidance") is not None


def grammar_for(response_format: Any) -> str:
    """The llguidance grammar of a request's response format (ValueError
    when it is not one HFL knows)."""
    from llguidance import LLMatcher, grammar_from

    if response_format == "json":
        response_format = {"type": "object"}
    if isinstance(response_format, dict):
        return str(LLMatcher.grammar_from_json_schema(response_format, overrides=_JSON_OPTIONS))
    if isinstance(response_format, str) and response_format.startswith("GBNF:"):
        return str(grammar_from("gbnf", response_format[len("GBNF:") :]))
    raise ValueError(f"not a response format: {response_format!r}")


def _ll_tokenizer(hf_tokenizer: Any, n_vocab: int, eos: list[int] | None) -> Any:
    """llguidance's view of the model's tokenizer, built once per tokenizer
    and vocabulary size (building it walks the whole vocabulary)."""
    key = (id(hf_tokenizer), n_vocab)
    with _LOCK:
        cached = _TOKENIZERS.get(key)
        # The entry holds the tokenizer too, so its id cannot be reused by
        # another object while the entry exists.
        if cached is not None and cached[0] is hf_tokenizer:
            return cached[1]
        from llguidance.hf import from_tokenizer

        built = from_tokenizer(hf_tokenizer, n_vocab=n_vocab, eos_token=eos or None)
        _TOKENIZERS[key] = (hf_tokenizer, built)
        return built


class Guide:
    """One request's constraint: :meth:`mask` gives the tokens the format
    allows next, after the ones drawn so far."""

    def __init__(
        self, hf_tokenizer: Any, response_format: Any, eos: list[int] | None = None
    ) -> None:
        self._hf, self._grammar, self._eos = hf_tokenizer, grammar_for(response_format), eos
        self._matcher: Any = None
        self._bitmask: Any = None

    def mask(self, last_token: int | None, n_vocab: int) -> Any:
        """The numpy bitmask for the next token; ``last_token`` is the one
        just drawn (None at the first step, after the prompt). None once the
        format has failed — nothing sensible left to allow."""
        import numpy as np
        from llguidance import LLMatcher
        from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask

        if self._matcher is None:
            tokenizer = _ll_tokenizer(self._hf, n_vocab, self._eos)
            self._matcher = LLMatcher(tokenizer, self._grammar, log_level=0)
            if self._matcher.is_error():
                raise ValueError(self._matcher.get_error())
            self._bitmask = allocate_token_bitmask(1, n_vocab)
        elif last_token is not None:
            self._matcher.consume_token(int(last_token))
        if self._matcher.is_error():
            return None  # past the end (a token after EOS): leave the logits be
        fill_next_token_bitmask(self._matcher, self._bitmask, 0)
        return np.asarray(self._bitmask)


def mlx_processor(tokenizer: Any, response_format: Any) -> Any:
    """An mlx-lm logits processor: ``(tokens, logits) -> logits``. mlx-lm
    calls it once after the prompt, then once per drawn token (the last in
    ``tokens``)."""
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    eos = [int(t) for t in getattr(tokenizer, "eos_token_ids", []) or []]
    guide = Guide(hf, response_format, eos)
    started = [False]

    def process(tokens: Any, logits: Any) -> Any:
        from llguidance.mlx import apply_token_bitmask

        last = int(tokens[-1].item()) if started[0] else None
        started[0] = True
        mask = guide.mask(last, int(logits.shape[-1]))
        return logits if mask is None else apply_token_bitmask(logits, mask)

    return process


def torch_processor(tokenizer: Any, response_format: Any, eos: list[int] | None) -> Any:
    """A Transformers ``LogitsProcessor`` for one request (batch of one)."""
    from transformers import LogitsProcessor

    guide = Guide(tokenizer, response_format, eos)

    class _Guided(LogitsProcessor):
        def __init__(self) -> None:
            self.started = False

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            import torch
            from llguidance.torch import apply_token_bitmask_inplace

            last = int(input_ids[0, -1]) if self.started else None
            self.started = True
            mask = guide.mask(last, int(scores.shape[-1]))
            if mask is not None:
                apply_token_bitmask_inplace(scores, torch.from_numpy(mask).to(scores.device))
            return scores

    return _Guided()

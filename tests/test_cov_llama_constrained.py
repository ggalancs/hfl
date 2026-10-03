# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl.engine.constrained`` against a stand-in llguidance (and a stand-in
torch / transformers for the Transformers processor), so the guide's
bookkeeping — the tokenizer cache, first-step mask, consumption, rollback
under speculative decoding, the stop at the end of the format — runs in
the CI venv, which has none of the three.

The stand-in matcher allows exactly one token next: the one whose id is
the number of tokens consumed so far (0, then 1, then 2 ...). It stops
after ``STOP_AFTER`` tokens; a token consumed after that puts it in error,
as llguidance does with a token past the end of the format.
"""

from __future__ import annotations

import importlib.machinery
import json
import sys
import types

import numpy as np
import pytest

STOP_AFTER = 3


class _FakeMatcher:
    def __init__(self, tokenizer, grammar, log_level=0):
        self.tokenizer, self.grammar = tokenizer, grammar
        self.consumed: list[int] = []
        self.rolled_back = 0
        self.error = "bad grammar" if grammar == "BROKEN" else ""

    @staticmethod
    def grammar_from_json_schema(schema, overrides=None):
        return "schema:" + json.dumps(schema, sort_keys=True) + "|" + json.dumps(overrides)

    def is_error(self):
        return bool(self.error)

    def get_error(self):
        return self.error

    def is_stopped(self):
        return len(self.consumed) >= STOP_AFTER

    def consume_token(self, token):
        if self.is_stopped():
            self.error = "token after the end"
        self.consumed.append(token)

    def rollback(self, n):
        self.rolled_back += n
        del self.consumed[len(self.consumed) - n :]


def _fill(matcher, bitmask, index):
    bitmask[index, :] = 0
    allowed = len(matcher.consumed)
    bitmask[index, allowed // 32] |= np.int32(np.uint32(1 << (allowed % 32)).view(np.int32))


@pytest.fixture
def llg(monkeypatch):
    """A stand-in ``llguidance`` package in ``sys.modules``."""
    built: list[tuple] = []
    pkg = types.ModuleType("llguidance")
    pkg.__spec__ = importlib.machinery.ModuleSpec("llguidance", None)
    pkg.LLMatcher = _FakeMatcher
    pkg.grammar_from = lambda kind, text: f"{kind}:{text}"
    hf = types.ModuleType("llguidance.hf")

    def from_tokenizer(tok, n_vocab, eos_token=None):
        built.append((tok, n_vocab, eos_token))
        return ("lltok", n_vocab)

    hf.from_tokenizer = from_tokenizer
    npmod = types.ModuleType("llguidance.numpy")
    npmod.allocate_token_bitmask = lambda batch, n: np.zeros(
        (batch, (n + 31) // 32), dtype=np.int32
    )
    npmod.fill_next_token_bitmask = _fill
    mlxmod = types.ModuleType("llguidance.mlx")
    mlxmod.apply_token_bitmask = lambda logits, mask: ("masked", logits, mask)
    for name, mod in {
        "llguidance": pkg,
        "llguidance.hf": hf,
        "llguidance.numpy": npmod,
        "llguidance.mlx": mlxmod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    import hfl.engine.constrained as constrained

    monkeypatch.setattr(constrained, "_TOKENIZERS", {})
    pkg.built = built
    return pkg


def _allowed(mask) -> list[int]:
    bits = np.unpackbits(
        np.ascontiguousarray(mask, dtype="<i4").reshape(1, -1).view(np.uint8),
        axis=-1,
        bitorder="little",
    )
    return [int(i) for i in np.flatnonzero(bits[0])]


class TestAvailableAndGrammar:
    def test_available_follows_the_installed_package(self, llg, monkeypatch):
        from hfl.engine.constrained import available

        assert available() is True
        monkeypatch.setitem(sys.modules, "llguidance", None)
        assert available() is False

    def test_json_is_an_object_schema_with_one_line_separators(self, llg):
        from hfl.engine.constrained import _JSON_OPTIONS, grammar_for

        out = grammar_for("json")
        assert out.startswith('schema:{"type": "object"}')
        assert json.dumps(_JSON_OPTIONS) in out
        assert _JSON_OPTIONS["whitespace_flexible"] is False

    def test_a_schema_and_a_gbnf_grammar(self, llg):
        from hfl.engine.constrained import grammar_for

        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        assert grammar_for(schema).startswith("schema:" + json.dumps(schema, sort_keys=True))
        assert grammar_for('GBNF:root ::= "x"') == 'gbnf:root ::= "x"'

    @pytest.mark.parametrize("bad", ["yaml", 42, None, ["json"]])
    def test_anything_else_is_refused(self, llg, bad):
        from hfl.engine.constrained import grammar_for

        with pytest.raises(ValueError, match="not a response format"):
            grammar_for(bad)


class TestTokenizerCache:
    def test_built_once_per_tokenizer_and_vocabulary(self, llg):
        from hfl.engine.constrained import _ll_tokenizer

        tok = object()
        first = _ll_tokenizer(tok, 100, [2])
        assert _ll_tokenizer(tok, 100, [2]) is first
        assert len(llg.built) == 1
        _ll_tokenizer(tok, 200, None)
        assert len(llg.built) == 2
        # No EOS list is passed as None, not as an empty list.
        assert llg.built[-1] == (tok, 200, None)

    def test_a_stale_entry_for_a_reused_id_is_rebuilt(self, llg, monkeypatch):
        import hfl.engine.constrained as constrained

        tok = object()
        # An entry under this id that holds another tokenizer object.
        constrained._TOKENIZERS[(id(tok), 50)] = (object(), "stale")
        assert constrained._ll_tokenizer(tok, 50, None) == ("lltok", 50)
        assert constrained._TOKENIZERS[(id(tok), 50)][0] is tok


class TestGuide:
    def test_masks_follow_the_tokens_drawn(self, llg):
        from hfl.engine.constrained import Guide

        guide = Guide(object(), "json", eos=[9])
        assert _allowed(guide.mask(None, 40)) == [0]
        assert _allowed(guide.mask(0, 40)) == [1]
        # No token drawn (None) after the first step consumes nothing.
        assert _allowed(guide.mask(None, 40)) == [1]
        assert _allowed(guide.mask(1, 40)) == [2]

    def test_a_grammar_the_matcher_rejects_is_a_value_error(self, llg, monkeypatch):
        import hfl.engine.constrained as constrained

        monkeypatch.setattr(constrained, "grammar_for", lambda fmt: "BROKEN")
        guide = constrained.Guide(object(), "json")
        with pytest.raises(ValueError, match="bad grammar"):
            guide.mask(None, 16)

    def test_past_the_end_the_logits_are_left_alone(self, llg):
        from hfl.engine.constrained import Guide

        guide = Guide(object(), "json")
        guide.mask(None, 16)
        for token in range(STOP_AFTER):
            assert guide.mask(token, 16) is not None
        # One more token: the matcher errors and the guide gives no mask.
        assert guide.mask(STOP_AFTER, 16) is None

    def test_sync_mask_rolls_back_rejected_draft_tokens(self, llg):
        from hfl.engine.constrained import Guide

        guide = Guide(object(), "json")
        assert _allowed(guide.sync_mask([], 32)) == [0]
        # The draft proposed 0, 1, 7; then the target kept 0 and drew 1 again.
        guide.sync_mask([0, 1, 7], 32)
        matcher = guide._matcher
        assert matcher.consumed == [0, 1, 7]
        assert _allowed(guide.sync_mask([0, 1], 32)) == [2]
        assert matcher.rolled_back == 1
        assert matcher.consumed == [0, 1]

    def test_sync_mask_stops_consuming_at_the_end_of_the_format(self, llg):
        from hfl.engine.constrained import Guide

        guide = Guide(object(), "json")
        mask = guide.sync_mask([0, 1, 2, 3, 4], 32)
        # Only STOP_AFTER tokens consumed; the matcher never went into error.
        assert guide._matcher.consumed == [0, 1, 2]
        assert guide._seen == [0, 1, 2]
        assert not guide._matcher.is_error()
        assert mask is not None


class TestMlxProcessor:
    def test_masks_only_what_came_after_the_prompt(self, llg):
        from hfl.engine.constrained import mlx_processor

        inner = object()
        tokenizer = types.SimpleNamespace(_tokenizer=inner, eos_token_ids=[5, 6])
        process = mlx_processor(tokenizer, "json")
        logits = np.zeros((1, 32))
        prompt = np.array([11, 12, 13])
        tag, out_logits, mask = process(prompt, logits)
        assert tag == "masked" and out_logits is logits
        assert _allowed(mask) == [0]
        # Two tokens drawn after the 3-token prompt.
        _, _, mask = process(np.array([11, 12, 13, 0, 1]), logits)
        assert _allowed(mask) == [2]
        # The tokenizer llguidance saw is the wrapped HF one, with its EOS ids.
        assert llg.built[-1] == (inner, 32, [5, 6])

    def test_past_the_end_the_logits_pass_through(self, llg):
        from hfl.engine.constrained import mlx_processor

        process = mlx_processor(object(), "json")
        logits = np.zeros((1, 16))
        process(np.array([1]), logits)
        process(np.array([1, 0, 1, 2]), logits)
        # The guide stopped consuming at the end: the mask still applies.
        result = process(np.array([1, 0, 1, 2, 3]), logits)
        assert result[0] == "masked"


# ---------------------------------------------------------------- torch path


class _T:
    """The handful of torch.Tensor operations ``mask_scores`` uses, on numpy."""

    def __init__(self, data):
        self.a = np.asarray(data)
        self.device = "cpu"

    @property
    def shape(self):
        return self.a.shape

    def __getitem__(self, key):
        return _T(self.a[key])

    def __setitem__(self, key, value):
        self.a[key] = value.a if isinstance(value, _T) else value

    def __invert__(self):
        return _T(~self.a)

    def to(self, device):
        return self

    def masked_fill_(self, mask, value):
        self.a[..., mask.a] = value
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bool = np.bool_
    torch.zeros = lambda n, dtype=None: _T(np.zeros(n, dtype=dtype))
    torch.from_numpy = lambda arr: _T(arr)
    transformers = types.ModuleType("transformers")
    transformers.LogitsProcessor = type("LogitsProcessor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return torch


def _bitmask(allowed: list[int], words: int):
    mask = np.zeros((1, words), dtype=np.int32)
    for token in allowed:
        mask[0, token // 32] |= np.int32(np.uint32(1 << (token % 32)).view(np.int32))
    return mask


class TestTorchPath:
    def test_mask_scores_keeps_only_the_allowed_tokens(self, fake_torch):
        from hfl.engine.constrained import mask_scores

        scores = _T(np.zeros((1, 70)))
        mask_scores(scores, _bitmask([1, 33, 64], 3))
        kept = [i for i in range(70) if np.isfinite(scores.a[0, i])]
        assert kept == [1, 33, 64]

    def test_a_vocabulary_wider_than_the_mask_refuses_the_rest(self, fake_torch):
        from hfl.engine.constrained import mask_scores

        scores = _T(np.zeros((1, 40)))
        mask_scores(scores, _bitmask([3], 1))
        assert np.isfinite(scores.a[0, 3])
        assert np.isneginf(scores.a[0, 32:]).all()

    def test_torch_processor_follows_the_generated_ids(self, llg, fake_torch):
        from hfl.engine.constrained import torch_processor

        processor = torch_processor(object(), "json", [7])
        assert isinstance(processor, sys.modules["transformers"].LogitsProcessor)
        scores = _T(np.zeros((1, 8)))
        out = processor(np.array([[4, 4]]), scores)
        # First call: the prompt's last id is not consumed; token 0 allowed.
        assert out is scores
        assert [i for i in range(8) if np.isfinite(scores.a[0, i])] == [0]
        scores = _T(np.zeros((1, 8)))
        processor(np.array([[4, 4, 0]]), scores)
        assert [i for i in range(8) if np.isfinite(scores.a[0, i])] == [1]

    def test_torch_processor_leaves_scores_once_the_format_failed(self, llg, fake_torch):
        from hfl.engine.constrained import torch_processor

        processor = torch_processor(object(), "json", None)
        processor(np.array([[9]]), _T(np.zeros((1, 8))))
        for token in range(STOP_AFTER):
            processor(np.array([[token]]), _T(np.zeros((1, 8))))
        scores = _T(np.zeros((1, 8)))
        processor(np.array([[STOP_AFTER]]), scores)
        assert np.isfinite(scores.a).all()

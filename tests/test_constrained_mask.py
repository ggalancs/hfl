# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The JSON/schema mask on the Transformers engine, without ``torch.compile``.

llguidance's torch helper compiles its kernel at run time, which needs a C++
compiler: on Windows (no ``cl`` on PATH) every ``format`` request failed
with an InductorError (found on a real Windows 10). HFL applies the bitmask
with plain torch; this pins it to the same result."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from hfl.engine.constrained import mask_scores  # noqa: E402


def _bitmask(allowed: list[int], words: int):
    mask = np.zeros((1, words), dtype=np.int32)
    for token in allowed:
        mask[0, token // 32] |= np.int32(np.uint32(1 << (token % 32)).view(np.int32))
    return mask


def test_only_the_allowed_tokens_stay() -> None:
    allowed = [0, 5, 31, 32, 63, 70, 99]
    scores = torch.zeros(1, 100)
    mask_scores(scores, _bitmask(allowed, 4))
    kept = [i for i in range(100) if scores[0, i] != float("-inf")]
    assert kept == allowed


def test_tokens_beyond_the_mask_are_refused() -> None:
    scores = torch.zeros(1, 40)
    mask_scores(scores, _bitmask([3], 1))  # one word: tokens 0-31
    assert scores[0, 3] == 0 and bool(torch.isinf(scores[0, 32:]).all())


def test_the_same_as_llguidance_s_own() -> None:
    lt = pytest.importorskip("llguidance.torch")
    rng = np.random.default_rng(0)
    mask = rng.integers(-(2**31), 2**31 - 1, size=(1, 5), dtype=np.int64).astype(np.int32)
    ours, theirs = torch.randn(1, 150), None
    theirs = ours.clone()
    mask_scores(ours, mask)
    try:
        lt.apply_token_bitmask_inplace(theirs, torch.from_numpy(mask))
    except Exception as exc:  # no C++ compiler here: nothing to compare against
        pytest.skip(f"llguidance's compiled kernel unavailable: {exc}")
    assert torch.equal(torch.isinf(ours), torch.isinf(theirs))

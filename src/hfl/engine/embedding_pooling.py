# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Embedding pooling strategies (Phase 12 P1 — V2 row 18).

Sentence-transformers models trained with CLS pooling or
last-token pooling produce subtly broken outputs when served with
the default mean pooling. Most HF users blindly reuse whichever
pooling the upstream README mentions, so HFL now accepts a
``pooling`` field on ``/api/embed`` and applies the matching
strategy at the boundary.

Strategies:
- ``mean``: average token embeddings, masked by attention mask.
- ``cls``: return the first token's embedding.
- ``last``: return the last non-pad token's embedding.

Pure-Python implementation so the module imports cleanly without
numpy. When numpy is available we take the fast path; otherwise
we fall back to a list-comprehension path that's O(N) in tokens
and adequate for the scale of sentence embeddings (≤ 512 tokens).
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

__all__ = ["POOLING_STRATEGIES", "Pooling", "pool"]


Pooling = Literal["mean", "cls", "last"]
POOLING_STRATEGIES: tuple[Pooling, ...] = ("mean", "cls", "last")


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover — numpy is a transitive dep
        return None
    return np


def pool(
    token_embeddings: Sequence[Sequence[float]],
    attention_mask: Sequence[int] | None = None,
    strategy: Pooling = "mean",
) -> list[float]:
    """Pool a ``[n_tokens, hidden_size]`` matrix into a single vector.

    ``token_embeddings`` is a 2-D matrix of floats (one row per
    token). ``attention_mask`` has the same length as the first
    dimension; ``None`` treats every token as valid.

    Returns a flat list of floats of length ``hidden_size``.
    Unknown strategies fall back to ``mean`` with a pure-Python
    implementation.
    """
    if not token_embeddings:
        raise ValueError("token_embeddings is empty")
    n_tokens = len(token_embeddings)
    if attention_mask is None:
        mask: list[int] = [1] * n_tokens
    else:
        if len(attention_mask) != n_tokens:
            raise ValueError("attention_mask length must match token count")
        mask = list(attention_mask)

    strategy_lc = strategy.lower() if isinstance(strategy, str) else "mean"
    if strategy_lc not in POOLING_STRATEGIES:
        strategy_lc = "mean"

    np = _require_numpy()
    if np is not None:
        mat = np.asarray(token_embeddings, dtype=np.float32)
        mask_arr = np.asarray(mask, dtype=np.float32)
        return _pool_numpy(mat, mask_arr, strategy_lc, np)

    return _pool_python(token_embeddings, mask, strategy_lc)


def _pool_numpy(mat, mask, strategy: str, np):  # type: ignore[no-untyped-def]
    if strategy == "cls":
        return mat[0].tolist()
    if strategy == "last":
        # Largest index where mask is non-zero.
        valid = np.nonzero(mask)[0]
        idx = int(valid[-1]) if valid.size else int(mat.shape[0] - 1)
        return mat[idx].tolist()
    # Mean pooling weighted by mask.
    denom = float(mask.sum()) or 1.0
    summed = (mat.T * mask).T.sum(axis=0)
    return (summed / denom).tolist()


def _pool_python(
    matrix: Sequence[Sequence[float]],
    mask: list[int],
    strategy: str,
) -> list[float]:
    hidden = len(matrix[0])
    if strategy == "cls":
        return [float(x) for x in matrix[0]]
    if strategy == "last":
        idx = _last_valid_index(mask)
        return [float(x) for x in matrix[idx]]
    # Mean pooling.
    acc = [0.0] * hidden
    total = 0
    for row, keep in zip(matrix, mask):
        if not keep:
            continue
        for i, value in enumerate(row):
            acc[i] += float(value)
        total += 1
    if total == 0:
        return [0.0] * hidden
    return [value / total for value in acc]


def _last_valid_index(mask: Iterable[int]) -> int:
    idx = -1
    for i, value in enumerate(mask):
        if value:
            idx = i
    return max(idx, 0)

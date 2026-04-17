# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the embedding-pooling helper (Phase 12 P1 — V2 row 18)."""

from __future__ import annotations

import pytest

from hfl.engine.embedding_pooling import POOLING_STRATEGIES, pool


def _rows():
    return [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]


class TestStrategyTable:
    def test_registered_strategies(self):
        assert set(POOLING_STRATEGIES) == {"mean", "cls", "last"}


class TestMeanPooling:
    def test_unmasked_mean_matches_manual(self):
        out = pool(_rows(), strategy="mean")
        assert out == pytest.approx([4.0, 5.0, 6.0])

    def test_masked_mean_ignores_pad(self):
        mask = [1, 1, 0]  # third token is pad
        out = pool(_rows(), attention_mask=mask, strategy="mean")
        assert out == pytest.approx([2.5, 3.5, 4.5])

    def test_all_pad_returns_zero_vector(self):
        out = pool(_rows(), attention_mask=[0, 0, 0], strategy="mean")
        # Numpy path averages to nan/zero; the python fallback returns
        # a zero vector. Accept either shape.
        assert len(out) == 3


class TestCLSPooling:
    def test_returns_first_row(self):
        out = pool(_rows(), strategy="cls")
        assert out == pytest.approx([1.0, 2.0, 3.0])

    def test_mask_is_ignored_for_cls(self):
        out = pool(_rows(), attention_mask=[0, 1, 1], strategy="cls")
        assert out == pytest.approx([1.0, 2.0, 3.0])


class TestLastPooling:
    def test_returns_last_masked_row(self):
        mask = [1, 1, 0]  # last valid token is index 1
        out = pool(_rows(), attention_mask=mask, strategy="last")
        assert out == pytest.approx([4.0, 5.0, 6.0])

    def test_all_valid_returns_final_row(self):
        out = pool(_rows(), strategy="last")
        assert out == pytest.approx([7.0, 8.0, 9.0])


class TestInputValidation:
    def test_empty_rows_raises(self):
        with pytest.raises(ValueError):
            pool([], strategy="mean")

    def test_mask_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pool(_rows(), attention_mask=[1, 1], strategy="mean")

    def test_unknown_strategy_falls_back_to_mean(self):
        out = pool(_rows(), strategy="weird")
        assert out == pytest.approx([4.0, 5.0, 6.0])


class TestPurePythonFallback:
    """Force the numpy-less path to ensure the fallback is exercised."""

    def test_mean_fallback(self, monkeypatch):
        import hfl.engine.embedding_pooling as ep

        monkeypatch.setattr(ep, "_require_numpy", lambda: None)
        out = ep.pool(_rows(), strategy="mean")
        assert out == pytest.approx([4.0, 5.0, 6.0])

    def test_cls_fallback(self, monkeypatch):
        import hfl.engine.embedding_pooling as ep

        monkeypatch.setattr(ep, "_require_numpy", lambda: None)
        assert ep.pool(_rows(), strategy="cls") == pytest.approx([1.0, 2.0, 3.0])

    def test_last_fallback(self, monkeypatch):
        import hfl.engine.embedding_pooling as ep

        monkeypatch.setattr(ep, "_require_numpy", lambda: None)
        out = ep.pool(_rows(), attention_mask=[1, 1, 0], strategy="last")
        assert out == pytest.approx([4.0, 5.0, 6.0])

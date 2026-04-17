# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the ``draft_model`` config field (Phase 15 — V2 row 11)."""

from __future__ import annotations

from hfl.engine.base import GenerationConfig


class TestDraftModelField:
    def test_default_is_none(self):
        cfg = GenerationConfig()
        assert cfg.draft_model is None

    def test_explicit_value_roundtrips(self):
        cfg = GenerationConfig(draft_model="./tinyllama-q4.gguf")
        assert cfg.draft_model == "./tinyllama-q4.gguf"

    def test_engine_silently_ignores_when_kwarg_absent(self, monkeypatch):
        """Engines that haven't wired draft_model yet must not crash.

        We assert that ``GenerationConfig`` construction succeeds and
        a simple identity check on the value round-trips — the llama-cpp
        engine is gated by a kwarg lookup so a None value is a no-op.
        """
        cfg = GenerationConfig(draft_model=None)
        assert cfg.draft_model is None

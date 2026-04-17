# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the shared-prefix prompt cache (Phase 11 P1 — V2 row 10)."""

from __future__ import annotations

import pytest

from hfl.engine.prompt_cache import PromptPrefixCache, get_prompt_cache, reset_prompt_cache


@pytest.fixture(autouse=True)
def _fresh():
    reset_prompt_cache()
    yield
    reset_prompt_cache()


class TestBasicBehaviour:
    def test_empty_cache_returns_zero(self):
        c = PromptPrefixCache()
        assert c.longest_prefix("m", [1, 2, 3]) == 0

    def test_recording_and_lookup_exact_match(self):
        c = PromptPrefixCache()
        c.record("m", [1, 2, 3, 4])
        assert c.longest_prefix("m", [1, 2, 3, 4]) == 4

    def test_partial_prefix_match(self):
        c = PromptPrefixCache()
        c.record("m", [1, 2, 3, 4, 5])
        # Request shares 1-2-3 then diverges.
        assert c.longest_prefix("m", [1, 2, 3, 9, 9]) == 3

    def test_shorter_request_than_cached(self):
        c = PromptPrefixCache()
        c.record("m", [1, 2, 3, 4, 5])
        assert c.longest_prefix("m", [1, 2]) == 2

    def test_longer_request_than_cached(self):
        c = PromptPrefixCache()
        c.record("m", [1, 2])
        # Whole cached sequence is a valid prefix of the request.
        assert c.longest_prefix("m", [1, 2, 3, 4, 5]) == 2

    def test_empty_tokens_returns_zero(self):
        c = PromptPrefixCache()
        c.record("m", [1, 2, 3])
        assert c.longest_prefix("m", []) == 0

    def test_unknown_model_returns_zero(self):
        c = PromptPrefixCache()
        c.record("a", [1, 2, 3])
        assert c.longest_prefix("b", [1, 2, 3]) == 0


class TestEviction:
    def test_lru_evicts_oldest(self):
        c = PromptPrefixCache(max_entries=2)
        c.record("a", [1])
        c.record("b", [2])
        c.record("c", [3])
        assert c.longest_prefix("a", [1]) == 0  # evicted
        assert c.longest_prefix("b", [2]) == 1
        assert c.longest_prefix("c", [3]) == 1

    def test_access_promotes_to_mru(self):
        c = PromptPrefixCache(max_entries=2)
        c.record("a", [1])
        c.record("b", [2])
        c.longest_prefix("a", [1])  # touch → promote
        c.record("c", [3])
        assert c.longest_prefix("b", [2]) == 0  # b was LRU, evicted
        assert c.longest_prefix("a", [1]) == 1

    def test_drop_removes_entry(self):
        c = PromptPrefixCache()
        c.record("m", [1, 2])
        c.drop("m")
        assert c.longest_prefix("m", [1, 2]) == 0

    def test_clear_empties_everything(self):
        c = PromptPrefixCache()
        c.record("a", [1])
        c.record("b", [2])
        c.clear()
        assert len(c) == 0


class TestSingleton:
    def test_singleton_returns_same_instance(self, temp_config):
        a = get_prompt_cache()
        b = get_prompt_cache()
        assert a is b

    def test_reset_drops_singleton(self, temp_config):
        a = get_prompt_cache()
        reset_prompt_cache()
        b = get_prompt_cache()
        assert a is not b

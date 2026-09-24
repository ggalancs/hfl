# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Memory footprints and the admission planner behind multi-residency.

The footprint numbers are pinned to measurements taken on a 128 GiB Mac
(2026-09-24): an 8.38 GB Qwen3-14B Q4_K_M GGUF took 9.0 GB resident at a
4096-token context, and gemma-4-31b-it-4bit (MLX) held 1.239 GiB of KV
cache after a 6 001-token prompt.
"""

from __future__ import annotations

import json
import random

import pytest

from hfl.engine import footprint as fp
from hfl.engine.footprint import GIB, estimate_footprint, footprint_of_loaded
from hfl.engine.residency import MemoryView, ResidentView, plan_admission

# gemma-4-31b-it text_config, the fields the estimator reads.
GEMMA4_TEXT = {
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "head_dim": 256,
    "hidden_size": 5376,
    "sliding_window": 1024,
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10,
    "global_head_dim": 512,
    "num_global_key_value_heads": 4,
    "max_position_embeddings": 262144,
}


def _write(path, size):
    path.write_bytes(b"\0" * size)
    return path


class TestGGUF:
    @pytest.fixture(autouse=True)
    def _layout(self, monkeypatch):
        # Qwen3-14B: 40 layers, 40 heads, 8 KV heads, 5120 wide -> head_dim 128.
        info = {"block_count": 40, "embedding_length": 5120, "head_count": 40, "head_count_kv": 8}
        monkeypatch.setattr("hfl.engine.llama_cpp._read_gguf_model_info", lambda p: dict(info))

    def test_weights_plus_kv_at_the_requested_context(self, tmp_path):
        model = _write(tmp_path / "m.gguf", 1000)
        f = estimate_footprint(model, 4096)
        assert f.weights_bytes == 1000
        # 2 (K,V) x 40 layers x 8 heads x 128 dims x 2 bytes per token.
        assert f.kv_bytes == 2 * 40 * 8 * 128 * 2 * 4096
        assert f.n_ctx == 4096 and f.kv_known

    def test_measured_qwen3_14b_kv_part(self, tmp_path):
        """0.625 GiB of KV at 4096 tokens: with 8.38 GB of weights that is
        the 9.0 GB measured resident."""
        f = estimate_footprint(_write(tmp_path / "m.gguf", 10), 4096)
        assert f.kv_bytes / GIB == pytest.approx(0.625, rel=0.001)

    def test_quantised_kv_cache_is_smaller(self, tmp_path, monkeypatch):
        from hfl.config import config

        model = _write(tmp_path / "m.gguf", 10)
        f16 = estimate_footprint(model, 4096).kv_bytes
        monkeypatch.setattr(config, "kv_cache_type", "q8_0")
        assert estimate_footprint(model, 4096).kv_bytes < 0.6 * f16

    def test_split_shards_are_summed(self, tmp_path):
        for i in (1, 2, 3):
            _write(tmp_path / f"big-0000{i}-of-00003.gguf", 100)
        _write(tmp_path / "other.gguf", 5000)
        f = estimate_footprint(tmp_path / "big-00001-of-00003.gguf", 1)
        assert f.weights_bytes == 300

    def test_a_directory_ignores_the_vision_projector(self, tmp_path):
        _write(tmp_path / "model-Q4_K_M.gguf", 100)
        _write(tmp_path / "mmproj-model-f16.gguf", 50)
        assert estimate_footprint(tmp_path, 1).weights_bytes == 100

    def test_undecided_context_is_capped_by_the_model(self, tmp_path, monkeypatch):
        info = {"block_count": 1, "embedding_length": 8, "head_count": 1, "head_count_kv": 1}
        monkeypatch.setattr(
            "hfl.engine.llama_cpp._read_gguf_model_info",
            lambda p: {**info, "max_context": 2048},
        )
        assert estimate_footprint(_write(tmp_path / "m.gguf", 1), 0).n_ctx == 2048

    def test_no_layout_means_weights_only_and_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr("hfl.engine.llama_cpp._read_gguf_model_info", lambda p: None)
        f = estimate_footprint(_write(tmp_path / "m.gguf", 77), 4096)
        assert f.total_bytes == 77 and not f.kv_known


class TestDirectories:
    def _model(self, tmp_path, cfg, files=(("model.safetensors", 1000),)):
        for name, size in files:
            _write(tmp_path / name, size)
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        return tmp_path

    def test_hybrid_attention_matches_the_measured_gemma4_cache(self, tmp_path):
        """Uniform layers would have said ~5.5 GiB; the cache measured 1.239."""
        d = self._model(tmp_path, {"text_config": GEMMA4_TEXT})
        f = estimate_footprint(d, 6001)
        assert f.kv_bytes / GIB == pytest.approx(1.239, rel=0.01)

    def test_uniform_layers_use_the_plain_formula(self, tmp_path):
        cfg = {"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 64}
        f = estimate_footprint(self._model(tmp_path, cfg), 100)
        # 2 x 2 layers x 4 kv heads (defaults to heads) x 16 dims x 2 bytes x 100
        assert f.kv_bytes == 2 * 2 * 4 * 16 * 2 * 100

    def test_bin_copies_are_not_counted_twice(self, tmp_path):
        d = self._model(
            tmp_path,
            {"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 2},
            files=(("model.safetensors", 100), ("pytorch_model.bin", 100)),
        )
        assert estimate_footprint(d, 1).weights_bytes == 100

    def test_undecided_context_defaults_to_a_conversation_not_the_maximum(self, tmp_path):
        d = self._model(tmp_path, {"text_config": GEMMA4_TEXT})
        assert estimate_footprint(d, 0).n_ctx == fp.DEFAULT_GROWING_CTX

    def test_unreadable_path_is_unknown_not_free(self, tmp_path):
        f = estimate_footprint(tmp_path / "missing", 4096)
        assert not f.known


class TestAfterLoad:
    def test_the_engine_reported_context_replaces_the_guess(self, tmp_path):
        d = tmp_path
        _write(d / "model.safetensors", 10)
        (d / "config.json").write_text(
            json.dumps({"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 2})
        )

        class Engine:
            context_size = 32

        assert footprint_of_loaded(d, Engine()).n_ctx == 32


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------

GB = 10**9


def mem(total=100, in_use=40, rss=10):
    return MemoryView(total=total * GB, in_use=in_use * GB, hfl_rss=rss * GB)


def res(name, gb, last_used, busy=False, mine=False):
    return ResidentView(name, gb * GB, last_used, busy=busy, mine=mine)


class TestPlanner:
    def test_fits_alongside(self):
        plan = plan_admission(20 * GB, mem(), [res("a", 10, 1)], 0.85)
        assert plan.fits and plan.evict == () and plan.reason == "fits"
        # others 30 + overhead 0 + a 10 + new 20
        assert plan.used_after == 60 * GB

    def test_evicts_least_recently_used_and_only_as_many_as_needed(self):
        residents = [res("new", 20, 3), res("old", 20, 1), res("mid", 20, 2)]
        plan = plan_admission(30 * GB, mem(in_use=90, rss=60), residents, 0.85)
        # others 30; kept 60 + new 30 = 120 > 85. Drop old (100), then mid (80).
        assert plan.fits and plan.evict == ("old", "mid")
        assert plan.used_after <= plan.limit

    def test_a_busy_model_is_waited_for_not_evicted(self):
        residents = [res("busy", 40, 1, busy=True)]
        plan = plan_admission(30 * GB, mem(in_use=70, rss=40), residents, 0.85)
        assert not plan.fits and plan.reason == "wait" and plan.wait_for == ("busy",)
        assert "busy" not in plan.evict

    def test_the_loaders_own_model_is_never_waited_for(self):
        residents = [res("mine", 40, 1, mine=True)]
        plan = plan_admission(30 * GB, mem(in_use=70, rss=40), residents, 0.85)
        assert not plan.fits and plan.reason == "blocked" and plan.wait_for == ()

    def test_too_big_even_alone(self):
        plan = plan_admission(80 * GB, mem(in_use=30, rss=0), [], 0.85)
        assert not plan.fits and plan.reason == "too_big"
        assert plan.floor == 110 * GB

    def test_too_big_is_decided_before_evicting_anything(self):
        """Unloading everything to then refuse would cost the user their
        models and gain nothing."""
        plan = plan_admission(80 * GB, mem(in_use=40, rss=10), [res("a", 10, 1)], 0.85)
        assert plan.reason == "too_big" and plan.evict == ()

    def test_process_overhead_is_not_attributed_to_other_programs(self):
        # rss 15 but models account for 10: 5 GB of interpreter/libraries.
        plan = plan_admission(10 * GB, mem(in_use=45, rss=15), [res("a", 10, 1)], 0.85)
        assert plan.used_after == (30 + 5 + 10 + 10) * GB

    def test_count_ceiling(self):
        residents = [res("a", 1, 1), res("b", 1, 2)]
        plan = plan_admission(1 * GB, mem(), residents, 0.85, max_models=2)
        assert plan.fits and plan.evict == ("a",)

    def test_no_ceiling_by_default(self):
        residents = [res(str(i), 1, i) for i in range(20)]
        assert plan_admission(1 * GB, mem(in_use=60, rss=30), residents, 0.85).evict == ()


def test_planner_invariants_over_random_scenarios():
    """Whatever the mix: an admitted load stays within the limit, and no
    busy or loader-held model is ever in the eviction list."""
    rng = random.Random(1234)
    checked = 0
    for _ in range(5000):
        residents = [
            res(
                f"m{i}",
                rng.randint(1, 30),
                rng.random(),
                busy=rng.random() < 0.3,
                mine=rng.random() < 0.1,
            )
            for i in range(rng.randint(0, 6))
        ]
        rss = sum(r.footprint for r in residents) // GB + rng.randint(0, 5)
        in_use = rss + rng.randint(0, 60)
        m = MemoryView(total=128 * GB, in_use=in_use * GB, hfl_rss=rss * GB)
        plan = plan_admission(rng.randint(1, 60) * GB, m, residents, rng.uniform(0.5, 0.95))
        protected = {r.name for r in residents if r.busy or r.mine}
        assert not protected & set(plan.evict)
        if plan.fits:
            assert plan.used_after <= plan.limit
            checked += 1
        if plan.reason == "wait":
            assert set(plan.wait_for) <= {r.name for r in residents if r.busy}
    assert checked > 500, "the scenarios never exercised an admitted load"

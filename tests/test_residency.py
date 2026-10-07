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


class TestGPUPool:
    def test_ram_fits_but_the_card_does_not(self):
        plan = plan_admission(
            10 * GB,
            mem(total=128, in_use=40, rss=20),
            [res("a", 10, 1), res("b", 10, 2)],
            0.85,
            gpu=MemoryView(total=32 * GB, in_use=21 * GB, hfl_rss=20 * GB),
        )
        # GPU: 1 other + a 10 + b 10 + new 10 = 31 > 27.2 -> drop a -> 21.
        assert plan.fits and plan.evict == ("a",)
        assert plan.gpu_used_after <= plan.gpu_limit

    def test_too_big_for_the_card_names_the_gpu(self):
        plan = plan_admission(
            30 * GB,
            mem(total=128, in_use=20, rss=0),
            [],
            0.85,
            gpu=MemoryView(total=24 * GB, in_use=1 * GB, hfl_rss=0),
        )
        assert plan.reason == "too_big" and plan.constraint == "gpu"
        assert plan.gpu_floor == 31 * GB

    def test_without_a_gpu_nothing_changes(self):
        with_none = plan_admission(10 * GB, mem(), [res("a", 10, 1)], 0.85, gpu=None)
        plain = plan_admission(10 * GB, mem(), [res("a", 10, 1)], 0.85)
        assert with_none == plain


class TestNvidiaSmi:
    @staticmethod
    def _smi(monkeypatch, outputs):
        import subprocess

        from hfl.engine import residency

        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            query = cmd[1]
            if query not in outputs:
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=outputs[query], stderr="")

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(subprocess, "run", run)
        return residency, calls

    def test_two_cards_are_summed_and_our_share_found(self, monkeypatch):
        import os

        residency, _ = self._smi(
            monkeypatch,
            {
                "--query-gpu=memory.total,memory.used": "24576, 2048\n24576, 1024\n",
                "--query-compute-apps=pid,used_memory": f"{os.getpid()}, 1500\n999999, 300\n",
            },
        )
        view = residency.current_gpu_memory()
        mib = 1024**2
        assert view == MemoryView(total=49152 * mib, in_use=3072 * mib, hfl_rss=1500 * mib)

    def test_a_failing_nvidia_smi_is_no_measurement(self, monkeypatch):
        residency, _ = self._smi(monkeypatch, {})
        assert residency.current_gpu_memory() is None

    def test_no_nvidia_smi_is_no_measurement(self, monkeypatch):
        from hfl.engine import residency

        monkeypatch.setattr("shutil.which", lambda name: None)
        assert residency.current_gpu_memory() is None

    def test_rocm_is_a_gpu_we_cannot_measure(self, monkeypatch):
        from hfl.engine import residency

        monkeypatch.setattr(residency, "_UNMEASURED_GPU", None)
        monkeypatch.setattr(residency, "current_gpu_memory", lambda: None)
        monkeypatch.setattr(
            "shutil.which", lambda name: "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else None
        )
        assert residency.discrete_gpu_unmeasured() is True

    def test_a_measured_gpu_is_not_unmeasured(self, monkeypatch):
        from hfl.engine import residency

        monkeypatch.setattr(residency, "_UNMEASURED_GPU", None)
        monkeypatch.setattr(residency, "current_gpu_memory", lambda: MemoryView(1, 0, 0))
        assert residency.discrete_gpu_unmeasured() is False


class TestRosetta:
    """An x86_64 Python on Apple Silicon sees 4 KB pages while the kernel
    counts 16 KB ones: psutil reported a quarter of the available memory
    (19.2 of 77.0 GB) and HFL's x86_64 build refused a 0.6 GB model on a
    128 GB Mac with 96 % free (local CI of 0.27.0, 2026-10-07)."""

    @staticmethod
    def _mac(monkeypatch, translated, vm_stat_head, own_page=4096):
        import resource
        import subprocess
        import sys

        from hfl.engine import residency

        def run(cmd, **kw):
            out = {"sysctl": translated + "\n", "vm_stat": vm_stat_head + "\nPages free: 1.\n"}[
                cmd[0]
            ]
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(resource, "getpagesize", lambda: own_page)
        monkeypatch.setattr(residency, "_PAGE_RATIO", None)
        return residency

    HEAD = "Mach Virtual Memory Statistics: (page size of 16384 bytes)"

    def test_translated_counts_machine_pages(self, monkeypatch):
        residency = self._mac(monkeypatch, "1", self.HEAD)
        assert residency._rosetta_page_ratio() == 4

    def test_native_is_one(self, monkeypatch):
        residency = self._mac(monkeypatch, "0", self.HEAD, own_page=16384)
        assert residency._rosetta_page_ratio() == 1

    def test_unreadable_is_one(self, monkeypatch):
        residency = self._mac(monkeypatch, "1", "something else")
        assert residency._rosetta_page_ratio() == 1

    def test_not_a_mac_is_one(self, monkeypatch):
        import sys

        from hfl.engine import residency

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(residency, "_PAGE_RATIO", None)
        assert residency._rosetta_page_ratio() == 1

    def test_available_memory_is_corrected(self, monkeypatch):
        psutil = pytest.importorskip("psutil")
        from types import SimpleNamespace

        from hfl.engine import residency

        gib = 1024**3
        # What psutil read under Rosetta on that Mac: 19.2 GB of 77.0 available.
        fake = SimpleNamespace(total=128 * gib, available=int(19.25 * gib))
        monkeypatch.setattr(psutil, "virtual_memory", lambda: fake)
        monkeypatch.setattr(residency, "_own_processes", list)
        monkeypatch.setattr(residency, "_rosetta_page_ratio", lambda: 4)
        view = residency.current_memory()
        assert view is not None and view.in_use == 128 * gib - 77 * gib


class TestAmdSmi:
    """AMD GPUs read through rocm-smi (amd-smi as its successor). Without it
    HFL kept one model at a time on an MI300X with 192 GB (audit E16 on the
    AMD Developer Cloud, 2026-10-07). The outputs are the real ones from that
    MI300X with llama-server (PID 28389) holding a model."""

    ROCM_VRAM = (
        '{"card0": {"VRAM Total Memory (B)": "205822885888", '
        '"VRAM Total Used Memory (B)": "2654650368"}}'
    )
    ROCM_PIDS = '{"system": {"PID28389": "llama-server, 1, 2353860608, 11011, 0"}}'
    AMD_MEM = (
        '{"gpu_data": [{"gpu": 0, "mem_usage": {"total_vram": {"value": 196288, "unit": "MB"}, '
        '"used_vram": {"value": 2531, "unit": "MB"}}}]}'
    )
    AMD_PROCS = (
        '[{"gpu": 0, "process_list": [{"process_info": {"name": "llama-server", '
        '"pid": 28389, "mem_usage": {"value": 2353860608, "unit": "B"}}}]}]'
    )

    @staticmethod
    def _tools(monkeypatch, outputs, present=("rocm-smi",)):
        import subprocess

        from hfl.engine import residency

        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            key = (cmd[0].rsplit("/", 1)[-1], *cmd[1:])
            if key not in outputs:
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=outputs[key], stderr="")

        monkeypatch.setattr(
            "shutil.which", lambda name: "/opt/rocm/bin/" + name if name in present else None
        )
        monkeypatch.setattr(subprocess, "run", run)
        return residency, calls

    def test_rocm_smi_total_used_and_our_share(self, monkeypatch):
        residency, _ = self._tools(
            monkeypatch,
            {
                ("rocm-smi", "--showmeminfo", "vram", "--json"): self.ROCM_VRAM,
                ("rocm-smi", "--showpids", "--json"): self.ROCM_PIDS,
            },
        )
        monkeypatch.setattr(residency, "_own_pids", lambda: {28389})
        view = residency.current_gpu_memory()
        assert view == MemoryView(
            total=205822885888, in_use=2654650368, hfl_rss=2353860608, attributed=True
        )

    def test_rocm_smi_another_programs_model_is_not_ours(self, monkeypatch):
        residency, _ = self._tools(
            monkeypatch,
            {
                ("rocm-smi", "--showmeminfo", "vram", "--json"): self.ROCM_VRAM,
                ("rocm-smi", "--showpids", "--json"): self.ROCM_PIDS,
            },
        )
        monkeypatch.setattr(residency, "_own_pids", lambda: {1})
        view = residency.current_gpu_memory()
        assert view is not None and view.hfl_rss == 0 and view.attributed is False

    def test_no_process_list_still_measures_the_card(self, monkeypatch):
        residency, _ = self._tools(
            monkeypatch,
            {
                ("rocm-smi", "--showmeminfo", "vram", "--json"): self.ROCM_VRAM,
                ("rocm-smi", "--showpids", "--json"): "WARNING: No JSON data to report\n",
            },
        )
        view = residency.current_gpu_memory()
        assert view is not None and view.total == 205822885888 and view.attributed is False

    def test_amd_smi_when_rocm_smi_is_gone(self, monkeypatch):
        residency, _ = self._tools(
            monkeypatch,
            {
                ("amd-smi", "metric", "--mem-usage", "--json"): self.AMD_MEM,
                ("amd-smi", "process", "--json"): self.AMD_PROCS,
            },
            present=("amd-smi",),
        )
        monkeypatch.setattr(residency, "_own_pids", lambda: {28389})
        mib = 1024**2
        view = residency.current_gpu_memory()
        assert view == MemoryView(
            total=196288 * mib, in_use=2531 * mib, hfl_rss=2353860608, attributed=True
        )

    def test_a_failing_tool_is_no_measurement(self, monkeypatch):
        residency, _ = self._tools(monkeypatch, {}, present=("rocm-smi", "amd-smi"))
        assert residency.current_gpu_memory() is None

    def test_garbled_output_is_no_measurement(self, monkeypatch):
        residency, _ = self._tools(
            monkeypatch,
            {("rocm-smi", "--showmeminfo", "vram", "--json"): '{"card0": {"VRAM": "lots"}}'},
        )
        assert residency.current_gpu_memory() is None

    def test_a_readable_amd_gpu_is_measured_not_one_model_at_a_time(self, monkeypatch):
        residency, _ = self._tools(
            monkeypatch,
            {
                ("rocm-smi", "--showmeminfo", "vram", "--json"): self.ROCM_VRAM,
                ("rocm-smi", "--showpids", "--json"): self.ROCM_PIDS,
            },
        )
        monkeypatch.setattr(residency, "_UNMEASURED_GPU", None)
        assert residency.discrete_gpu_unmeasured() is False

    def test_nvidia_is_read_first(self, monkeypatch):
        residency, calls = self._tools(
            monkeypatch,
            {
                (
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ): "24576, 2048\n",
                (
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ): "",
            },
            present=("nvidia-smi", "rocm-smi"),
        )
        view = residency.current_gpu_memory()
        assert view is not None and view.total == 24576 * 1024**2
        assert not any("rocm-smi" in c[0] for c in calls)


class TestChildProcessesAreOurs:
    """llama-server and vLLM run models in child processes. Counted as other
    programs', each model was charged twice at admission and could not be
    evicted to make room (a second model planned with the first counted in
    "in use" and again as its footprint)."""

    def test_a_childs_ram_counts_as_ours(self):
        # Without psutil there is no measurement at all (the CI venv).
        pytest.importorskip("psutil")
        import subprocess
        import sys
        import time

        from hfl.engine import residency

        alone = residency.current_memory()
        assert alone is not None
        # A child holding ~200 MB, as a llama-server holds its model.
        child = subprocess.Popen(
            [sys.executable, "-c",
             "b = bytearray(200 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); "
             "import time; time.sleep(60)"]
        )  # fmt: skip
        try:
            time.sleep(1.5)
            with_child = residency.current_memory()
            assert with_child is not None
            assert with_child.hfl_rss - alone.hfl_rss > 150 * 1024**2
        finally:
            child.kill()
            child.wait()

    def test_a_childs_vram_counts_as_ours(self, monkeypatch):
        import os

        from hfl.engine import residency

        child_pid = 424242
        monkeypatch.setattr(residency, "_own_pids", lambda: {os.getpid(), child_pid})
        TestNvidiaSmi._smi(
            monkeypatch,
            {
                "--query-gpu=memory.total,memory.used": "24576, 4096\n",
                "--query-compute-apps=pid,used_memory": (
                    f"{os.getpid()}, 300\n{child_pid}, 2000\n999999, 500\n"
                ),
            },
        )
        view = residency.current_gpu_memory()
        assert view is not None and view.hfl_rss == 2300 * 1024**2


class TestUnattributedGpu:
    """In a container nvidia-smi lists processes under IDs HFL never sees
    (PID 23 listed as 1, on Modal): HFL's models were charged twice, once
    inside "in use" and once as residents (planned 3.6 GB, took 2.6 GB)."""

    def test_no_process_of_ours_listed_is_unattributed(self, monkeypatch):
        from hfl.engine import residency

        monkeypatch.setattr(residency, "_own_pids", lambda: {23})
        TestNvidiaSmi._smi(
            monkeypatch,
            {
                "--query-gpu=memory.total,memory.used": "24576, 2048\n",
                "--query-compute-apps=pid,used_memory": "1, 2000\n",
            },
        )
        view = residency.current_gpu_memory()
        assert view is not None and view.attributed is False

    def test_the_planner_then_finds_its_models_inside_in_use(self):
        from hfl.engine.residency import plan_admission

        gib = 1024**3
        ram = MemoryView(total=64 * gib, in_use=8 * gib, hfl_rss=2 * gib)
        # Two resident models of 1 GiB each already on the GPU (3 GiB in use).
        gpu = MemoryView(total=24 * gib, in_use=3 * gib, hfl_rss=0, attributed=False)
        residents = [ResidentView("a", gib, 1.0), ResidentView("b", gib, 2.0)]
        plan = plan_admission(gib // 2, ram, residents, 0.85, gpu=gpu)
        assert plan.fits
        assert plan.gpu_used_after == 3 * gib + gib // 2  # not 5.5 GiB

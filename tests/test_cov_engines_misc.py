# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Engine support modules, their less-travelled paths: the VRAM probes
(NVML, Metal, sysctl, ROCm sysfs), llama-server's child guard (run for
real against short-lived Python children), torch-before-llama.cpp import
ordering, memory snapshots with a GPU and without psutil, the LoRA
fallbacks, the streaming benchmark, the verifier's failure verdicts and
signature outcomes, and the llama.cpp installer's edges.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import signal
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

from hfl.engine import vram

# ======================================================================
# vram
# ======================================================================


def _pynvml(monkeypatch, *, init_error=None, totals=(), query_error=None, shutdown_error=None):
    calls: list[str] = []
    nv = types.ModuleType("pynvml")

    def init():
        calls.append("init")
        if init_error:
            raise init_error

    def count():
        if query_error:
            raise query_error
        return len(totals)

    def shutdown():
        calls.append("shutdown")
        if shutdown_error:
            raise shutdown_error

    nv.nvmlInit = init  # type: ignore[attr-defined]
    nv.nvmlDeviceGetCount = count  # type: ignore[attr-defined]
    nv.nvmlDeviceGetHandleByIndex = lambda i: i  # type: ignore[attr-defined]
    nv.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(total=totals[h])  # type: ignore[attr-defined]
    nv.nvmlShutdown = shutdown  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", nv)
    return calls


class TestVramProbes:
    def test_nvml_sums_every_gpu_and_shuts_down(self, monkeypatch):
        calls = _pynvml(monkeypatch, totals=(24 * 1024**3, 8 * 1024**3))
        assert vram._probe_nvidia() == pytest.approx(32.0)
        assert calls == ["init", "shutdown"]

    def test_nvml_that_will_not_start_finds_nothing(self, monkeypatch):
        calls = _pynvml(monkeypatch, init_error=RuntimeError("driver mismatch"))
        assert vram._probe_nvidia() is None and calls == ["init"]

    def test_nvml_without_memory_or_with_a_failing_query_finds_nothing(self, monkeypatch):
        _pynvml(monkeypatch, totals=(), shutdown_error=RuntimeError("already down"))
        assert vram._probe_nvidia() is None
        calls = _pynvml(monkeypatch, query_error=RuntimeError("gpu fell off the bus"))
        assert vram._probe_nvidia() is None and calls == ["init", "shutdown"]

    def test_nvidia_smi_output_that_is_not_a_number_finds_nothing(self, monkeypatch):
        import hfl.engine.residency as residency

        monkeypatch.setattr(residency, "_nvidia_smi", lambda args: [["[N/A]"]])
        assert vram._probe_nvidia_smi() is None
        monkeypatch.setattr(residency, "_nvidia_smi", lambda args: [[]])
        assert vram._probe_nvidia_smi() is None
        monkeypatch.setattr(residency, "_nvidia_smi", lambda args: [["8192"], ["16384"]])
        assert vram._probe_nvidia_smi() == pytest.approx(24.0)

    def test_metal_is_only_probed_on_macos(self, monkeypatch):
        monkeypatch.setattr(vram.platform, "system", lambda: "Linux")
        assert vram._probe_metal() is None

    @staticmethod
    def _torch(monkeypatch, *, mps=True, recommended=None, has_mps=True):
        torch = types.ModuleType("torch")
        torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
            mps=types.SimpleNamespace(is_available=lambda: mps) if has_mps else None
        )
        torch.mps = types.SimpleNamespace(recommended_max_memory=recommended)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch)

    @staticmethod
    def _sysctl(monkeypatch, value=None, error=None):
        import ctypes
        import ctypes.util

        class _Libc:
            def sysctlbyname(self, name, size_ref, length_ref, new, newlen):
                assert name == b"hw.memsize"
                if value is not None:
                    size_ref._obj.value = value

        def cdll(path):
            if error:
                raise error
            return _Libc()

        monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libc")
        monkeypatch.setattr(ctypes, "CDLL", cdll)

    def test_metal_reports_torchs_recommended_budget(self, monkeypatch):
        monkeypatch.setattr(vram.platform, "system", lambda: "Darwin")
        self._torch(monkeypatch, recommended=lambda: 48 * 1024**3)
        assert vram._probe_metal() == pytest.approx(48.0)

    def test_metal_without_mps_finds_nothing(self, monkeypatch):
        monkeypatch.setattr(vram.platform, "system", lambda: "Darwin")
        self._torch(monkeypatch, has_mps=False)
        assert vram._probe_metal() is None
        self._torch(monkeypatch, mps=False)
        assert vram._probe_metal() is None

    def test_metal_falls_back_to_sysctl(self, monkeypatch):
        monkeypatch.setattr(vram.platform, "system", lambda: "Darwin")

        def broken():
            raise RuntimeError("no Metal device")

        self._torch(monkeypatch, recommended=broken)
        self._sysctl(monkeypatch, value=36 * 1024**3)
        assert vram._probe_metal() == pytest.approx(36.0)
        monkeypatch.setitem(sys.modules, "torch", None)  # no torch at all
        assert vram._probe_metal() == pytest.approx(36.0)

    def test_metal_with_a_silent_or_failing_sysctl_finds_nothing(self, monkeypatch):
        monkeypatch.setattr(vram.platform, "system", lambda: "Darwin")
        self._torch(monkeypatch, recommended=None)
        self._sysctl(monkeypatch, value=None)
        assert vram._probe_metal() is None
        self._sysctl(monkeypatch, error=OSError("no libc"))
        assert vram._probe_metal() is None

    def test_rocm_reads_each_cards_vram(self, monkeypatch, tmp_path):
        drm = tmp_path / "drm"
        for name, content in (("card0", str(16 * 1024**3)), ("card1", "garbage")):
            (drm / name / "device").mkdir(parents=True)
            (drm / name / "device" / "mem_info_vram_total").write_text(content + "\n")
        (drm / "card0-HDMI-A-1").mkdir()  # a connector, not a card
        (drm / "renderD128").mkdir()
        monkeypatch.setattr(vram, "Path", lambda p: drm)
        assert vram._probe_rocm() == pytest.approx(16.0)

    def test_rocm_without_sysfs_or_vram_finds_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vram, "Path", lambda p: tmp_path / "missing")
        assert vram._probe_rocm() is None
        (tmp_path / "empty").mkdir()
        monkeypatch.setattr(vram, "Path", lambda p: tmp_path / "empty")
        assert vram._probe_rocm() is None

    def test_an_unreadable_drm_folder_yields_no_cards(self):
        class _Root:
            def iterdir(self):
                raise PermissionError("denied")

        assert list(vram._iter_rocm_cards(_Root())) == []  # type: ignore[arg-type]

    def test_a_negative_budget_takes_the_floor_tier(self):
        tier = vram.pick_ctx_size(-1.0)
        assert tier.ctx == vram.CTX_TIERS[-1][1] and tier.vram_gib == -1.0


# ======================================================================
# _child_guard
# ======================================================================


@pytest.fixture
def guard(monkeypatch):
    """The guard module, with the signal handlers it installs restored."""
    from hfl.engine import _child_guard

    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield _child_guard
    for s, handler in saved.items():
        signal.signal(s, handler)


PY = sys.executable


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
class TestChildGuard:
    def test_a_bad_command_line_prints_the_usage(self, guard, capsys):
        assert guard.main(["guard", "1"]) == 2
        assert guard.main(["guard", "1", "x", "cmd"]) == 2
        assert "usage: _child_guard" in capsys.readouterr().err

    def test_the_childs_exit_code_is_the_guards(self, guard):
        argv = ["guard", str(os.getppid()), "--", PY, "-c", "raise SystemExit(3)"]
        assert guard.main(argv) == 3

    def test_a_child_killed_by_sigkill_is_not_reported_as_a_clean_exit(self, guard):
        # The OOM killer's way: it used to come out as code 0.
        code = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
        argv = ["guard", str(os.getppid()), "--", PY, "-c", code]
        assert guard.main(argv) == 128 + signal.SIGKILL

    def test_a_parent_that_is_gone_stops_the_child(self, guard, monkeypatch):
        started: list = []
        real_popen = subprocess.Popen

        def popen(args):
            started.append(real_popen(args))
            return started[-1]

        monkeypatch.setattr(guard.subprocess, "Popen", popen)
        monkeypatch.setattr(guard, "_parent_gone", lambda parent, handle: True)
        argv = ["guard", "1", "--", PY, "-c", "import time; time.sleep(30)"]
        assert guard.main(argv) == 128 + signal.SIGTERM  # as a shell reports it
        assert started[0].returncode == -signal.SIGTERM

    def test_sigterm_is_forwarded_to_the_child(self, guard, monkeypatch):
        forwarded: list = []

        def parent_gone(parent, handle):
            handler = signal.getsignal(signal.SIGTERM)
            forwarded.append(handler)
            handler(signal.SIGTERM, None)  # as if HFL stopped the guard
            return False

        monkeypatch.setattr(guard, "_parent_gone", parent_gone)
        argv = ["guard", "1", "--", PY, "-c", "import time; time.sleep(30)"]
        assert guard.main(argv) == 128 + signal.SIGTERM
        # After the child is gone a late signal is not sent anywhere.
        forwarded[0](signal.SIGINT, None)

    def test_parent_liveness(self, guard, monkeypatch):
        ppid = os.getppid()
        assert guard._parent_gone(ppid) is False
        assert guard._parent_gone(ppid + 100000) is True  # re-parented

        def lookup(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(guard.os, "kill", lookup)
        assert guard._parent_gone(ppid) is True

        def denied(pid, sig):
            raise PermissionError

        monkeypatch.setattr(guard.os, "kill", denied)
        assert guard._parent_gone(ppid) is False  # alive, someone else's

    def test_stop_kills_a_child_that_ignores_terminate(self, guard):
        class _Child:
            def __init__(self, running):
                self.running, self.actions = running, []

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.actions.append("terminate")

            def kill(self):
                self.actions.append("kill")

            def wait(self, timeout):
                self.actions.append("wait")
                if self.actions.count("wait") == 1:
                    raise subprocess.TimeoutExpired("llama-server", timeout)
                return -9

        stubborn = _Child(running=True)
        guard._stop(stubborn)
        assert stubborn.actions == ["terminate", "wait", "kill", "wait"]
        done = _Child(running=False)
        guard._stop(done)
        assert done.actions == []


def test_the_windows_watch_uses_a_process_handle(monkeypatch):
    """The Windows liveness check, its kernel32 calls replaced."""
    import ctypes

    from hfl.engine import _child_guard

    state = {"open": 77, "wait": 0, "opened": []}

    class _OpenProcess:
        restype = None

        def __call__(self, access, inherit, pid):
            state["opened"].append((access, inherit, pid))
            return state["open"]

    class _Kernel32:
        OpenProcess = _OpenProcess()

        def WaitForSingleObject(self, handle, ms):  # noqa: N802
            assert ms == 0 and handle.value == 77
            return state["wait"]

    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error: _Kernel32(), raising=False)
    assert _child_guard._windows_watch(4242) == 77
    assert state["opened"] == [(0x00100000, False, 4242)]  # SYNCHRONIZE only
    state["open"] = None
    assert _child_guard._windows_watch(4242) == 0  # already gone
    assert _child_guard._windows_parent_gone(0) is True
    assert _child_guard._windows_parent_gone(77) is True  # WAIT_OBJECT_0: exited
    state["wait"] = 258  # WAIT_TIMEOUT: still running
    assert _child_guard._windows_parent_gone(77) is False


# ======================================================================
# native_order
# ======================================================================


@pytest.fixture
def linux_without_torch(monkeypatch):
    from hfl.engine import native_order

    monkeypatch.setattr(native_order.sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    return native_order


def _specs(monkeypatch, **specs):
    def find_spec(name, *a):
        value = specs.get(name)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


class TestNativeOrder:
    def test_torch_goes_first_for_a_cuda_llama_cpp(
        self, linux_without_torch, monkeypatch, tmp_path
    ):
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "libggml-cuda.so").write_bytes(b"")
        llama = types.SimpleNamespace(submodule_search_locations=[str(tmp_path)])
        _specs(monkeypatch, torch=object(), llama_cpp=llama)
        assert linux_without_torch._torch_goes_first() is True

    def test_not_for_a_cpu_build_nor_without_torch_or_llama_cpp(
        self, linux_without_torch, monkeypatch, tmp_path
    ):
        (tmp_path / "lib").mkdir()
        cpu = types.SimpleNamespace(submodule_search_locations=[str(tmp_path)])
        _specs(monkeypatch, torch=object(), llama_cpp=cpu)
        assert linux_without_torch._torch_goes_first() is False
        _specs(monkeypatch, torch=None, llama_cpp=cpu)
        assert linux_without_torch._torch_goes_first() is False
        _specs(monkeypatch, torch=object(), llama_cpp=None)
        assert linux_without_torch._torch_goes_first() is False
        flat = types.SimpleNamespace(submodule_search_locations=None)
        _specs(monkeypatch, torch=object(), llama_cpp=flat)
        assert linux_without_torch._torch_goes_first() is False
        _specs(monkeypatch, torch=ValueError("torch.__spec__ is None"))
        assert linux_without_torch._torch_goes_first() is False

    def test_the_hook_imports_torch_once_then_steps_aside(self, monkeypatch):
        from hfl.engine import native_order

        hook = native_order._TorchFirst()
        monkeypatch.setattr(sys, "meta_path", [hook, *sys.meta_path])
        monkeypatch.setattr(native_order, "_torch_goes_first", lambda: True)
        fake_torch = types.ModuleType("torch")
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert hook.find_spec("numpy", None) is None and hook in sys.meta_path
        assert hook.find_spec("llama_cpp", None) is None
        assert hook not in sys.meta_path
        # A broken torch must not stop llama.cpp; the hook is already gone.
        monkeypatch.setitem(sys.modules, "torch", None)
        assert hook.find_spec("llama_cpp", None) is None

    def test_install_is_idempotent_and_skipped_once_llama_cpp_is_in(self, monkeypatch):
        from hfl.engine import native_order

        others = [m for m in sys.meta_path if not isinstance(m, native_order._TorchFirst)]

        def hooks() -> int:
            return sum(isinstance(m, native_order._TorchFirst) for m in sys.meta_path)

        monkeypatch.setattr(sys, "meta_path", list(others))
        monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
        native_order.install_torch_first()
        native_order.install_torch_first()
        assert hooks() == 1 and isinstance(sys.meta_path[0], native_order._TorchFirst)
        monkeypatch.setattr(sys, "meta_path", list(others))
        monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
        native_order.install_torch_first()
        assert hooks() == 0  # llama.cpp already imported: too late to matter


# ======================================================================
# memory
# ======================================================================


def _gputil(gpus=None, error=None):
    module = types.ModuleType("GPUtil")

    def get_gpus():
        if error:
            raise error
        return gpus or []

    module.getGPUs = get_gpus  # type: ignore[attr-defined]
    return module


class TestMemory:
    def test_a_gpu_is_measured_through_gputil(self, monkeypatch):
        from hfl.engine import memory

        gpu = types.SimpleNamespace(memoryUsed=2048, memoryTotal=8192)
        monkeypatch.setattr(memory, "HAS_GPUTIL", True)
        monkeypatch.setattr(memory, "GPUtil", _gputil([gpu]), raising=False)
        snap = memory.get_memory_snapshot(0)
        assert (snap.gpu_used_gb, snap.gpu_total_gb, snap.gpu_available_gb) == (2.0, 8.0, 6.0)
        assert snap.gpu_id == 0 and snap.gpu_percent == pytest.approx(25.0)
        assert "GPU 0 VRAM: 2.0/8.0 GB (25.0% used)" in memory.format_memory_info()
        assert memory.get_memory_snapshot(3).gpu_total_gb is None  # no such GPU

    def test_gpu_preferred_but_too_small_falls_back_to_system_ram(self, monkeypatch):
        from hfl.engine import memory

        monkeypatch.setattr(memory, "get_available_memory", lambda: (64.0, 4.0))
        assert memory.check_memory_available(16.0) is True
        assert memory.check_memory_available(100.0) is False

    def test_a_failing_gpu_query_is_no_gpu(self, monkeypatch):
        from hfl.engine import memory

        monkeypatch.setattr(memory, "HAS_GPUTIL", True)
        monkeypatch.setattr(memory, "GPUtil", _gputil(error=OSError("nvidia-smi")), raising=False)
        snap = memory.get_memory_snapshot()
        assert snap.gpu_total_gb is None and snap.gpu_percent is None

    def test_partial_gpu_figures_and_an_unfinished_tracker(self):
        from hfl.engine import memory

        snap = memory.MemorySnapshot(1.0, 1.0, 2.0, gpu_used_gb=None, gpu_total_gb=8.0)
        assert snap.gpu_percent is None
        tracker = memory.MemoryTracker()
        assert tracker.gpu_delta_gb is None and tracker.memory_used_gb == 0.0

    def test_without_psutil_system_memory_reads_zero(self, monkeypatch):
        """The module as imported where psutil is missing and GPUtil is not."""
        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setitem(sys.modules, "GPUtil", _gputil())
        path = Path(importlib.util.find_spec("hfl.engine.memory").origin)
        spec = importlib.util.spec_from_file_location("hfl_memory_without_psutil", path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses look it up
        spec.loader.exec_module(module)
        assert module.HAS_PSUTIL is False and module.HAS_GPUTIL is True
        snap = module.get_memory_snapshot()
        assert (snap.system_used_gb, snap.system_total_gb, snap.system_percent) == (0.0, 0.0, 0.0)


# ======================================================================
# lora fallbacks
# ======================================================================


class TestLoraFallbacks:
    def test_the_old_apply_from_file_api(self):
        from hfl.engine import lora

        applied: list = []
        inner = types.SimpleNamespace(apply_lora_from_file=lambda p: applied.append(p))
        lora._set_lora(types.SimpleNamespace(_model=inner), "/a.gguf", 1.0)
        assert applied == ["/a.gguf"]

    def test_a_model_without_any_lora_api_is_refused(self):
        from hfl.engine import lora

        engine = types.SimpleNamespace(_model=types.SimpleNamespace())
        with pytest.raises(RuntimeError, match="hot-swap"):
            lora._set_lora(engine, "/a.gguf", 1.0)
        with pytest.raises(RuntimeError, match="removal"):
            lora._unset_lora(engine, "a1")

    def test_removal_through_the_older_unload_alias(self):
        from hfl.engine import lora

        removed: list = []
        inner = types.SimpleNamespace(remove_lora_adapter=None, unload_lora=removed.append)
        lora._unset_lora(types.SimpleNamespace(_model=inner), "a1")
        assert removed == ["a1"]

    def test_get_registry_keeps_the_one_another_thread_built(self, monkeypatch):
        from hfl.engine import lora

        built = lora.LoraRegistry()

        @contextlib.contextmanager
        def racing_lock():
            lora._GLOBAL = built  # another thread won the race meanwhile
            yield

        lora.reset_registry()
        monkeypatch.setattr(lora, "_GLOBAL_LOCK", racing_lock())
        try:
            assert lora.get_registry() is built
        finally:
            monkeypatch.undo()
            lora.reset_registry()


# ======================================================================
# benchmark
# ======================================================================


class TestBenchmarkStream:
    def test_a_streaming_engine_reports_ttft(self):
        from hfl.engine import benchmark

        class _Engine:
            def generate_stream(self, prompt, cfg):
                assert cfg.max_tokens == 3 and cfg.temperature == 0.0
                yield from ["a", "b", "c"]

        run = benchmark._measure_one(_Engine(), "hello", 3)
        assert run.measurement_mode == "stream" and run.tokens_generated == 3
        assert run.ttft_ms is not None and run.ttft_ms <= run.total_ms

    def test_an_empty_stream_has_no_ttft(self):
        from hfl.engine import benchmark

        class _Engine:
            def generate_stream(self, prompt, cfg):
                return iter(())

        run = benchmark._measure_one(_Engine(), "hello", 3)
        assert run.ttft_ms is None and run.tokens_generated == 0

    def test_percentiles_over_streamed_runs(self):
        from hfl.engine import benchmark

        runs = [
            benchmark.BenchmarkRun(10, 5, ttft, total, 50.0, "stream")
            for ttft, total in ((10.0, 100.0), (30.0, 300.0), (20.0, 200.0))
        ]
        summary = benchmark._summarise(10, runs)
        assert summary.ttft_p50_ms == 20.0 and summary.ttft_p95_ms == 20.0
        assert summary.measurement_mode == "stream"
        one = benchmark._summarise(10, runs[:1])
        assert one.ttft_p95_ms == 10.0


# ======================================================================
# verifier
# ======================================================================


class _Manifest:
    def __init__(self, **fields):
        self.name = "m"
        self.format = "gguf"
        self.declared_capabilities: list = []
        self.__dict__.update(fields)


class TestVerifierVerdicts:
    def test_a_tokenizer_that_raises_fails_the_round_trip(self):
        from hfl.engine.verifier import _check_tokenizer_round_trip

        class _Tok:
            def encode(self, text):
                raise ValueError("bad vocab")

        check = _check_tokenizer_round_trip(types.SimpleNamespace(tokenizer=_Tok()))
        assert not check.passed and "bad vocab" in check.detail

    def test_chat_template_verdicts(self):
        from hfl.engine.verifier import _check_chat_template

        audio = _check_chat_template(types.SimpleNamespace(), _Manifest(format="audio"))
        assert audio.passed and audio.detail == "not applicable"

        empty = types.SimpleNamespace(apply_chat_template=lambda *a, **k: "  ")
        check = _check_chat_template(types.SimpleNamespace(tokenizer=empty), _Manifest())
        assert not check.passed and check.detail == "empty render"

        def boom(*a, **k):
            raise ValueError("no chat_template")

        broken = types.SimpleNamespace(apply_chat_template=boom)
        check = _check_chat_template(types.SimpleNamespace(tokenizer=broken), _Manifest())
        assert not check.passed and "no chat_template" in check.detail

    def test_a_tool_parser_that_raises_fails(self, monkeypatch):
        import hfl.api.tool_parsers as tool_parsers
        from hfl.engine.verifier import _check_tool_parser

        def boom(*a, **k):
            raise RuntimeError("parser crashed")

        monkeypatch.setattr(tool_parsers, "dispatch", boom)
        check = _check_tool_parser(_Manifest())
        assert not check.passed and "parser crashed" in check.detail

    def test_embedding_verdicts(self):
        from hfl.engine.verifier import _check_embedding_dim

        manifest = _Manifest(declared_capabilities=["embeddings"])
        check = _check_embedding_dim(types.SimpleNamespace(), manifest)
        assert not check.passed and check.detail == "no embed() method"

        def boom(texts):
            raise RuntimeError("OOM")

        check = _check_embedding_dim(types.SimpleNamespace(embed=boom), manifest)
        assert not check.passed and "OOM" in check.detail
        check = _check_embedding_dim(types.SimpleNamespace(embed=lambda t: [0.1, 0.2]), manifest)
        assert not check.passed and "unexpected shape" in check.detail


class TestVerifierSignature:
    SIGNED = {"signature": {"key_id": "hfl-release", "sig": "..."}}

    def test_signed_without_a_trust_root_is_skipped(self, temp_config):
        from hfl.engine.verifier import _check_signature

        check = _check_signature(_Manifest(**self.SIGNED))
        assert check.passed and check.skipped and "no trust root" in check.detail

    @pytest.fixture
    def trust(self, temp_config, monkeypatch):
        import hfl.observability.signing as signing

        (Path(temp_config.home_dir) / "trusted-publishers.json").write_text("{}")
        outcome: dict = {}

        def verify(envelope, trust_root):
            assert trust_root == "ROOT"
            if "raise" in outcome:
                raise outcome["raise"]
            return outcome["value"]

        monkeypatch.setattr(signing.TrustRoot, "load", staticmethod(lambda path: "ROOT"))
        monkeypatch.setattr(signing, "verify_manifest_envelope", verify)
        return outcome, signing

    def test_a_trusted_signature_passes_naming_the_key(self, trust):
        from hfl.engine.verifier import _check_signature

        outcome, _ = trust
        outcome["value"] = True
        check = _check_signature(_Manifest(**self.SIGNED))
        assert check.passed and not check.skipped and "'hfl-release'" in check.detail

    def test_a_signature_rejected_without_an_error_still_fails(self, trust):
        from hfl.engine.verifier import _check_signature

        outcome, _ = trust
        outcome["value"] = False
        check = _check_signature(_Manifest(**self.SIGNED))
        assert not check.passed and "rejected without an error" in check.detail

    def test_no_ed25519_backend_is_skipped_not_failed(self, trust):
        from hfl.engine.verifier import _check_signature

        outcome, signing = trust
        outcome["raise"] = signing.SignatureUnavailableError("install cryptography")
        check = _check_signature(_Manifest(**self.SIGNED))
        assert check.passed and check.skipped and "no ed25519 backend" in check.detail

    def test_an_invalid_signature_fails(self, trust):
        from hfl.engine.verifier import _check_signature

        outcome, signing = trust
        outcome["raise"] = signing.SignatureInvalidError("digest mismatch")
        check = _check_signature(_Manifest(**self.SIGNED))
        assert not check.passed and "digest mismatch" in check.detail


# ======================================================================
# llama_server_dist
# ======================================================================

FAKE_SERVER = "#!/bin/sh\necho 'version: 1 (build 10964, commit x)'\n"


def _tar(path: Path, members: dict[str, bytes], links=(), dirs=()) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name in dirs:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            tf.addfile(info)
    return path


@pytest.fixture
def offline_dist(monkeypatch):
    from hfl.engine import llama_server_dist as dist

    def setup(archive: Path):
        key = (*dist.platform_key(), "test")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        monkeypatch.setitem(dist.ASSETS, key, [(archive.name, 1, digest)])

        def fetch(name, sha256, dest, progress):
            (dest / name).write_bytes(archive.read_bytes())
            return dest / name

        monkeypatch.setattr(dist, "_download", fetch)

    return dist, setup


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX test programs")
class TestLlamaServerDist:
    def test_an_archive_without_a_server_installs_nothing(self, offline_dist, tmp_path):
        dist, setup = offline_dist
        setup(_tar(tmp_path / "a.tar.gz", {"x/llama-quantize": b"#!/bin/sh\n"}))
        target = tmp_path / "home" / "llama.cpp"
        with pytest.raises(dist.InstallError, match="holds no llama-server"):
            dist.install("test", target=target)
        assert not target.exists() and list(target.parent.iterdir()) == []

    def test_a_reinstall_replaces_the_previous_one(self, offline_dist, tmp_path):
        dist, setup = offline_dist
        archive = _tar(
            tmp_path / "a.tar.gz",
            {"x/llama-server": FAKE_SERVER.encode(), "x/libggml.dylib": b"real"},
            links=[("y/libggml.dylib", "libggml.0.dylib")],  # same name as a file: kept a file
            dirs=["x/"],
        )
        setup(archive)
        target = tmp_path / "home" / "llama.cpp"
        target.mkdir(parents=True)
        (target / "stale").write_text("old build")
        server = dist.install("test", target=target)
        assert server == target / "llama-server" and os.access(server, os.X_OK)
        assert not (target / "stale").exists()  # the old install is gone
        assert not (target / "llama-quantize").exists()  # not in this archive
        lib = target / "libggml.dylib"
        assert not lib.is_symlink() and lib.read_bytes() == b"real"

    def test_a_tar_member_without_content_is_skipped(self, monkeypatch, tmp_path):
        from hfl.engine import llama_server_dist as dist

        archive = _tar(tmp_path / "a.tar.gz", {"x/llama-server": b"bin"})
        monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda self, member: None)
        out = tmp_path / "out"
        out.mkdir()
        dist._unpack(archive, out)
        assert list(out.iterdir()) == []

    def test_a_zip_keeps_only_our_files(self, tmp_path):
        from hfl.engine import llama_server_dist as dist

        archive = tmp_path / "llama-bin-win.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("build/bin/llama-server.exe", b"MZ")
            zf.writestr("build/bin/ggml.dll", b"dll")
            zf.writestr("build/bin/README.md", b"docs")
            zf.writestr("build/bin/.hidden", b"x")
            zf.writestr("build/bin/llama-server.exe/", b"")  # a folder named like ours
        out = tmp_path / "out"
        out.mkdir()
        dist._unpack(archive, out)
        assert sorted(p.name for p in out.iterdir()) == ["ggml.dll", "llama-server.exe"]
        assert (out / "llama-server.exe").read_bytes() == b"MZ"

    def test_kept_names(self):
        from hfl.engine import llama_server_dist as dist

        assert dist._kept("dir/") is None and dist._kept("a/.DS_Store") is None
        assert dist._kept("a\\b\\LICENSE-curl") == "LICENSE-curl"

    def test_a_platform_without_builds_is_refused(self, monkeypatch):
        from hfl.engine import llama_server_dist as dist

        monkeypatch.setattr(dist, "platform_key", lambda: ("plan9", "mips"))
        with pytest.raises(dist.InstallError, match="publishes no  build for plan9/mips"):
            dist.install()
        with pytest.raises(dist.InstallError, match=r"builds here: none"):
            dist.install("cuda")

    def test_a_server_that_cannot_start_is_reported(self, tmp_path):
        from hfl.engine import llama_server_dist as dist

        with pytest.raises(dist.InstallError, match="does not start here"):
            dist.check(tmp_path / "missing-llama-server")

    def test_download_reports_progress_and_http_errors(self, monkeypatch, tmp_path):
        import httpx

        from hfl.engine import llama_server_dist as dist

        chunks = [b"ab", b"cde"]

        class _Response:
            def raise_for_status(self):
                pass

            def iter_bytes(self, size):
                yield from chunks

        @contextlib.contextmanager
        def stream(method, url, **kwargs):
            yield _Response()

        monkeypatch.setattr(httpx, "stream", stream)
        seen: list[int] = []
        digest = hashlib.sha256(b"abcde").hexdigest()
        assert dist._download("x.tgz", digest, tmp_path, seen.append).read_bytes() == b"abcde"
        assert seen == [2, 3]

        def offline(method, url, **kwargs):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(httpx, "stream", offline)
        with pytest.raises(dist.InstallError, match="could not download .*no route to host"):
            dist._download("x.tgz", digest, tmp_path, None)

    def test_a_bundle_that_cannot_be_made_executable_is_not_offered(self, monkeypatch, tmp_path):
        from hfl.engine import llama_server_dist as dist

        folder = tmp_path / "llama.cpp"
        folder.mkdir()
        (folder / "llama-server").write_text("#!/bin/sh\n")
        (folder / "llama-server").chmod(0o644)

        def readonly(self, mode):
            raise PermissionError("read-only bundle")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        monkeypatch.setattr(Path, "chmod", readonly)
        assert dist.bundled_binary() is None

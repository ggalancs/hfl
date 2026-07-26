# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the acceleration-reporting layer.

Context: a user on an M3 Max reported that HFL "wasn't using hardware
acceleration". It was — 41/41 layers on Metal, 393 tok/s prefill — but every
trace of that fact went to /dev/null, because ``load()`` wrapped the Llama
constructor in a stderr suppressor. The load looked identical whether it ran
on the GPU or the CPU.

The actual cause of the slowness was the laptop running on battery (8x slower
generation, measured). Both halves are covered here: the summary that proves
acceleration happened, and the power-source probe that explains the speed.
"""

from __future__ import annotations

import subprocess

import pytest

from hfl.engine import llama_cpp as engine_module
from hfl.engine import power as power_module

# Verbatim excerpt of what llama.cpp emits on this machine.
_METAL_LOG = """
ggml_metal_device_init: GPU name:   MTL0 (Apple M3 Max)
ggml_metal_device_init: GPU family: MTLGPUFamilyApple9  (1009)
ggml_metal_device_init: has unified memory    = true
load_tensors: layer   0 assigned to device MTL0, is_swa = 0
load_tensors: offloaded 41/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   417.30 MiB
load_tensors:  MTL0_Mapped model buffer size =  8579.06 MiB
llama_kv_cache:       MTL0 KV buffer size =   640.00 MiB
"""

_CUDA_LOG = """
load_tensors: offloaded 33/33 layers to GPU
load_tensors:        CPU model buffer size =   102.50 MiB
load_tensors:      CUDA0 model buffer size =  4095.00 MiB
"""

_CPU_ONLY_LOG = """
load_tensors:   CPU_Mapped model buffer size =  8996.36 MiB
llama_context: n_ctx = 4096
"""


class TestAccelerationSummary:
    """``_summarize_acceleration`` turns llama.cpp's loader dump into the one
    line a user needs to answer "is this on the GPU?"."""

    def test_metal_summary_names_device_layers_and_size(self):
        summary = engine_module._summarize_acceleration(_METAL_LOG)
        assert summary is not None
        assert "Apple M3 Max" in summary
        assert "41/41 layers on GPU" in summary
        # 8579.06 MiB of weights on the device -> 8.4 GiB. The CPU buffer
        # (417 MiB) must NOT be counted: the point of the number is how much
        # landed on the accelerator.
        assert "8.4 GiB" in summary
        assert "MTL0" in summary

    def test_cuda_log_is_summarized_too(self):
        """The parser keys off the device prefix, not off Metal specifically,
        so a CUDA host gets the same treatment."""
        summary = engine_module._summarize_acceleration(_CUDA_LOG)
        assert summary is not None
        assert "33/33 layers on GPU" in summary
        assert "CUDA0" in summary
        assert "4.0 GiB" in summary

    def test_cpu_only_load_reports_nothing(self):
        """No offload line and no device buffer -> no claim of acceleration."""
        assert engine_module._summarize_acceleration(_CPU_ONLY_LOG) is None

    def test_empty_log_is_not_an_error(self):
        assert engine_module._summarize_acceleration("") is None

    def test_partial_offload_is_reported_honestly(self):
        """A model too big for VRAM gets some layers on CPU. The summary must
        show the ratio rather than implying a full offload."""
        summary = engine_module._summarize_acceleration(
            "load_tensors: offloaded 20/41 layers to GPU\n"
            "load_tensors:      CUDA0 model buffer size =  2000.00 MiB\n"
        )
        assert summary is not None
        assert "20/41 layers on GPU" in summary


class TestEngineExposesAcceleration:
    def test_base_engine_defaults_to_unknown(self):
        """Backends that don't report it must say "unknown" (None), never
        claim CPU — MLX and vLLM accelerate without emitting these lines."""
        from hfl.engine.base import InferenceEngine

        assert InferenceEngine.acceleration.fget(object()) is None  # type: ignore[attr-defined]

    def test_llama_engine_starts_unknown_and_clears_on_unload(self):
        engine = engine_module.LlamaCppEngine()
        assert engine.acceleration is None
        engine._acceleration = "MTL0 · 41/41 layers on GPU"
        assert engine.acceleration == "MTL0 · 41/41 layers on GPU"


class TestPowerSource:
    """The battery probe. Advisory only: it must never raise, and must never
    guess on platforms where the question doesn't apply."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        power_module._PROBED = False
        power_module._CACHED = None
        yield
        power_module._PROBED = False
        power_module._CACHED = None

    def _fake_pmset(self, monkeypatch, stdout: str, returncode: int = 0):
        def _run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout)

        monkeypatch.setattr(power_module.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(power_module.subprocess, "run", _run)

    def test_detects_battery(self, monkeypatch):
        self._fake_pmset(monkeypatch, "Now drawing from 'Battery Power'\n -InternalBattery-0 86%")
        assert power_module.on_battery() is True
        assert power_module.power_source_label() == "battery"

    def test_detects_ac(self, monkeypatch):
        self._fake_pmset(monkeypatch, "Now drawing from 'AC Power'\n -InternalBattery-0 91%")
        assert power_module.on_battery() is False
        assert power_module.power_source_label() == "AC power"

    def test_non_darwin_is_unknown(self, monkeypatch):
        monkeypatch.setattr(power_module.platform, "system", lambda: "Linux")
        assert power_module.on_battery() is None
        assert power_module.power_source_label() == "unknown"

    def test_pmset_failure_is_unknown_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(power_module.platform, "system", lambda: "Darwin")

        def _boom(*args, **kwargs):
            raise OSError("pmset missing")

        monkeypatch.setattr(power_module.subprocess, "run", _boom)
        assert power_module.on_battery() is None

    def test_timeout_is_unknown_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(power_module.platform, "system", lambda: "Darwin")

        def _timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="pmset", timeout=2)

        monkeypatch.setattr(power_module.subprocess, "run", _timeout)
        assert power_module.on_battery() is None

    def test_result_is_cached(self, monkeypatch):
        calls = {"n": 0}

        def _run(*args, **kwargs):
            calls["n"] += 1
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="Now drawing from 'AC Power'"
            )

        monkeypatch.setattr(power_module.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(power_module.subprocess, "run", _run)

        power_module.on_battery()
        power_module.on_battery()
        power_module.on_battery()
        assert calls["n"] == 1, "pmset must not be forked once per model load"


class TestPsExposesAcceleration:
    """``/api/ps`` should answer "is it on the GPU, at what context?" without
    the caller having to read the server log."""

    def test_details_carry_acceleration_and_context(self):
        from hfl.api.routes_ps import _manifest_details

        class _Manifest:
            format = "gguf"
            architecture = "qwen2"
            parameters = "72B"
            quantization = "Q4_K_M"

        class _Engine:
            acceleration = "MTL0 (Apple M3 Max) · 81/81 layers on GPU"
            context_size = 16384

        details = _manifest_details(_Manifest(), _Engine())
        assert details["acceleration"] == "MTL0 (Apple M3 Max) · 81/81 layers on GPU"
        assert details["context_size"] == 16384
        # Ollama's own keys must survive untouched — clients key off them.
        assert details["format"] == "gguf"
        assert details["quantization_level"] == "Q4_K_M"

    def test_details_omit_unknown_fields(self):
        """A backend that reports nothing must not add empty keys that a
        client would render as "acceleration: none"."""
        from hfl.api.routes_ps import _manifest_details

        class _Manifest:
            format = "gguf"
            architecture = None
            parameters = None
            quantization = None

        class _Engine:
            acceleration = None
            context_size = 0

        details = _manifest_details(_Manifest(), _Engine())
        assert "acceleration" not in details
        assert "context_size" not in details

    def test_details_without_an_engine_still_work(self):
        from hfl.api.routes_ps import _manifest_details

        class _Manifest:
            format = "gguf"
            architecture = "qwen2"
            parameters = "72B"
            quantization = "Q4_K_M"

        details = _manifest_details(_Manifest())
        assert details["family"] == "qwen2"
        assert "acceleration" not in details

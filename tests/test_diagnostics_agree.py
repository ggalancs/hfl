# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""`hfl check`, `hfl debug` and `hfl doctor` report the same machine.

Each answers its own question — is everything installed / what to paste
into a bug report / why is my GPU not used — but they probed backends and
accelerators with two separate pieces of code that drifted apart (the same
Mac was "MPS" in one and "Apple Metal" in another, and none looked for the
llama-server that `hfl serve --parallel` needs). They now read one probe.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest
from typer.testing import CliRunner

from hfl.cli.commands import doctor


@pytest.fixture
def machine(monkeypatch):
    report = doctor.DoctorReport(
        python_version="3.12.0",
        platform_system="Linux",
        platform_machine="x86_64",
        llama_cpp_available=True,
        llama_cpp_build_features={"gpu_offload": True},
        llama_server=None,
        nvidia_devices=["NVIDIA GeForce RTX 4090"],
        transformers_available=True,
    )
    monkeypatch.setattr(doctor, "build_report", lambda: report)
    return report


def _run(command: str) -> str:
    from hfl.cli.main import app

    result = CliRunner().invoke(app, [command], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    return result.output


@pytest.mark.parametrize("command", ["check", "debug", "doctor"])
def test_all_three_show_the_same_accelerator_and_llama_server(machine, command, temp_config):
    out = _run(command)
    assert "NVIDIA" in out and "RTX 4090" in out
    assert "llama-server" in out
    assert "MPS" not in out  # the old torch-only wording of the other probe


def test_check_and_doctor_list_the_same_backends(machine, temp_config):
    check, report = _run("check"), _run("doctor")
    for name, _ok, _detail in doctor.backend_rows(machine):
        assert name in check and name in report


def test_the_recommendation_names_the_fix(monkeypatch):
    monkeypatch.setattr("hfl.engine.llama_server.binary", lambda: None)
    real = doctor.build_report()
    assert any("hfl serve --parallel" in r for r in real.recommendations)


def test_asking_llama_cpp_about_the_gpu_prints_nothing(monkeypatch, capfd):
    """The GPU query initialises the backend, which wrote ~20 ggml_metal_*
    lines straight to fd 2 before the report."""
    fake = ModuleType("llama_cpp")

    def llama_supports_gpu_offload() -> bool:
        os.write(2, b"ggml_metal_device_init: noise\n")
        return True

    fake.llama_supports_gpu_offload = llama_supports_gpu_offload
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    assert doctor._probe_llama_cpp() == (True, {"gpu_offload": True})
    assert "ggml_metal" not in capfd.readouterr().err


def test_nvidia_is_found_through_torch_without_pynvml(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = type(
        "cuda",
        (),
        {
            "is_available": staticmethod(lambda: True),
            "device_count": staticmethod(lambda: 1),
            "get_device_name": staticmethod(lambda i: "NVIDIA A100"),
        },
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import fails
    assert doctor._probe_nvidia() == ["NVIDIA A100"]

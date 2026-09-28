# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""How a model spans several GPUs: HFL_TENSOR_SPLIT, HFL_MAIN_GPU,
HFL_SPLIT_MODE (llama.cpp, llama-server). Not checked on multi-GPU
hardware; checked here: each backend gets the options as it names them
(the real llama-server rejects an invalid --split-mode / --tensor-split,
so these are parsed), and a malformed value stops with its name."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hfl.exceptions import InvalidConfigError


def _cfg(**over):
    base = {"gpu_tensor_split": None, "gpu_main": None, "gpu_split_mode": None}
    return SimpleNamespace(**{**base, **over})


def test_nothing_set_passes_nothing() -> None:
    from hfl.engine.llama_cpp import multi_gpu_kwargs

    assert multi_gpu_kwargs(_cfg()) == {}


def test_llama_cpp_gets_its_own_names_and_enum() -> None:
    from hfl.engine.llama_cpp import multi_gpu_kwargs

    got = multi_gpu_kwargs(_cfg(gpu_tensor_split=[3.0, 1.0], gpu_main=1, gpu_split_mode="row"))
    assert got == {"tensor_split": [3.0, 1.0], "main_gpu": 1, "split_mode": 2}


def test_llama_server_gets_its_flags(monkeypatch) -> None:
    from hfl.config import config
    from hfl.engine.llama_server import _multi_gpu_args

    monkeypatch.setattr(config, "gpu_tensor_split", [3.0, 1.0])
    monkeypatch.setattr(config, "gpu_main", 0)
    monkeypatch.setattr(config, "gpu_split_mode", "layer")
    assert _multi_gpu_args() == [
        "--tensor-split", "3,1", "--main-gpu", "0", "--split-mode", "layer",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "value"),
    [("HFL_TENSOR_SPLIT", "abc"), ("HFL_TENSOR_SPLIT", "0,0"), ("HFL_SPLIT_MODE", "bogus")],
)
def test_a_malformed_value_names_the_variable(monkeypatch, name, value) -> None:
    from hfl.config import HFLConfig

    monkeypatch.setenv(name, value)
    with pytest.raises(InvalidConfigError) as caught:
        HFLConfig()
    assert name in str(caught.value)

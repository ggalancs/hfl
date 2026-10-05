# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Repeated MLX loads do not grow HFL's memory.

An 8-hour soak that swapped an MLX model in and out 7,600 times grew HFL
from 1.5 GB to 9 GB. Two causes, both measured outside HFL: MLX's global
random state, split lazily by every layer's initial weights and never
evaluated (0.54 MB per build, without end), and transformers' tokenizer,
whose native memory grows ~0.4-0.7 GB over the first loads. HFL evaluates
the random state after each load and keeps each model's tokenizer.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest

from hfl.engine import mlx_engine


@pytest.fixture
def fake_utils(monkeypatch):
    built: list[tuple[str, object]] = []
    loaded: list[tuple[str, str | None]] = []
    utils = ModuleType("mlx_lm.utils")

    def load_model(path):
        loaded.append((str(path), None))
        return object(), {"eos_token_id": [1, 2]}

    def load_adapters(model, adapter):
        loaded.append(("adapter", adapter))

        class Adapted:
            def eval(self):
                return self

        return Adapted()

    def load_tokenizer(path, eos_token_ids=None):
        tokenizer = object()
        built.append((str(path), eos_token_ids))
        return tokenizer

    utils.load_model = load_model  # type: ignore[attr-defined]
    utils.load_adapters = load_adapters  # type: ignore[attr-defined]
    utils.load_tokenizer = load_tokenizer  # type: ignore[attr-defined]
    fake = ModuleType("mlx_lm")
    fake.utils = utils  # type: ignore[attr-defined]

    def whole_load(*_args, **_kwargs):  # a model folder never goes through it
        pytest.fail("mlx_lm.load used for a model folder")

    fake.load = whole_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)
    monkeypatch.setattr(mlx_engine, "_TOKENIZERS", {})
    return {"built": built, "loaded": loaded}


def _model_dir(tmp_path, name="m"):
    path = tmp_path / name
    path.mkdir()
    (path / "tokenizer.json").write_text("{}")
    (path / "config.json").write_text("{}")
    return path


def test_every_load_builds_the_model_but_one_tokenizer(fake_utils, tmp_path):
    path = _model_dir(tmp_path)
    first = mlx_engine._load_local(str(path), None)
    second = mlx_engine._load_local(str(path), None)
    assert len(fake_utils["loaded"]) == 2  # the weights load (and unload) each time
    assert len(fake_utils["built"]) == 1
    assert first[1] is second[1]
    assert fake_utils["built"][0][1] == [1, 2]  # the model's EOS ids, as mlx_lm.load passes


def test_a_tokenizer_replaced_on_disk_is_built_again(fake_utils, tmp_path):
    path = _model_dir(tmp_path)
    first = mlx_engine._load_local(str(path), None)[1]
    stat = (path / "tokenizer.json").stat()
    os.utime(path / "tokenizer.json", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    second = mlx_engine._load_local(str(path), None)[1]
    assert first is not second
    assert len(fake_utils["built"]) == 2


def test_at_most_eight_are_kept_the_oldest_go(fake_utils, tmp_path):
    paths = [_model_dir(tmp_path, f"m{i}") for i in range(10)]
    for path in paths:
        mlx_engine._load_local(str(path), None)
    assert len(mlx_engine._TOKENIZERS) == 8
    mlx_engine._load_local(str(paths[-1]), None)  # still kept
    assert len(fake_utils["built"]) == 10
    mlx_engine._load_local(str(paths[0]), None)  # the first went: built again
    assert len(fake_utils["built"]) == 11


def test_an_adapter_is_applied_on_the_new_model(fake_utils, tmp_path):
    path = _model_dir(tmp_path)
    model, _ = mlx_engine._load_local(str(path), "/adapters/a")
    assert ("adapter", "/adapters/a") in fake_utils["loaded"]
    assert model is not None


def test_load_takes_a_model_folder_through_the_kept_tokenizer(fake_utils, tmp_path, monkeypatch):
    path = _model_dir(tmp_path)
    monkeypatch.setattr(mlx_engine, "is_available", lambda: True)
    monkeypatch.setattr(
        sys.modules["mlx_lm"], "load", lambda *a, **k: pytest.fail("mlx_lm.load"), raising=False
    )
    engine = mlx_engine.MLXEngine()
    monkeypatch.setattr(engine, "_start_batching", lambda: None)
    monkeypatch.setattr(engine, "_new_prompt_store", lambda: None)
    engine.load(str(path))
    tokenizer = engine._tokenizer
    engine.unload()
    engine.load(str(path))
    assert engine._tokenizer is tokenizer
    assert len(fake_utils["built"]) == 1


@pytest.fixture
def fake_mx(monkeypatch):
    evaluated: list[object] = []
    core = ModuleType("mlx.core")
    random = ModuleType("mlx.core.random")
    random.state = ["key"]  # type: ignore[attr-defined]
    core.random = random  # type: ignore[attr-defined]
    core.eval = lambda *arrays: evaluated.extend(arrays)  # type: ignore[attr-defined]
    mlx = ModuleType("mlx")
    mlx.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return {"evaluated": evaluated, "core": core}


def test_a_load_evaluates_mlx_random_state(fake_utils, fake_mx, tmp_path, monkeypatch):
    path = _model_dir(tmp_path)
    monkeypatch.setattr(mlx_engine, "is_available", lambda: True)
    engine = mlx_engine.MLXEngine()
    monkeypatch.setattr(engine, "_start_batching", lambda: None)
    monkeypatch.setattr(engine, "_new_prompt_store", lambda: None)
    engine.load(str(path))
    assert fake_mx["evaluated"] == [fake_mx["core"].random.state]


def test_settling_the_random_state_never_fails_a_load(fake_mx, monkeypatch):
    def broken(*_arrays):
        raise RuntimeError("metal")

    monkeypatch.setattr(fake_mx["core"], "eval", broken)
    mlx_engine._settle_random_state()  # no exception


def test_without_mlx_there_is_nothing_to_settle(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx.core", None)  # import fails
    mlx_engine._settle_random_state()

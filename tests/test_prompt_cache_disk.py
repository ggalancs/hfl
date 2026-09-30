# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""llama-server's prompt cache kept on disk across unload/reload
(``HFL_PROMPT_CACHE_PERSIST``, plan 0.22 P1-9). Measured with llama-server:
a 1860-token prefix, restored after a restart, is evaluated as 1 token; the
same restart without restoring evaluates 1860."""

from __future__ import annotations

import os

import pytest

from hfl.engine import llama_server as ls

ARGV = ["llama-server", "-m", "m.gguf", "-c", "8192", "-np", "4"]


@pytest.fixture
def persist(temp_config, monkeypatch):
    monkeypatch.setattr("hfl.config.config", temp_config)
    temp_config.prompt_cache_persist = True
    temp_config.prompt_cache_max_gb = 1.0
    return temp_config


def test_off_by_default(temp_config, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("hfl.config.config", temp_config)
    assert ls._prompt_cache_dir(ARGV, str(tmp_path / "m.gguf")) is None


def test_one_folder_per_exact_model_and_configuration(persist, tmp_path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF1")
    first = ls._prompt_cache_dir(ARGV, str(model))
    assert first is not None and first == ls._prompt_cache_dir(ARGV, str(model))
    other_ctx = ARGV[:4] + ["4096"] + ARGV[5:]
    assert ls._prompt_cache_dir(other_ctx, str(model)) != first
    with_lora = [*ARGV, "--lora", str(tmp_path / "a.gguf")]
    assert ls._prompt_cache_dir(with_lora, str(model)) != first
    scaled = ls._prompt_cache_dir(with_lora, str(model), [0.5])
    assert scaled not in (first, ls._prompt_cache_dir(with_lora, str(model)))
    model.write_bytes(b"GGUF22")  # the file changed: its old KV must not come back
    os.utime(model, ns=(1, 1))
    assert ls._prompt_cache_dir(ARGV, str(model)) != first


def test_the_budget_drops_the_oldest_first(persist) -> None:
    root = ls._prompt_cache_root()
    old, new = root / "old", root / "new"
    for folder, stamp in ((old, 1), (new, 2)):
        folder.mkdir(parents=True)
        (folder / "slot-0.bin").write_bytes(b"x" * 600)
        os.utime(folder, (stamp, stamp))
    persist.prompt_cache_max_gb = 1000 / 1024**3  # 1000 bytes: one folder fits
    ls._trim_prompt_cache(new)
    assert not old.exists() and (new / "slot-0.bin").exists()
    persist.prompt_cache_max_gb = 100 / 1024**3  # not even this one
    ls._trim_prompt_cache(new)
    assert not (new / "slot-0.bin").exists()


class _Response:
    def __init__(self, body: dict) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.body


class _Client:
    """llama-server's slot endpoints: slot 1 holds 50 tokens, the rest none."""

    def __init__(self, folder) -> None:
        self.folder, self.calls = folder, []

    def post(self, url: str, json: dict, timeout: float) -> _Response:
        self.calls.append(url)
        slot = int(url.split("/")[2].split("?")[0])
        if "save" in url:
            (self.folder / json["filename"]).write_bytes(b"kv" if slot == 1 else b"")
            return _Response({"n_saved": 50 if slot == 1 else 0})
        return _Response({"n_restored": 50})

    def close(self) -> None:
        pass


def test_unload_saves_what_slots_hold_and_load_restores_it(persist, tmp_path) -> None:
    engine = ls.LlamaServerEngine()
    engine._cache_dir = folder = tmp_path / "cache"
    folder.mkdir()
    engine._argv, engine._client = ARGV, _Client(folder)
    engine._stop = lambda: None
    engine.unload()
    assert sorted(p.name for p in folder.iterdir()) == ["slot-1.bin"]  # empty ones dropped
    engine._client = client = _Client(folder)
    engine._restore_slots(4)
    assert client.calls == ["/slots/1?action=restore"]


def test_launch_points_llama_server_at_the_folder_and_restores(persist, tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    started: list[list[str]] = []

    def start(argv, model_path, log_path, timeout):
        started.append(argv)
        folder = argv[argv.index("--slot-save-path") + 1]
        from pathlib import Path

        (Path(folder) / "slot-2.bin").write_bytes(b"kv")
        return None, _Client(Path(folder))

    monkeypatch.setattr(ls, "start_server", start)
    engine = ls.LlamaServerEngine()
    engine._log_path = tmp_path / "log"
    engine._launch(list(ARGV), str(model), 10)
    assert "--slot-save-path" in started[0]
    assert engine._client.calls == ["/slots/2?action=restore"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl train`` / ``POST /api/train`` (plan 0.22 P1-14). mlx-lm itself is
replaced here by a stand-in process; measured for real on Qwen2.5-0.5B: the
base model did not know a fact in the data, and the adapter, the fused model
and its GGUF export all answered it (through /api/chat)."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from hfl.models.manifest import ModelManifest
from hfl.training import mlx_lora as trainer

CHAT = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}


def _jsonl(path: Path, rows: list) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


# -- data ------------------------------------------------------------------------


def test_a_file_is_split_into_train_and_valid(tmp_path) -> None:
    data = trainer.prepare_data(_jsonl(tmp_path / "d.jsonl", [CHAT] * 20), tmp_path / "out")
    assert (data.format, data.train, data.valid) == ("chat", 18, 2)
    assert len((tmp_path / "out" / "valid.jsonl").read_text().splitlines()) == 2


def test_a_folder_keeps_its_validation_set(tmp_path) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    _jsonl(folder / "train.jsonl", [{"text": "x"}] * 5)
    _jsonl(folder / "valid.jsonl", [{"text": "y"}] * 3)
    data = trainer.prepare_data(folder, tmp_path / "out")
    assert (data.format, data.train, data.valid) == ("text", 5, 3)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([CHAT, {"text": "x"}], "text row in a chat file"),
        ([{"prompt": "p"}], "not a training row"),
        ([CHAT], "at least two rows"),
        (["not json {"], "not JSON"),
    ],
)
def test_bad_data_says_where(tmp_path, rows, message) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    with pytest.raises(trainer.TrainingError, match=message):
        trainer.prepare_data(path, tmp_path / "out")


# -- the command and its output --------------------------------------------------

REAL = [  # mlx_lm lora 0.31 output, as measured
    "Iter 1: Val loss 4.990, Val took 0.415s",
    "Iter 20: Train loss 1.291, Learning Rate 1.000e-05, It/sec 6.593, Tokens/sec 1389.573, "
    "Trained Tokens 4215, Peak mem 1.679 GB",
]


def test_its_output_becomes_progress() -> None:
    assert trainer.parse_line(REAL[0]) == {"iteration": 1, "val_loss": 4.99}
    assert trainer.parse_line(REAL[1]) == {
        "iteration": 20, "train_loss": 1.291, "tokens_per_sec": 1389.573, "peak_memory_gb": 1.679,
    }  # fmt: skip
    assert trainer.parse_line("Loading pretrained model") is None


def test_the_command_line(tmp_path) -> None:
    data = trainer.Data(folder=tmp_path, format="chat", train=3, valid=1)
    adapter = tmp_path / "adapter"
    argv = trainer.command("/m", data, adapter, trainer.Options(batch_size=8))
    assert argv[argv.index("--batch-size") + 1] == "1"  # never more than the data
    assert "--mask-prompt" in argv and "--resume-adapter-file" not in argv
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"x")
    resumed = trainer.command("/m", data, adapter, trainer.Options(resume=True))
    assert resumed[resumed.index("--resume-adapter-file") + 1].endswith("adapters.safetensors")


def _fake(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "fake_mlx.py"
    script.write_text(body)
    return [sys.executable, str(script)]


def test_a_run_reports_progress_and_logs(tmp_path) -> None:
    argv = _fake(tmp_path, "print(%r)\nprint(%r)\n" % (REAL[0], REAL[1]))
    seen: list[dict] = []
    trainer.run(argv, tmp_path / "log", seen.append)
    assert [e["iteration"] for e in seen] == [1, 20]
    assert "Peak mem 1.679 GB" in (tmp_path / "log").read_text()


def test_a_failed_run_says_so(tmp_path) -> None:
    argv = _fake(tmp_path, "import sys\nprint('ValueError: bad model')\nsys.exit(3)\n")
    with pytest.raises(trainer.TrainingError, match="exit 3.*bad model"):
        trainer.run(argv, tmp_path / "log", lambda e: None)


def test_a_run_can_be_stopped(tmp_path) -> None:
    argv = _fake(
        tmp_path, "import time\nprint('Iter 1: Val loss 1.0', flush=True)\ntime.sleep(60)\n"
    )
    stop = threading.Event()
    seen: list[dict] = []

    def on_event(event: dict) -> None:
        seen.append(event)
        stop.set()  # as a client that went away

    started = time.monotonic()
    with pytest.raises(trainer.TrainingError, match="resume"):
        trainer.run(argv, tmp_path / "log", on_event, stop)
    assert seen and time.monotonic() - started < 20  # stopped, not waited out


# -- models ----------------------------------------------------------------------


def _base(tmp_path: Path, **extra) -> ModelManifest:
    folder = tmp_path / "base"
    folder.mkdir(exist_ok=True)
    fields = {
        "name": "q05",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_path": str(folder),
        "format": "safetensors",
        "license": "apache-2.0",
    }
    return ModelManifest(**{**fields, **extra})


def test_the_result_is_the_base_with_the_adapter(temp_config, tmp_path) -> None:
    from hfl.core.container import get_registry, reset_container

    reset_container()
    trained = trainer.register(_base(tmp_path), "mascot", tmp_path / "adapter")
    assert get_registry().get("mascot") is not None
    assert trained.adapter_paths == [str(tmp_path / "adapter")]
    assert (trained.parent_name, trained.license, trained.format) == (
        "q05", "apache-2.0", "safetensors",
    )  # fmt: skip
    reset_container()


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        ({"format": "gguf", "local_path": "/m.gguf"}, "--format safetensors"),
        ({"adapter_paths": ["/a"], "parent_name": "q"}, "already has an adapter"),
        ({"local_path": "/nowhere"}, "not on this machine"),
    ],
)
def test_what_cannot_be_trained_on(tmp_path, extra, why) -> None:
    assert why in (trainer.trainable(_base(tmp_path, **extra)) or "")


@pytest.mark.parametrize("name", ["../x", "a/b", "", "x" * 65, "-x"])
def test_names_are_folder_safe(name) -> None:
    with pytest.raises(trainer.TrainingError):
        trainer.check_name(name)


def test_elsewhere_it_says_what_to_use(monkeypatch) -> None:
    monkeypatch.setattr(trainer.sys, "platform", "linux")
    assert "hfl[train]" in (trainer.available() or "")


def test_mlx_takes_one_mlx_adapter(tmp_path) -> None:
    from hfl.engine.mlx_engine import MLXEngine

    (tmp_path / "adapter_config.json").write_text("{}")
    assert MLXEngine._adapter([str(tmp_path)]) == str(tmp_path)
    assert MLXEngine._adapter(None) is None
    with pytest.raises(ValueError, match="not an MLX adapter"):
        MLXEngine._adapter([str(tmp_path / "lora.gguf")])
    with pytest.raises(ValueError, match="one LoRA"):
        MLXEngine._adapter([str(tmp_path), str(tmp_path)])


# -- the API ---------------------------------------------------------------------


@pytest.fixture
def api(temp_config, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from hfl.api.server import app
    from hfl.core.container import get_registry, reset_container

    reset_container()
    monkeypatch.setattr(trainer, "available", lambda: None)
    get_registry().add(_base(tmp_path))
    get_registry().add(_base(tmp_path, name="gg", format="gguf", local_path="/m.gguf"))
    yield TestClient(app, client=("127.0.0.1", 5555))
    reset_container()


def test_api_refusals(api, monkeypatch) -> None:
    assert api.post("/api/train", json={"model": "nope", "data": "/d"}).status_code == 404
    assert api.post("/api/train", json={"model": "gg", "data": "/d"}).status_code == 400
    assert (
        api.post("/api/train", json={"model": "q05", "data": "/d", "name": "../x"}).status_code
        == 400
    )
    from hfl.training import hf_lora

    monkeypatch.setattr(trainer, "available", lambda: trainer.ELSEWHERE)
    monkeypatch.setattr(hf_lora, "available", lambda: hf_lora.MISSING)
    refused = api.post("/api/train", json={"model": "q05", "data": "/d"})
    assert refused.status_code == 501 and "hfl[train]" in refused.json()["error"]


def test_api_streams_the_run(api, monkeypatch, tmp_path) -> None:
    def fake_run(argv, log, on_event, stop=None):
        on_event({"iteration": 10, "train_loss": 0.5})

    monkeypatch.setattr(trainer, "run", fake_run)
    data = _jsonl(tmp_path / "d.jsonl", [CHAT] * 10)
    body = api.post("/api/train", json={"model": "q05", "data": str(data), "name": "t1"}).text
    events = [json.loads(line) for line in body.splitlines()]
    assert [e["status"] for e in events] == ["checking data", "data", "training", "success"]
    assert api.post("/api/show", json={"model": "t1"}).status_code == 200


def test_api_reports_a_failed_run(api, monkeypatch, tmp_path) -> None:
    def failing(argv, log, on_event, stop=None):
        raise trainer.TrainingError("mlx_lm lora failed (exit 1): out of memory")

    monkeypatch.setattr(trainer, "run", failing)
    data = _jsonl(tmp_path / "d.jsonl", [CHAT] * 10)
    last = api.post(
        "/api/train", json={"model": "q05", "data": str(data), "name": "t2", "stream": False}
    )
    assert last.status_code == 400 and "out of memory" in last.json()["error"]


# -- outside Apple Silicon: Transformers + PEFT ----------------------------------


def test_its_progress_reads_as_mlx_lms() -> None:
    """hf_lora_run prints mlx-lm's words, so one parser reads both."""
    line = "Iter 10: Train loss 2.345, Tokens/sec 812.5, Peak mem 1.250 GB"
    assert trainer.parse_line(line) == {
        "iteration": 10, "train_loss": 2.345, "tokens_per_sec": 812.5, "peak_memory_gb": 1.25,
    }  # fmt: skip
    assert trainer.parse_line("Iter 30: Val loss 1.500") == {"iteration": 30, "val_loss": 1.5}


def test_the_transformers_command_line(tmp_path) -> None:
    from hfl.training import hf_lora

    data = trainer.Data(folder=tmp_path, format="completions", train=3, valid=1)
    argv = hf_lora.command("/m", data, tmp_path / "a", trainer.Options(batch_size=8, resume=True))
    assert argv[1:4] == ["-m", "hfl.training.hf_lora_run", "train"]
    assert argv[argv.index("--format") + 1] == "completions"
    assert argv[argv.index("--batch-size") + 1] == "3" and "--resume" in argv


def test_without_peft_it_says_what_to_install(monkeypatch) -> None:
    import importlib.util

    from hfl.training import hf_lora

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "peft" else real(name, *a)
    )
    assert "hfl[train]" in (hf_lora.available() or "")


def test_the_trained_model_is_merged_then_registered(tmp_path, monkeypatch) -> None:
    """The Transformers engine loads no separate adapter."""
    from hfl.training import hf_lora

    ran, registered = [], []
    monkeypatch.setattr(hf_lora, "run", lambda argv, log, on_event: ran.append(argv))
    monkeypatch.setattr(
        hf_lora, "register_fused", lambda base, name, out: registered.append((name, out)) or out
    )
    base = ModelManifest(name="q05", repo_id="o/q", local_path="/models/q05", format="safetensors")
    adapter = tmp_path / "adapters" / "mine"
    hf_lora.register(base, "mine", adapter, tmp_path / "log")
    assert ran[0][3:4] == ["merge"] and ran[0][ran[0].index("--model") + 1] == "/models/q05"
    assert registered == [("mine", tmp_path / "models" / "mine")]


def test_the_cli_backend_can_be_chosen(monkeypatch, tmp_path) -> None:
    from typer.testing import CliRunner

    from hfl.cli import main
    from hfl.training import hf_lora

    monkeypatch.setattr(hf_lora, "available", lambda: "HF-CHOSEN")
    monkeypatch.setattr(trainer, "available", lambda: None)
    out = CliRunner().invoke(
        main.app, ["train", "q05", "--data", str(tmp_path), "--backend", "transformers"]
    )
    assert out.exit_code == 1 and "HF-CHOSEN" in out.output

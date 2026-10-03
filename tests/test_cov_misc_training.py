# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl train``'s plumbing beyond test_training.py: where it can run, the
rest of the data checks, an interrupted run, fusing, the merged and GGUF
results in the registry, and a run through another backend."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import threading
from pathlib import Path

import pytest

from hfl.models.manifest import ModelManifest
from hfl.training import hf_lora
from hfl.training import mlx_lora as trainer

CHAT = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}


def _jsonl(path: Path, rows: list) -> Path:
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


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


@pytest.fixture
def registry(temp_config):
    from hfl.core.container import get_registry, reset_container

    reset_container()
    yield get_registry
    reset_container()


# -- where it runs --------------------------------------------------------------------


def test_apple_silicon_with_mlx_lm_can_train(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    assert trainer.available() is None


def test_apple_silicon_without_mlx_lm_points_elsewhere(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    assert trainer.available() == trainer.ELSEWHERE


def test_with_transformers_torch_and_peft_hf_can_train(monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: asked.append(name) or object()
    )
    assert hf_lora.available() is None
    assert asked == ["transformers", "torch", "peft"]


# -- data -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["[1, 2]"], "line 1: not a training row"),
        ([{"messages": []}], "not a training row"),
        ([{"messages": [{"role": "user"}]}], "not a training row"),
        (["", "   "], "no rows"),
    ],
)
def test_more_bad_rows(tmp_path, rows, message) -> None:
    with pytest.raises(trainer.TrainingError, match=message):
        trainer.prepare_data(_jsonl(tmp_path / "d.jsonl", rows), tmp_path / "out")


def test_blank_lines_are_skipped_and_lines_counted(tmp_path) -> None:
    path = _jsonl(tmp_path / "d.jsonl", [CHAT, "", CHAT, "nope"])
    with pytest.raises(trainer.TrainingError, match="d.jsonl, line 4: not JSON"):
        trainer.prepare_data(path, tmp_path / "out")


def test_completions_rows(tmp_path) -> None:
    rows = [{"prompt": "p", "completion": "c"}] * 4
    data = trainer.prepare_data(_jsonl(tmp_path / "d.jsonl", rows), tmp_path / "out")
    assert (data.format, data.train, data.valid) == ("completions", 3, 1)


def test_a_folder_without_train_jsonl(tmp_path) -> None:
    (tmp_path / "in").mkdir()
    with pytest.raises(trainer.TrainingError, match="no train.jsonl"):
        trainer.prepare_data(tmp_path / "in", tmp_path / "out")


def test_a_folder_whose_validation_set_is_another_format(tmp_path) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    _jsonl(folder / "train.jsonl", [CHAT] * 3)
    _jsonl(folder / "valid.jsonl", [{"text": "x"}])
    with pytest.raises(trainer.TrainingError, match="valid.jsonl is text, train.jsonl is chat"):
        trainer.prepare_data(folder, tmp_path / "out")


def test_a_folder_without_a_validation_set_gets_one(tmp_path) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    _jsonl(folder / "train.jsonl", [{"text": str(i)} for i in range(10)])
    data = trainer.prepare_data(folder, tmp_path / "out")
    assert (data.train, data.valid) == (9, 1)


def test_a_source_that_does_not_exist(tmp_path) -> None:
    with pytest.raises(trainer.TrainingError, match="no such file or folder"):
        trainer.prepare_data(tmp_path / "missing.jsonl", tmp_path / "out")


def test_text_data_does_not_mask_the_prompt(tmp_path) -> None:
    data = trainer.Data(folder=tmp_path, format="text", train=10, valid=10)
    argv = trainer.command("/m", data, tmp_path / "a", trainer.Options(resume=True))
    assert "--mask-prompt" not in argv
    assert "--resume-adapter-file" not in argv  # nothing saved yet to resume
    assert argv[argv.index("--batch-size") + 1] == "4"


# -- an interrupted run ------------------------------------------------------------------


def _script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "fake.py"
    script.write_text(body)
    return [sys.executable, str(script)]


SLOW = "import time\nprint('Iter 1: Val loss 1.0', flush=True)\ntime.sleep(60)\n"


def test_ctrl_c_stops_the_trainer_and_propagates(tmp_path) -> None:
    def on_event(event: dict) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        trainer.run(_script(tmp_path, SLOW), tmp_path / "log", on_event)
    assert "Iter 1: Val loss 1.0" in (tmp_path / "log").read_text()


def test_an_error_while_reading_kills_the_trainer(tmp_path, monkeypatch) -> None:
    import subprocess

    procs: list[subprocess.Popen] = []
    real = subprocess.Popen

    def popen(*a, **kw):
        procs.append(real(*a, **kw))
        return procs[-1]

    monkeypatch.setattr(subprocess, "Popen", popen)

    def on_event(event: dict) -> None:
        raise RuntimeError("consumer broke")

    with pytest.raises(RuntimeError, match="consumer broke"):
        trainer.run(_script(tmp_path, SLOW), tmp_path / "log", on_event)
    assert procs[0].returncode is not None  # killed and reaped, not left running


def test_a_failure_quotes_the_last_non_blank_line(tmp_path) -> None:
    argv = _script(tmp_path, "import sys\nprint('boom')\nprint('')\nsys.exit(2)\n")
    with pytest.raises(trainer.TrainingError, match=r"failed \(exit 2\): boom$"):
        trainer.run(argv, tmp_path / "log", lambda e: None)


def test_a_failed_python_m_step_names_the_module(tmp_path) -> None:
    argv = [sys.executable, "-m", "json.tool", "--no-such-flag"]
    with pytest.raises(trainer.TrainingError, match=r"^json.tool --no-such-flag failed \(exit 2\)"):
        trainer.run(argv, tmp_path / "log", lambda e: None)


def test_a_failed_hf_lora_run_names_its_step(tmp_path) -> None:
    from hfl.utils.self_exec import module_argv

    argv = module_argv("hfl.training.hf_lora_run", "train")  # argparse: missing options
    with pytest.raises(trainer.TrainingError) as caught:
        trainer.run(argv, tmp_path / "log", lambda e: None)
    assert caught.value.message.startswith("hfl.training.hf_lora_run train failed (exit 2)")


# -- fusing and registering ------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "quantized"),
    [
        ({"quantization": {"bits": 4}}, True),
        ({"quantization_config": {"quant_method": "awq"}}, True),
        ({"hidden_size": 8}, False),
        (None, False),
        ("not json", False),
    ],
)
def test_quantized_reads_the_config(tmp_path, config, quantized) -> None:
    if config is not None:
        text = config if isinstance(config, str) else json.dumps(config)
        (tmp_path / "config.json").write_text(text)
    assert trainer._quantized(tmp_path) is quantized


@pytest.mark.parametrize(("dequantize", "quantized", "flag"), [
    (True, True, True), (True, False, False), (False, True, False),
])  # fmt: skip
def test_fuse_dequantizes_only_a_quantized_base_when_asked(
    tmp_path, monkeypatch, dequantize, quantized, flag
) -> None:
    base = _base(tmp_path)
    if quantized:
        (Path(base.local_path) / "config.json").write_text('{"quantization": {"bits": 4}}')
    ran: list[list[str]] = []
    monkeypatch.setattr(trainer, "run", lambda argv, log, on_event: ran.append(argv))
    trainer.fuse(base, tmp_path / "ad", tmp_path / "out", tmp_path / "log", dequantize=dequantize)
    (argv,) = ran
    assert argv[1:4] == ["-m", "mlx_lm", "fuse"]
    assert argv[argv.index("--model") + 1] == base.local_path
    assert argv[argv.index("--save-path") + 1] == str(tmp_path / "out")
    assert ("--dequantize" in argv) is flag


def test_a_fused_model_is_registered_as_safetensors(registry, tmp_path) -> None:
    folder = tmp_path / "fused"
    (folder / "sub").mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(b"x" * 100)
    (folder / "sub" / "tokenizer.json").write_bytes(b"y" * 20)
    base = _base(tmp_path, adapter_paths=["/old"], quantization="Q4_K_M")
    made = trainer.register_fused(base, "mine", folder)
    assert (made.name, made.format, made.local_path) == ("mine", "safetensors", str(folder))
    assert made.size_bytes == 120 and made.adapter_paths == [] and made.parent_name == "q05"
    assert made.license == "apache-2.0"  # carried over
    assert registry().get("mine") is not None


def test_a_fused_model_becomes_gguf(registry, tmp_path, monkeypatch) -> None:
    from hfl.converter import gguf_converter

    calls: list[tuple] = []

    class Converter:
        def convert(self, source, output, quantize):
            calls.append((source, output, quantize))
            path = Path(f"{output}.{quantize}.gguf")
            path.write_bytes(b"g" * 64)
            return path

    monkeypatch.setattr(gguf_converter, "GGUFConverter", Converter)
    fused = tmp_path / "fused-mine"
    fused.mkdir()
    made = trainer.to_gguf(_base(tmp_path), "mine-q4", fused, "Q4_K_M")
    assert calls == [(fused, tmp_path / "mine-q4", "Q4_K_M")]
    assert (made.format, made.quantization, made.size_bytes) == ("gguf", "Q4_K_M", 64)
    assert made.local_path == str(tmp_path / "mine-q4.Q4_K_M.gguf")
    assert registry().get("mine-q4").format == "gguf"


# -- the run as events, through another backend --------------------------------------------


class _Backend:
    def __init__(self, argv: list[str], fail: BaseException | None = None) -> None:
        self.argv, self.fail = argv, fail
        self.registered: list[tuple] = []

    def command(self, model_path, data, adapter, options):
        self.command_args = (model_path, data.format, adapter, options)
        return self.argv

    def register(self, base, name, adapter, log):
        if self.fail is not None:
            raise self.fail
        self.registered.append((base.name, name, adapter, log))


def _source(tmp_path: Path) -> Path:
    return _jsonl(tmp_path / "d.jsonl", [CHAT] * 4)


def test_events_through_a_backend(tmp_path) -> None:
    argv = _script(tmp_path, "print('Iter 10: Train loss 1.5, Tokens/sec 9.0, Peak mem 0.0 GB')\n")
    backend = _Backend(argv)
    home = tmp_path / "home"
    events = list(trainer.events_of(
        _base(tmp_path), "mine", _source(tmp_path), trainer.Options(), home, backend=backend
    ))  # fmt: skip
    assert events[0] == {"status": "checking data"}
    assert events[1] == {"status": "data", "format": "chat", "train": 3, "valid": 1}
    assert events[2]["status"] == "training" and events[2]["train_loss"] == 1.5
    assert events[-1] == {
        "status": "success",
        "model": "mine",
        "adapter": str(home / "adapters" / "mine"),
    }
    assert backend.registered == [
        ("q05", "mine", home / "adapters" / "mine", home / "logs" / "train-mine.log")
    ]


def test_an_unexpected_failure_is_reported_without_its_details(tmp_path) -> None:
    backend = _Backend(_script(tmp_path, "pass\n"), fail=RuntimeError("secret internals"))
    events = list(trainer.events_of(
        _base(tmp_path), "mine", _source(tmp_path), trainer.Options(), tmp_path / "h",
        threading.Event(), backend=backend,
    ))  # fmt: skip
    assert events[-1] == {"status": "error", "error": "training failed"}


def test_a_training_error_is_reported_as_said(tmp_path) -> None:
    backend = _Backend(_script(tmp_path, "pass\n"), fail=trainer.TrainingError("merge failed"))
    events = list(trainer.events_of(
        _base(tmp_path), "mine", _source(tmp_path), trainer.Options(), tmp_path / "h",
        backend=backend,
    ))  # fmt: skip
    assert events[-1] == {"status": "error", "error": "merge failed"}

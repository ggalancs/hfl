# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The Transformers + PEFT training script, run in-process against small
fakes of torch, transformers and peft (none of which CI installs).

The fakes record what the script asks of them, so these check HFL's own
decisions: which device and dtype, which rows are learned and which tokens
masked, which layers get LoRA, what is printed (mlx-lm's words), when the
adapter is saved and when validation runs."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from hfl.training import hf_lora_run as run_mod
from hfl.training.mlx_lora import parse_line

# -- fakes ---------------------------------------------------------------------------


class _Loss:
    def __init__(self, value: float, log: list[str]) -> None:
        self.value, self.log = value, log

    def backward(self) -> None:
        self.log.append("backward")

    def __float__(self) -> float:
        return self.value


class _Linear:
    pass


class _Other:
    pass


class _Param:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad


class _Model:
    def __init__(self, layers: int | None = 4) -> None:
        self.config = types.SimpleNamespace(num_hidden_layers=layers)
        self.log: list[str] = []
        self.device: str | None = None
        self.batches: list[dict] = []
        self.saved: list[tuple] = []
        self.mode = "eval"

    def to(self, device: str) -> _Model:
        self.device = device
        return self

    def named_modules(self):
        return [
            ("model.layers.0.self_attn.q_proj", _Linear()),
            ("model.layers.0.self_attn.v_proj", _Linear()),
            ("model.layers.0.mlp.down_proj", _Linear()),
            ("model.layers.0.input_layernorm", _Other()),
            ("lm_head", _Linear()),
        ]

    def parameters(self):
        return [_Param(True), _Param(False), _Param(True)]

    def train(self) -> None:
        self.mode = "train"
        self.log.append("train")

    def eval(self) -> None:
        self.mode = "eval"
        self.log.append("eval")

    def __call__(self, **batch: Any):
        self.batches.append({"mode": self.mode, **batch})
        return types.SimpleNamespace(loss=_Loss(2.0 if self.mode == "train" else 1.5, self.log))

    def save_pretrained(self, path: str, **kw: Any) -> None:
        self.saved.append((path, kw))
        self.log.append(f"save:{path}")

    def merge_and_unload(self) -> _Model:
        self.log.append("merged")
        return self


class _Tokenizer:
    """Characters as token ids; a chat template that is the contents in order,
    with a 0 'assistant turn' marker when a generation prompt is asked for."""

    def __init__(self, pad: int | None = 7, batch_encoding: bool = False) -> None:
        self.pad_token_id = pad
        self.batch_encoding = batch_encoding
        self.saved: list[str] = []

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [ord(c) for m in messages for c in m["content"]]
        if add_generation_prompt:
            ids = ids + [0]
        else:
            # the full conversation: the answer follows the same marker
            ids = (
                ids[: -len(messages[-1]["content"])]
                + [0]
                + [ord(c) for c in messages[-1]["content"]]
            )
        if self.batch_encoding:
            return _Encoding(ids)
        return ids

    def save_pretrained(self, path: str) -> None:
        self.saved.append(path)


class _Encoding(dict):
    def __init__(self, ids: list[int]) -> None:
        super().__init__(input_ids=ids)
        self.input_ids = ids


class _Fakes:
    def __init__(self) -> None:
        self.cuda = False
        self.bf16 = True
        self.mps = False
        self.model = _Model()
        self.tokenizer = _Tokenizer()
        self.lora_configs: list[dict] = []
        self.peft_loaded: list[tuple] = []
        self.optim: list[dict] = []
        self.loaded: list[tuple] = []


@pytest.fixture
def fakes(monkeypatch) -> _Fakes:
    f = _Fakes()

    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: f.cuda,
        is_bf16_supported=lambda: f.bf16,
        max_memory_allocated=lambda: 3_500_000_000,
    )
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: f.mps)
    )
    torch.tensor = lambda data, device=None: {"data": data, "device": device}  # type: ignore[attr-defined]

    class _NoGrad:
        def __enter__(self):
            f.model.log.append("no_grad")

        def __exit__(self, *a):
            return False

    torch.no_grad = _NoGrad  # type: ignore[attr-defined]
    torch.nn = types.SimpleNamespace(Linear=_Linear)  # type: ignore[attr-defined]

    class AdamW:
        def __init__(self, params, lr):
            f.optim.append({"params": list(params), "lr": lr})

        def step(self):
            f.model.log.append("step")

        def zero_grad(self):
            f.model.log.append("zero_grad")

    torch.optim = types.SimpleNamespace(AdamW=AdamW)  # type: ignore[attr-defined]

    peft = types.ModuleType("peft")

    def lora_config(**kw):
        f.lora_configs.append(kw)
        return kw

    class PeftModel:
        @staticmethod
        def from_pretrained(model, path, is_trainable=False):
            f.peft_loaded.append((path, is_trainable))
            return model

    def get_peft_model(model, config):
        model.log.append("peft")
        return model

    peft.LoraConfig = lora_config  # type: ignore[attr-defined]
    peft.PeftModel = PeftModel  # type: ignore[attr-defined]
    peft.get_peft_model = get_peft_model  # type: ignore[attr-defined]

    transformers = types.ModuleType("transformers")

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, dtype=None):
            f.loaded.append(("model", path, dtype))
            return f.model

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            f.loaded.append(("tokenizer", path))
            return f.tokenizer

    transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
    transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return f


def _data(tmp_path: Path, train: list[dict], valid: list[dict]) -> Path:
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (folder / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in valid))
    return folder


def _args(tmp_path: Path, data: Path, **kw: Any) -> list[str]:
    argv = ["train", "--model", "/base", "--data", str(data), "--adapter", str(tmp_path / "ad")]
    for key, value in kw.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return argv


CHAT = {"messages": [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "xy"}]}

# -- device ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cuda", "bf16", "mps", "expected"),
    [
        (True, True, False, ("cuda", "bf16")),
        (True, False, False, ("cuda", "fp16")),
        (False, True, True, ("mps", "fp32")),
        (False, True, False, ("cpu", "fp32")),
    ],
)
def test_the_device_and_its_dtype(fakes, cuda, bf16, mps, expected) -> None:
    fakes.cuda, fakes.bf16, fakes.mps = cuda, bf16, mps
    assert run_mod._device() == expected


def test_peak_memory_is_cudas_or_nothing(fakes) -> None:
    assert run_mod._peak_gb("cuda") == pytest.approx(3.5)
    assert run_mod._peak_gb("cpu") == 0.0


# -- encoding rows -------------------------------------------------------------------


def test_text_rows_learn_everything_up_to_the_limit() -> None:
    ids, labels = run_mod._encode({"text": "hello"}, _Tokenizer(), "text", 3)
    assert ids == labels == [ord("h"), ord("e"), ord("l")]


def test_chat_rows_learn_only_the_answer() -> None:
    ids, labels = run_mod._encode(CHAT, _Tokenizer(), "chat", 100)
    # "ab" + marker are the prompt; "xy" the answer
    assert ids == [ord("a"), ord("b"), 0, ord("x"), ord("y")]
    assert labels == [run_mod.IGNORE] * 3 + [ord("x"), ord("y")]


def test_completion_rows_are_a_user_turn_and_an_answer() -> None:
    row = {"prompt": "ab", "completion": "xy"}
    assert run_mod._encode(row, _Tokenizer(), "completions", 100) == run_mod._encode(
        CHAT, _Tokenizer(), "chat", 100
    )


def test_a_batch_encoding_template_is_read_as_ids() -> None:
    ids, labels = run_mod._encode(CHAT, _Tokenizer(batch_encoding=True), "chat", 100)
    assert ids == [ord("a"), ord("b"), 0, ord("x"), ord("y")]
    assert labels[:3] == [run_mod.IGNORE] * 3


def test_a_row_cut_before_its_answer_has_nothing_to_learn() -> None:
    assert run_mod._encode(CHAT, _Tokenizer(), "chat", 3) is None


def test_a_batch_is_padded_and_masked(fakes) -> None:
    batch = run_mod._batch([([1, 2, 3], [-100, 2, 3]), ([4], [4])], pad_id=9, device="cpu")
    assert batch["input_ids"] == {"data": [[1, 2, 3], [4, 9, 9]], "device": "cpu"}
    assert batch["labels"]["data"] == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"]["data"] == [[1, 1, 1], [1, 0, 0]]


def test_rows_skip_blank_lines(tmp_path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"text": "a"}\n\n{"text": "b"}\n')
    assert run_mod._rows(path) == [{"text": "a"}, {"text": "b"}]


# -- training ---------------------------------------------------------------------------


def test_training_reports_in_mlx_lms_words_and_saves(fakes, tmp_path, capsys) -> None:
    data = _data(tmp_path, [CHAT] * 3, [CHAT])
    argv = _args(tmp_path, data, iters=20, batch_size=2, num_layers=2, save_every=10,
                 learning_rate=0.001)  # fmt: skip
    assert run_mod.main(argv) == 0
    out = capsys.readouterr().out.splitlines()

    events = [e for e in map(parse_line, out) if e]
    train = [e for e in events if "train_loss" in e]
    val = [e for e in events if "val_loss" in e]
    assert [e["iteration"] for e in train] == [10, 20]
    assert train[0]["train_loss"] == 2.0 and train[0]["peak_memory_gb"] == 0.0
    assert [e["iteration"] for e in val] == [10, 20] and val[0]["val_loss"] == 1.5
    saved = [line for line in out if "Saved adapter weights" in line]
    assert saved == [f"Iter {n}: Saved adapter weights to {tmp_path / 'ad'}" for n in (10, 20)]

    m = fakes.model
    assert m.device == "cpu" and ("model", "/base", "fp32") in fakes.loaded
    # LoRA on every linear layer but the head, on the last two layers.
    (config,) = fakes.lora_configs
    assert config["target_modules"] == ["down_proj", "q_proj", "v_proj"]
    assert config["layers_to_transform"] == [2, 3]
    # only trainable parameters are optimised, at the given rate
    assert len(fakes.optim[0]["params"]) == 2 and fakes.optim[0]["lr"] == 0.001
    assert m.log.count("backward") == m.log.count("step") == m.log.count("zero_grad") == 20
    train_batches = [b for b in m.batches if b["mode"] == "train"]
    assert len(train_batches) == 20 and all(len(b["input_ids"]["data"]) == 2 for b in train_batches)
    # validation runs without gradients and returns to training afterwards
    assert "no_grad" in m.log and m.mode == "train"


def test_resume_continues_the_saved_adapter(fakes, tmp_path) -> None:
    data = _data(tmp_path, [{"text": "abc"}] * 2, [])
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    run_mod.main(_args(tmp_path, data, format="text", iters=1, resume=True))
    assert fakes.peft_loaded == [(str(adapter), True)]
    assert fakes.lora_configs == []  # not a fresh adapter
    assert not [b for b in fakes.model.batches if b["mode"] == "eval"]  # no valid rows


def test_resume_without_a_saved_adapter_starts_fresh(fakes, tmp_path) -> None:
    data = _data(tmp_path, [{"text": "abc"}], [{"text": "d"}])
    run_mod.main(_args(tmp_path, data, format="text", iters=1, resume=True, num_layers=0))
    assert fakes.peft_loaded == []
    assert fakes.lora_configs[0]["layers_to_transform"] is None  # 0: every layer


def test_a_model_without_a_layer_count_trains_every_layer(fakes, tmp_path) -> None:
    fakes.model = _Model(layers=None)
    data = _data(tmp_path, [{"text": "abc"}], [{"text": "d"}])
    run_mod.main(_args(tmp_path, data, format="text", iters=1))
    assert fakes.lora_configs[0]["layers_to_transform"] is None


def test_cuda_reports_peak_memory_and_pads_with_zero_without_a_pad(fakes, tmp_path, capsys) -> None:
    fakes.cuda = True
    fakes.tokenizer = _Tokenizer(pad=None)
    data = _data(tmp_path, [{"text": "a"}, {"text": "abc"}], [{"text": "d"}])
    run_mod.main(_args(tmp_path, data, format="text", iters=1, batch_size=2))
    event = parse_line(capsys.readouterr().out.splitlines()[0])
    assert event is not None and event["peak_memory_gb"] == pytest.approx(3.5)
    ids = fakes.model.batches[0]["input_ids"]
    assert ids["device"] == "cuda" and [0, 0] in [row[1:] for row in ids["data"]]


def test_nothing_to_learn_within_the_length_stops(fakes, tmp_path) -> None:
    data = _data(tmp_path, [CHAT, CHAT], [CHAT])
    with pytest.raises(SystemExit, match="max-seq-length"):
        run_mod.main(_args(tmp_path, data, iters=1, max_seq_length=2))


def test_val_loss_averages_its_batches(fakes) -> None:
    pairs = [([1], [1])] * 5
    assert run_mod._val_loss(fakes.model, pairs, 0, "cpu", 2) == 1.5
    assert len(fakes.model.batches) == 3  # 2 + 2 + 1
    assert run_mod._val_loss(fakes.model, [], 0, "cpu", 2) == 0.0  # no rows: no division by zero


# -- merging ----------------------------------------------------------------------------


def test_merge_writes_safetensors_and_the_tokenizer(fakes, tmp_path, capsys) -> None:
    out = str(tmp_path / "merged")
    assert run_mod.main(["merge", "--model", "/base", "--adapter", "/ad", "--out", out]) == 0
    assert fakes.peft_loaded == [("/ad", False)]
    assert "merged" in fakes.model.log
    assert fakes.model.saved == [(out, {"safe_serialization": True})]
    assert fakes.tokenizer.saved == [out]
    assert f"Merged into {out}" in capsys.readouterr().out


def test_an_action_is_required() -> None:
    with pytest.raises(SystemExit):
        run_mod.main([])


def test_options_have_mlx_lms_defaults(fakes, tmp_path, monkeypatch) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(run_mod, "train", seen.append)
    run_mod.main(["train", "--model", "m", "--data", "d", "--adapter", "a"])
    (args,) = seen
    assert (args.format, args.iters, args.batch_size, args.num_layers) == ("chat", 600, 4, 16)
    assert (args.max_seq_length, args.save_every, args.resume) == (2048, 100, False)

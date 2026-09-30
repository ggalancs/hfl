# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""LoRA training with Transformers + PEFT, run as its own process.

    python -m hfl.training.hf_lora_run train --model DIR --data DIR --adapter DIR ...
    python -m hfl.training.hf_lora_run merge --model DIR --adapter DIR --out DIR

What ``mlx_lm lora`` is on Apple Silicon, for every other machine (CUDA,
or the CPU). It reads the same data (train.jsonl + valid.jsonl in the
chat, completions or text format) and prints its progress in mlx-lm's
words — "Iter N: Train loss X, Tokens/sec Y, Peak mem Z GB" — so HFL
reads both the same way (``mlx_lora.parse_line``). Like ``--mask-prompt``,
chat and completions rows learn the answers, not the questions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, cast

IGNORE = -100  # the label PyTorch's cross-entropy skips


def _device() -> tuple[str, Any]:
    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _encode(row: dict[str, Any], tokenizer: Any, fmt: str, limit: int) -> tuple[list, list] | None:
    """Token ids and labels for one row: the prompt masked out for chat and
    completions (only the answer is learned), the whole text otherwise."""
    if fmt == "text":
        ids = tokenizer(row["text"])["input_ids"][:limit]
        return ids, list(ids)
    if fmt == "completions":
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
    else:
        messages = row["messages"]
    full = tokenizer.apply_chat_template(messages, tokenize=True)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    if hasattr(full, "input_ids"):  # a BatchEncoding in some versions
        full, prompt = full["input_ids"], prompt["input_ids"]
    full, cut = list(full)[:limit], min(len(prompt), limit)
    labels = [IGNORE] * cut + full[cut:]
    if all(label == IGNORE for label in labels):
        return None  # nothing to learn once cut to the limit
    return full, labels


def _batch(pairs: list[tuple[list, list]], pad_id: int, device: str) -> dict[str, Any]:
    import torch

    width = max(len(ids) for ids, _ in pairs)
    ids = [a + [pad_id] * (width - len(a)) for a, _ in pairs]
    labels = [b + [IGNORE] * (width - len(b)) for _, b in pairs]
    mask = [[1] * len(a) + [0] * (width - len(a)) for a, _ in pairs]
    return {
        "input_ids": torch.tensor(ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(mask, device=device),
    }


def _peak_gb(device: str) -> float:
    import torch

    if device == "cuda":
        return float(torch.cuda.max_memory_allocated()) / 1e9
    return 0.0


def _val_loss(model: Any, rows: list, pad_id: int, device: str, batch_size: int) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            out = model(**_batch(rows[start : start + batch_size], pad_id, device))
            total += float(out.loss)
            count += 1
    model.train()
    return total / max(count, 1)


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    # cast: transformers' stubs type ``.to`` oddly, and are absent in CI.
    model = cast(Any, model).to(device)
    adapter = Path(args.adapter)
    if args.resume and (adapter / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    else:
        layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
        chosen = None
        if args.num_layers > 0 and layers:
            chosen = list(range(max(0, layers - args.num_layers), layers))
        # PEFT takes layers_to_transform only with named modules (its
        # "all-linear" shorthand refused it, on an L4): every linear
        # layer's name but the output head's.
        linear = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and "lm_head" not in name
            }
        )
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=linear,
            layers_to_transform=chosen,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    model.train()

    data = Path(args.data)
    encoded: dict[str, list[tuple[list, list]]] = {}
    for split in ("train", "valid"):
        rows = _rows(data / f"{split}.jsonl")
        coded = [_encode(r, tokenizer, args.format, args.max_seq_length) for r in rows]
        encoded[split] = [pair for pair in coded if pair is not None]
    if not encoded["train"]:
        raise SystemExit("no row has anything to learn within --max-seq-length")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    rng = random.Random(0)
    losses: list[float] = []
    tokens, started = 0, time.monotonic()
    for step in range(1, args.iters + 1):
        pairs = rng.sample(encoded["train"], min(args.batch_size, len(encoded["train"])))
        out = model(**_batch(pairs, pad_id, device))
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(out.loss))
        tokens += sum(len(ids) for ids, _ in pairs)
        if step % 10 == 0 or step == args.iters:
            rate = tokens / max(time.monotonic() - started, 1e-9)
            print(
                f"Iter {step}: Train loss {sum(losses) / len(losses):.3f}, "
                f"Tokens/sec {rate:.1f}, Peak mem {_peak_gb(device):.3f} GB",
                flush=True,
            )
            losses.clear()
            tokens, started = 0, time.monotonic()
        if step % args.save_every == 0 or step == args.iters:
            model.save_pretrained(str(adapter))
            print(f"Iter {step}: Saved adapter weights to {adapter}", flush=True)
        if encoded["valid"] and (step % max(args.save_every, 10) == 0 or step == args.iters):
            val = _val_loss(model, encoded["valid"], pad_id, device, args.batch_size)
            print(f"Iter {step}: Val loss {val:.3f}", flush=True)


def merge(args: argparse.Namespace) -> None:
    """The adapter merged into a full copy of the base model, as safetensors
    (what the Transformers engine serves and llama.cpp's converter reads)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, dtype = _device()
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"Merged into {args.out}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hfl.training.hf_lora_run")
    sub = parser.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--adapter", required=True)
    t.add_argument("--format", default="chat", choices=("chat", "completions", "text"))
    t.add_argument("--iters", type=int, default=600)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--num-layers", type=int, default=16)
    t.add_argument("--learning-rate", type=float, default=1e-5)
    t.add_argument("--max-seq-length", type=int, default=2048)
    t.add_argument("--save-every", type=int, default=100)
    t.add_argument("--resume", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--model", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    (train if args.action == "train" else merge)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

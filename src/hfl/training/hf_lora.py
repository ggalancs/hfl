# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl train`` outside Apple Silicon: LoRA with Transformers + PEFT.

The same interface as :mod:`hfl.training.mlx_lora` — the data, options,
progress and run are that module's own — with the training done by
:mod:`hfl.training.hf_lora_run` in a process of its own. The Transformers
engine does not load a separate adapter, so the trained model is the
adapter merged into a copy of the base, registered under the chosen name
(the adapter itself stays on disk for ``--resume``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hfl.training.mlx_lora import (
    Data,
    Options,
    TrainingError,
    check_name,
    prepare_data,
    register_fused,
    run,
    to_gguf,
    trainable,
)

__all__ = [
    "Data",
    "Options",
    "TrainingError",
    "available",
    "check_name",
    "command",
    "prepare_data",
    "register",
    "run",
    "to_gguf",
    "trainable",
]

MISSING = "Training outside Apple Silicon uses Transformers and PEFT: pip install 'hfl[train]'"


def available() -> str | None:
    """None when Transformers, torch and PEFT are installed; else what to do."""
    import importlib.util

    for package in ("transformers", "torch", "peft"):
        if importlib.util.find_spec(package) is None:
            return MISSING
    return None


def command(model_path: str, data: Data, adapter: Path, options: Options) -> list[str]:
    """The training run's command line."""
    from hfl.utils.self_exec import module_argv

    argv = module_argv(
        "hfl.training.hf_lora_run", "train",
        "--model", model_path,
        "--data", str(data.folder),
        "--adapter", str(adapter),
        "--format", data.format,
        "--iters", str(options.iters),
        "--batch-size", str(max(1, min(options.batch_size, data.train))),
        "--num-layers", str(options.num_layers),
        "--learning-rate", str(options.learning_rate),
        "--max-seq-length", str(options.max_seq_length),
        "--save-every", str(options.save_every),
    )  # fmt: skip
    if options.resume:
        argv.append("--resume")
    return argv


def register(base: Any, name: str, adapter: Path, log: Path) -> Any:
    """The adapter merged into a copy of ``base``, registered as ``name``."""
    out = adapter.parent.parent / "models" / name
    from hfl.utils.self_exec import module_argv

    argv = module_argv(
        "hfl.training.hf_lora_run", "merge",
        "--model", str(base.local_path),
        "--adapter", str(adapter),
        "--out", str(out),
    )  # fmt: skip
    run(argv, log, lambda event: None)
    return register_fused(base, name, out)

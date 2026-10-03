# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A LoRA adapter trained on Apple Silicon with mlx-lm (``mlx_lm lora``).

``hfl train <model> --data <file|folder>`` checks the data, runs
``mlx_lm lora`` in a process of its own (a crash or Ctrl-C there leaves
HFL and the registry untouched), turns its output into progress events and
registers the result as a model of its own — the base model with the
adapter (Modelfile ``ADAPTER``), so it is chatted with like any other.

Measured on Qwen2.5-0.5B-Instruct (bf16): 60 iterations in 9 s, peak
memory 1.7 GB; the base model did not know the fact in the data, the
adapter answered it for a question phrased unlike any in the data.

Where it runs: macOS on Apple Silicon with the ``[mlx]`` extra. Elsewhere
:func:`available` says which tools to train with instead.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import random
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ELSEWHERE = (
    "Training with MLX runs on macOS with Apple Silicon and the [mlx] extra "
    "(pip install 'hfl[mlx]'); elsewhere HFL trains with Transformers and "
    "PEFT (pip install 'hfl[train]')."
)

FORMATS = ("chat", "completions", "text")


class TrainingError(Exception):
    """Training cannot start or did not finish; the message says why.

    The message is HFL's own (it may quote a line of the user's data or the
    trainer's last output), kept in ``message`` as ``HFLError`` keeps its:
    what a route returns, rather than ``str()`` of an exception."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def available() -> str | None:
    """None when this machine can train; otherwise why not, and what to use."""
    import importlib.util

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return ELSEWHERE
    if importlib.util.find_spec("mlx_lm") is None:
        return ELSEWHERE
    return None


# -- data ---------------------------------------------------------------------


def _row_format(row: Any) -> str | None:
    """Which of mlx-lm's formats a JSONL row is in, or None."""
    if not isinstance(row, dict):
        return None
    messages = row.get("messages")
    if (
        isinstance(messages, list)
        and messages
        and all(
            isinstance(m, dict)
            and isinstance(m.get("role"), str)
            and isinstance(m.get("content"), str)
            for m in messages
        )
    ):
        return "chat"
    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return "completions"
    if isinstance(row.get("text"), str):
        return "text"
    return None


def _read_rows(path: Path) -> tuple[list[str], str]:
    """The lines of a JSONL file, all in one format (TrainingError on the
    first one that is not)."""
    lines, kind = [], None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingError(f"{path.name}, line {number}: not JSON ({exc.msg})") from exc
        found = _row_format(row)
        if found is None:
            raise TrainingError(
                f"{path.name}, line {number}: not a training row — "
                '{"messages": [...]}, {"prompt": ..., "completion": ...} or {"text": ...}'
            )
        if kind is not None and found != kind:
            raise TrainingError(f"{path.name}, line {number}: {found} row in a {kind} file")
        kind = found
        lines.append(line)
    if kind is None:
        raise TrainingError(f"{path.name}: no rows")
    return lines, kind


@dataclass
class Data:
    folder: Path  # train.jsonl + valid.jsonl, as mlx-lm reads them
    format: str
    train: int
    valid: int


def prepare_data(source: Path, folder: Path, seed: int = 0) -> Data:
    """``source`` (a JSONL file, or a folder with ``train.jsonl`` and maybe
    ``valid.jsonl``) checked and written to ``folder`` for mlx-lm. Without a
    validation set, a tenth of the rows (at least one) becomes it."""
    if source.is_dir():
        train_file = source / "train.jsonl"
        if not train_file.is_file():
            raise TrainingError(f"{source}: no train.jsonl")
        train, kind = _read_rows(train_file)
        valid_file = source / "valid.jsonl"
        valid: list[str] = []
        if valid_file.is_file():
            valid, valid_kind = _read_rows(valid_file)
            if valid_kind != kind:
                raise TrainingError(f"valid.jsonl is {valid_kind}, train.jsonl is {kind}")
    elif source.is_file():
        train, kind = _read_rows(source)
        valid = []
    else:
        raise TrainingError(f"{source}: no such file or folder")
    if not valid:
        if len(train) < 2:
            raise TrainingError("at least two rows are needed: one to train on, one to validate")
        shuffled = train[:]
        random.Random(seed).shuffle(shuffled)
        cut = max(1, len(shuffled) // 10)
        valid, train = shuffled[:cut], shuffled[cut:]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "train.jsonl").write_text("\n".join(train) + "\n", encoding="utf-8")
    (folder / "valid.jsonl").write_text("\n".join(valid) + "\n", encoding="utf-8")
    return Data(folder=folder, format=kind, train=len(train), valid=len(valid))


# -- the run ------------------------------------------------------------------


@dataclass
class Options:
    iters: int = 600
    batch_size: int = 4
    num_layers: int = 16
    learning_rate: float = 1e-5
    max_seq_length: int = 2048
    save_every: int = 100
    resume: bool = False


_ITER = re.compile(r"Iter (\d+): (Train|Val) loss ([\d.]+)")
_FIELD = {
    "tokens_per_sec": re.compile(r"Tokens/sec ([\d.]+)"),
    "peak_memory_gb": re.compile(r"Peak mem ([\d.]+) GB"),
}


def parse_line(line: str) -> dict[str, Any] | None:
    """One line of ``mlx_lm lora`` output as a progress event, or None."""
    found = _ITER.search(line)
    if found is None:
        return None
    event: dict[str, Any] = {
        "iteration": int(found.group(1)),
        ("train_loss" if found.group(2) == "Train" else "val_loss"): float(found.group(3)),
    }
    for key, pattern in _FIELD.items():
        value = pattern.search(line)
        if value:
            event[key] = float(value.group(1))
    return event


def command(model_path: str, data: Data, adapter: Path, options: Options) -> list[str]:
    """The ``mlx_lm lora`` command line for this run."""
    argv = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", str(data.folder),
        "--adapter-path", str(adapter),
        "--iters", str(options.iters),
        # mlx-lm refuses a batch larger than the data.
        "--batch-size", str(max(1, min(options.batch_size, data.train, data.valid))),
        "--num-layers", str(options.num_layers),
        "--learning-rate", str(options.learning_rate),
        "--max-seq-length", str(options.max_seq_length),
        "--save-every", str(options.save_every),
        "--steps-per-report", "10",
        "--steps-per-eval", str(max(options.save_every, 10)),
    ]  # fmt: skip
    if data.format in ("chat", "completions"):
        argv.append("--mask-prompt")  # learn the answers, not the questions
    saved = adapter / "adapters.safetensors"
    if options.resume and saved.is_file():
        argv += ["--resume-adapter-file", str(saved)]
    return argv


def run(
    argv: list[str],
    log: Path,
    on_event: Callable[[dict[str, Any]], None],
    stop: threading.Event | None = None,
) -> None:
    """Run ``argv``, its output to ``log`` and each progress line to
    ``on_event``. ``stop`` (or Ctrl-C) ends it: mlx-lm has saved the adapter
    every ``save_every`` iterations, which ``resume`` continues from."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log, "a", encoding="utf-8") as sink:
        sink.write(f"\n# {datetime.now().isoformat()} {' '.join(argv)}\n")
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        watcher: threading.Thread | None = None
        if stop is not None:

            def watch() -> None:
                stop.wait()
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)

            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
        tail: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sink.write(line)
                sink.flush()
                tail = (tail + [line.rstrip()])[-20:]
                event = parse_line(line)
                if event is not None:
                    on_event(event)
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=60)
            raise
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if stop is not None and stop.is_set():
        raise TrainingError("stopped; `--resume` continues from the last saved adapter")
    if proc.returncode != 0:
        detail = next((line for line in reversed(tail) if line.strip()), "")
        # "mlx_lm lora", "hfl.training.hf_lora_run train": the step that failed.
        # A module run with -c (self_exec.module_argv, which keeps the working
        # directory off sys.path) names itself inside the code it runs.
        started = re.search(r"run_module\('([\w.]+)'", argv[2]) if argv[1:2] == ["-c"] else None
        if argv[1:2] == ["-m"]:
            step = " ".join(argv[2:4])
        elif started:
            step = " ".join([started.group(1), *argv[3:4]])
        else:
            step = Path(argv[0]).name
        raise TrainingError(f"{step} failed (exit {proc.returncode}): {detail[-300:]}")


def register(base: Any, name: str, adapter: Path) -> Any:
    """The trained model in the registry: the base model with the adapter,
    its license and provenance carried over."""
    from hfl.core.container import get_registry

    manifest = dataclasses.replace(
        base,
        name=name,
        alias=None,
        adapter_paths=[str(adapter)],
        parent_name=base.name,
        parent_digest=None,
        created_at=datetime.now().isoformat(),
        last_used=None,
    )
    get_registry().add(manifest)
    return manifest


NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def check_name(name: str) -> None:
    """A trained model's name is also a folder name: letters, digits, ``.``,
    ``_`` and ``-`` only."""
    if not NAME.match(name) or ".." in name:
        raise TrainingError(f"{name!r}: use letters, digits, '.', '_' and '-' (up to 64)")


def _quantized(model_dir: Path) -> bool:
    try:
        config = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(config.get("quantization") or config.get("quantization_config"))


def fuse(base: Any, adapter: Path, out: Path, log: Path, dequantize: bool = False) -> None:
    """The adapter merged into a copy of the base model (``mlx_lm fuse``)."""
    argv = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", str(base.local_path),
        "--adapter-path", str(adapter),
        "--save-path", str(out),
    ]  # fmt: skip
    if dequantize and _quantized(Path(str(base.local_path))):
        argv.append("--dequantize")  # llama.cpp's converter reads full weights
    run(argv, log, lambda event: None)


def register_fused(base: Any, name: str, folder: Path) -> Any:
    from hfl.core.container import get_registry

    manifest = dataclasses.replace(
        base,
        name=name,
        alias=None,
        local_path=str(folder),
        format="safetensors",
        adapter_paths=[],
        parent_name=base.name,
        parent_digest=None,
        size_bytes=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
        created_at=datetime.now().isoformat(),
        last_used=None,
    )
    get_registry().add(manifest)
    return manifest


def to_gguf(base: Any, name: str, fused: Path, quantize: str) -> Any:
    """A fused model as GGUF, through HFL's own converter (llama.cpp's
    convert_hf_to_gguf: many more architectures than mlx-lm's own export,
    which takes llama and mistral only), registered as ``name``."""
    from hfl.converter.gguf_converter import GGUFConverter
    from hfl.core.container import get_registry

    output = Path(str(fused)).with_name(name)  # the converter adds .<QUANT>.gguf
    path = GGUFConverter().convert(fused, output, quantize)
    manifest = dataclasses.replace(
        base,
        name=name,
        alias=None,
        local_path=str(path),
        format="gguf",
        quantization=quantize,
        adapter_paths=[],
        parent_name=base.name,
        parent_digest=None,
        size_bytes=Path(path).stat().st_size,
        created_at=datetime.now().isoformat(),
        last_used=None,
    )
    get_registry().add(manifest)
    return manifest


def trainable(base: Any) -> str | None:
    """None when mlx-lm can train on ``base``; otherwise why not."""
    fmt = (getattr(base, "format", "") or "").lower()
    if fmt == "gguf" or str(getattr(base, "local_path", "")).endswith(".gguf"):
        return (
            f"{base.name} is a GGUF file; training needs its safetensors: "
            f"hfl pull {base.repo_id} --format safetensors"
        )
    if getattr(base, "adapter_paths", None):
        return f"{base.name} already has an adapter: train on its base, {base.parent_name}"
    if not Path(str(base.local_path)).is_dir():
        return f"{base.name}: its files are not on this machine ({base.local_path})"
    return None


def events_of(
    base: Any,
    name: str,
    data_source: Path,
    options: Options,
    home: Path,
    stop: threading.Event | None = None,
    *,
    backend: Any = None,
) -> Iterator[dict[str, Any]]:
    """The whole run as events (for the API's NDJSON stream): checking,
    each progress line, and the registered model or an error. ``backend``
    is the trainer module (this one when None; ``hf_lora`` elsewhere)."""
    import queue

    events: queue.Queue[dict[str, Any] | None] = queue.Queue()
    failure: list[BaseException] = []
    adapter = home / "adapters" / name

    def work() -> None:
        try:
            data = prepare_data(data_source, home / "training" / name / "data")
            events.put({"status": "data", "format": data.format, "train": data.train,
                        "valid": data.valid})  # fmt: skip
            log = home / "logs" / f"train-{name}.log"
            if backend is None:
                run(command(str(base.local_path), data, adapter, options), log, events.put, stop)
                register(base, name, adapter)
            else:
                argv = backend.command(str(base.local_path), data, adapter, options)
                run(argv, log, events.put, stop)
                backend.register(base, name, adapter, log)
        except BaseException as exc:  # noqa: BLE001 — reported as an event
            failure.append(exc)
        finally:
            events.put(None)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    yield {"status": "checking data"}
    while (event := events.get()) is not None:
        yield {"status": "training", **event} if "iteration" in event else event
    thread.join()
    if failure:
        exc = failure[0]
        message = str(exc) if isinstance(exc, TrainingError) else "training failed"
        yield {"status": "error", "error": message}
        return
    yield {"status": "success", "model": name, "adapter": str(adapter)}

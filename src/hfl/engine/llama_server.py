# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""GGUF inference through a ``llama-server`` process per model.

Why a second llama.cpp backend: the in-process one (llama-cpp-python) keeps
one KV cache per model, so HFL must serialize every request to it.
``llama-server`` decodes several requests together in parallel slots of one
batch, so a coding agent's parallel calls, or two users, stop waiting in
line. It also runs llama.cpp as released (new architectures without waiting
for the Python binding) and in its own process, so a crash in the model
does not take the server down.

Opt in with ``HFL_LLM_LIBRARY=llama-server`` (GGUF models only; anything
else keeps its usual backend). ``HFL_NUM_PARALLEL`` sets the slots (1 included);
unset, the default is 4. The slots share one KV buffer sized to the
model's context (``--kv-unified``), so memory is what a single-slot load
would use.

The process listens on a random loopback port with a random API key, no web
UI and no ``/slots`` endpoint (it would show other requests' prompts): HFL is
its only client, and HFL's own authentication and limits stay in front.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from hfl.engine import cancel
from hfl.engine.base import (
    ChatMessage,
    CountedStream,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
    completion_prompt,
    reasoning_template_vars,
    repeat_penalty_for,
)
from hfl.utils.self_exec import child_guard_argv

logger = logging.getLogger(__name__)

DEFAULT_SLOTS = 4


def binary() -> str | None:
    """The ``llama-server`` to run: ``HFL_LLAMA_SERVER_BIN``, else the PATH's
    (an install of the user's own), else the copy an executable bundles,
    else the one ``hfl install llama-server`` put in ``~/.hfl/bin``."""
    configured = os.environ.get("HFL_LLAMA_SERVER_BIN")
    if configured:
        return configured if Path(configured).is_file() else None
    from hfl.engine.llama_server_dist import bundled_binary, managed_binary

    return shutil.which("llama-server") or bundled_binary() or managed_binary()


@functools.lru_cache(maxsize=4)
def has_gpu(exe: str) -> bool:
    """Whether this llama-server build sees a GPU (``--list-devices``).

    A CPU-only build lists ``(none)``; Accelerate's BLAS is listed on a Mac
    but is the CPU. Unknown (the command failed) counts as a GPU: HFL then
    keeps the concurrency it had rather than making models take turns.
    """
    try:
        done = subprocess.run(
            [exe, "--list-devices"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return True
    text = done.stdout + done.stderr
    if done.returncode != 0 or "Available devices" not in text:
        return True
    devices = [
        line.strip().split(":", 1)[0]
        for line in text.split("Available devices", 1)[1].splitlines()[1:]
        if ":" in line
    ]
    return any(not name.upper().startswith(("BLAS", "CPU")) for name in devices)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _slots() -> int:
    from hfl.config import config

    configured = max(1, int(getattr(config, "queue_max_inflight", 1) or 1))
    return configured if getattr(config, "parallel_explicit", False) else DEFAULT_SLOTS


@functools.lru_cache(maxsize=4)
def _help_text(exe: str) -> str:
    """``exe --help``: what this llama-server build accepts."""
    try:
        done = subprocess.run(
            [exe, "--help"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout + done.stderr


def _speculative_args(exe: str, draft: Any, gpu_layers: int) -> list[str]:
    """A Modelfile DRAFT as llama-server flags: a draft GGUF (``-md``), or
    ``prompt-lookup`` as its n-gram lookup (``--spec-type ngram-simple``,
    in builds that have it; older ones would refuse to start)."""
    if not isinstance(draft, str) or not draft:
        return []
    if draft == "prompt-lookup":
        if "ngram-simple" in _help_text(exe):
            logger.info("speculative decoding: DRAFT prompt-lookup (n-gram lookup)")
            return ["--spec-type", "ngram-simple"]
        logger.warning("DRAFT prompt-lookup: this llama-server has no n-gram lookup; ignored")
        return []
    if not (draft.endswith(".gguf") and Path(draft).is_file()):
        logger.warning("DRAFT %s: llama-server takes a GGUF file as its draft; ignored", draft)
        return []
    args = ["-md", draft, "-ngld", str(gpu_layers)]
    logger.info("speculative decoding: DRAFT %s", Path(draft).name)
    # Builds with --spec-type default it to none: the draft is loaded and
    # never used (measured: same eval time, no acceptance in the log).
    if "draft-simple" in _help_text(exe):
        args += ["--spec-type", "draft-simple"]
    return args


def _argv_slots(argv: list[str]) -> int:
    try:
        return int(argv[argv.index("-np") + 1])
    except (ValueError, IndexError):
        return 0


def _prompt_cache_root() -> Path:
    from hfl.config import config

    return Path(config.home_dir) / "cache" / "llama-server"


def _prompt_cache_dir(
    argv: list[str], model_path: str, lora_scales: list[float] | None = None
) -> Path | None:
    """The folder for this exact process's slots, or None when the prompt
    cache is not kept on disk. Keyed by everything that shapes the KV —
    the model file (path, size, mtime), its LoRA files, context, slots,
    cache type, template: KV restored into anything else would be silently
    wrong, so any change starts a fresh folder."""
    from hfl.config import config

    if not getattr(config, "prompt_cache_persist", False):
        return None
    import hashlib

    def stat(path: str) -> list[Any]:
        try:
            info = Path(path).stat()
        except OSError:
            return [path]
        return [path, info.st_size, info.st_mtime_ns]

    loras = argv[argv.index("--lora") + 1].split(",") if "--lora" in argv else []
    parts: list[Any] = [argv[1:], stat(model_path), [stat(p) for p in loras]]
    if lora_scales:  # set after the start, so not in argv
        parts.append(lora_scales)
    identity = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    folder = _prompt_cache_root() / f"{Path(model_path).stem}-{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _trim_prompt_cache(current: Path) -> None:
    """Keep the saved KV under ``HFL_PROMPT_CACHE_MAX_GB``, over all models:
    the least recently saved folders go first, then this one's files."""
    from hfl.config import config

    budget = float(getattr(config, "prompt_cache_max_gb", 4.0)) * 1024**3
    root = _prompt_cache_root()
    if not root.is_dir():
        return
    folders = sorted((f for f in root.iterdir() if f.is_dir()), key=lambda f: f.stat().st_mtime)

    def size(folder: Path) -> int:
        return sum(p.stat().st_size for p in folder.glob("slot-*.bin"))

    total = sum(size(f) for f in folders)
    for folder in [f for f in folders if f != current] + [current]:
        if total <= budget:
            return
        for saved in folder.glob("slot-*.bin"):
            total -= saved.stat().st_size
            saved.unlink(missing_ok=True)
        if folder != current:
            try:
                folder.rmdir()
            except OSError:
                pass
        logger.info("prompt cache: %s dropped to stay under %.1f GB", folder.name, budget / 1024**3)


def _multi_gpu_args() -> list[str]:
    """llama-server's flags for ``HFL_TENSOR_SPLIT`` / ``HFL_MAIN_GPU`` /
    ``HFL_SPLIT_MODE``; none when unset (the layers go over every GPU)."""
    from hfl.config import config

    args: list[str] = []
    if config.gpu_tensor_split:
        args += ["--tensor-split", ",".join(f"{share:g}" for share in config.gpu_tensor_split)]
    if config.gpu_main is not None:
        args += ["--main-gpu", str(config.gpu_main)]
    if config.gpu_split_mode:
        args += ["--split-mode", config.gpu_split_mode]
    return args


def _flash_attention_args(requested: Any) -> list[str]:
    """``-fa`` as llama-cpp-python gets it: a per-load ``flash_attn``, else
    ``HFL_FLASH_ATTENTION`` / ``OLLAMA_FLASH_ATTENTION``; unset leaves
    llama-server's own choice (``auto``)."""
    if isinstance(requested, bool):
        on: bool | None = requested
    else:
        raw = (
            os.environ.get("HFL_FLASH_ATTENTION") or os.environ.get("OLLAMA_FLASH_ATTENTION") or ""
        ).strip()
        on = raw.lower() in ("1", "true", "yes", "on") if raw else None
    if on is None:
        return []
    logger.info("llama-server: flash attention %s", "on" if on else "off")
    return ["-fa", "on" if on else "off"]


def _kv_cache_args(requested: Any) -> list[str]:
    """``HFL_KV_CACHE_TYPE`` (or a per-load ``kv_cache_type``) for keys and
    values, as llama-cpp-python applies it; ``f16`` is the default."""
    from hfl.config import config

    kv_type = str(requested or getattr(config, "kv_cache_type", "f16") or "f16").lower()
    if kv_type == "f16":
        return []
    if kv_type not in ("q4_0", "q8_0", "f32"):
        logger.warning("kv_cache_type=%r unsupported by llama-server, falling back to f16", kv_type)
        return []
    logger.info("KV cache quantised to %s", kv_type)
    return ["-ctk", kv_type, "-ctv", kv_type]


def _gpu_layers(requested: Any) -> int:
    """llama-cpp-python's ``-1`` (all layers) is llama-server's ``999``."""
    return int(requested) if isinstance(requested, int) and requested >= 0 else 999


def _sampling(cfg: GenerationConfig) -> dict[str, Any]:
    body: dict[str, Any] = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "repeat_penalty": cfg.repeat_penalty,
    }
    if cfg.seed >= 0:
        body["seed"] = cfg.seed
    if cfg.stop:
        body["stop"] = list(cfg.stop)
    return body


def _response_format(value: str | dict | None) -> dict[str, Any] | None:
    if value == "json":
        return {"type": "json_object"}
    if isinstance(value, dict):
        return {"type": "json_schema", "json_schema": {"schema": value}}
    return None  # GBNF passthrough is an llama-cpp-python feature


def _wire_messages(messages: list[ChatMessage], vision: bool = False) -> list[dict[str, Any]]:
    from hfl.engine.llama_cpp import _image_data_uri

    wire: list[dict[str, Any]] = []
    for message in messages:
        content: Any = message.content
        if message.images:
            if not vision:
                raise ValueError(
                    "this model cannot see images: it has no image projector (mmproj) "
                    "beside it; pull the model again to fetch it"
                )
            # OpenAI's content parts, which llama-server reads with --mmproj.
            content = [{"type": "text", "text": message.content}] if message.content else []
            content += [
                {"type": "image_url", "image_url": {"url": _image_data_uri(image)}}
                for image in message.images
            ]
        entry: dict[str, Any] = {"role": message.role, "content": content}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.get("id") or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": (call.get("function") or {}).get("name", ""),
                        "arguments": _arguments_text((call.get("function") or {}).get("arguments")),
                    },
                }
                for index, call in enumerate(message.tool_calls)
            ]
        if message.role == "tool":
            entry["tool_call_id"] = message.tool_call_id or ""
            if message.name:
                entry["name"] = message.name
        wire.append(entry)
    return wire


def _arguments_text(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments or {})


def _canonical_tool_calls(calls: list[dict[str, Any]] | None) -> list[dict] | None:
    if not calls:
        return None
    canonical: list[dict] = []
    for call in calls:
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            arguments = {}
        canonical.append({"function": {"name": fn.get("name", ""), "arguments": arguments}})
    return canonical


def _logprob_entries(raw: Any, top: int) -> list[dict] | None:
    """llama-server's per-token logprobs as HFL's entries (OpenAI's shape,
    without its token ids), dropping the end-of-generation token it lists
    last — OpenAI never shows one — and alternatives beyond ``top``."""
    if not isinstance(raw, list):
        return None

    def entry(item: dict) -> dict:
        return {
            "token": item.get("token", ""),
            "logprob": float(item.get("logprob", 0.0)),
            "bytes": item.get("bytes") or [],
        }

    entries = [
        {**entry(item), "top_logprobs": [entry(a) for a in (item.get("top_logprobs") or [])[:top]]}
        for item in raw
        if isinstance(item, dict)
    ]
    while entries and not entries[-1]["token"] and not entries[-1]["bytes"]:
        entries.pop()
    return entries


def _as_marker(call: dict) -> str:
    """A structured call written back as the ``<tool_call>`` text HFL's
    parsers read, for the streaming path that only carries text."""
    fn = call["function"]
    return (
        f"<tool_call>{json.dumps({'name': fn['name'], 'arguments': fn['arguments']})}</tool_call>"
    )


# Names of secrets: any variable with one of these words in its name.
_SECRET_WORDS = frozenset({"KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIALS"})


def _child_env(key: str) -> dict[str, str]:
    """The environment for llama-server and its guard: HFL's, without its
    secrets (HF_TOKEN, HFL_API_KEY, the search providers' keys…), and with
    llama-server's own key. llama-server parses whatever a client sends —
    prompts, grammars, images — and needs none of them; what it never holds
    it cannot give away."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not _SECRET_WORDS.intersection(name.upper().split("_"))
    }
    env["LLAMA_API_KEY"] = key
    return env


def start_server(
    base_argv: list[str], model_path: str, log_path: Path, timeout: float
) -> tuple[subprocess.Popen[bytes], httpx.Client]:
    """Start llama-server on a fresh port and key and wait until it answers:
    the process and a client for it. On failure it is stopped before the
    error propagates."""
    port, key = _free_port(), secrets.token_urlsafe(24)
    argv = [*base_argv, "--port", str(port)]
    with open(log_path, "ab") as log:
        # The key goes in the environment, not argv: argv is visible to
        # every local user in ``ps``.
        # Through the guard: if HFL dies without unloading (SIGKILL, a
        # crash), the guard stops llama-server instead of leaving it
        # holding the model in memory.
        guarded = child_guard_argv(os.getpid(), argv)
        proc = subprocess.Popen(
            guarded,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_child_env(key),
            # Its own process group, guard and llama-server: what
            # ``stop_server`` kills when SIGTERM is not enough.
            start_new_session=os.name != "nt",
        )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited while loading {Path(model_path).name}; see {log_path}"
                )
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"llama-server did not load {model_path} in time")
            time.sleep(0.25)
    except BaseException:
        stop_server(proc)
        raise
    client = httpx.Client(
        base_url=base,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(None, connect=10.0),
        # A new connection per request. llama-server closes kept-alive
        # connections on its own, and a request sent on one it just closed
        # fails with "Server disconnected without sending a response"
        # (measured on Linux: 2 of 12 streamed requests; 0 of 40 without
        # keep-alive). A local connection costs next to nothing.
        limits=httpx.Limits(max_keepalive_connections=0),
    )
    return proc, client


# How long llama-server gets to stop on SIGTERM before it is killed.
_STOP_WAIT = 30.0


def stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    """Stop a llama-server started by ``start_server`` (SIGTERM, then kill)."""
    if proc is not None and proc.poll() is None:
        if os.name == "nt":
            # SIGTERM is TerminateProcess there: it ends the guard alone,
            # and its llama-server would outlive it holding the model.
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30
            )
            proc.wait(timeout=10)
            return
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=_STOP_WAIT)
        except subprocess.TimeoutExpired:
            # Killing the guard alone left llama-server running, holding its
            # model, for good: one that got SIGTERM while still starting
            # ignored it (llama.cpp loses a signal that early; measured in
            # the soak, 11 hours). The whole group goes.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=10)


# A request that got no reply because the process was dying: refused, or
# accepted and then reset (measured: a request sent right after a SIGKILL
# got "Connection reset by peer"). Sent again only once the process is
# confirmed dead and started again (``_died``), so no reply is repeated.
_BEFORE_A_REPLY = (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)


class LlamaServerEngine(InferenceEngine):
    """One ``llama-server`` child process serving one GGUF model."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._client: httpx.Client | None = None
        self._model_path = ""
        self._n_ctx = 0
        self._slots = 0
        # Generating on the CPU alone (no GPU, or no layer offloaded).
        self._cpu_only = False
        self._log_path: Path | None = None
        # The template llama-server renders, and whether it lists the tools
        # itself; when not, HFL writes them in (``_tools_as_text``).
        self._chat_template = ""
        self._template_knows_tools = True
        # The image projector served with the model (None: text only).
        self._projector: Path | None = None
        # What the process is started with, to start it again with other
        # LoRA adapters: its argv, the BOS template when one was needed, and
        # the adapters as (id, path, scale).
        self._argv: list[str] = []
        self._template_args: list[str] = []
        self._loras: list[tuple[str, str, float]] = []
        self._timeout = 600.0
        # Where this process keeps its slots' KV between runs (None: off).
        self._cache_dir: Path | None = None
        # Its --slot-save-path when the prompt cache is off: where snapshots
        # pass through (created by this engine, removed when empty).
        self._work_dir: Path | None = None
        # One restart at a time when the process died under the model.
        self._revive_lock = threading.Lock()

    # ------------------------------------------------------------------ life

    def load(self, model_path: str, **kwargs: Any) -> None:
        from hfl.config import config
        from hfl.engine.llama_cpp import _read_gguf_model_info, resolve_n_ctx

        exe = binary()
        if exe is None:
            raise RuntimeError(
                "llama-server was not found: `hfl install llama-server` (llama.cpp's own "
                "build), install llama.cpp, or set HFL_LLAMA_SERVER_BIN"
            )
        requested = kwargs.get("n_ctx")
        explicit = isinstance(requested, int) and requested > 0
        info = _read_gguf_model_info(model_path)
        n_ctx: int | None
        if explicit or info is not None:
            n_ctx = resolve_n_ctx(
                model_path,
                info,
                requested if isinstance(requested, int) and explicit else config.default_ctx_size,
                explicit,
            )
        else:
            # Without the ``gguf`` package HFL cannot read the model's own
            # limit, and its machine-sized default could exceed it. Leave the
            # context unset: llama-server reads the GGUF itself and fits the
            # context to free memory (``--fit``, on by default).
            n_ctx = None
        slots = _slots()
        from hfl.engine.projector import find_projector

        self._loras = []
        for adapter in kwargs.get("lora_paths") or []:
            self._check_adapter_path(str(adapter))
            self._loras.append((str(adapter), str(adapter), 1.0))
        requested_projector = kwargs.get("clip_model_path")
        self._projector = (
            Path(requested_projector) if requested_projector else find_projector(Path(model_path))
        )
        base_argv = [
            exe,
            "-m",
            model_path,
            "--host",
            "127.0.0.1",
            *(["-c", str(n_ctx)] if n_ctx else []),
            "-np",
            str(slots),
            "--kv-unified",
            "-ngl",
            str(_gpu_layers(kwargs.get("n_gpu_layers"))),
            *_multi_gpu_args(),
            *_flash_attention_args(kwargs.get("flash_attn")),
            *_kv_cache_args(kwargs.get("kv_cache_type")),
            "--jinja",
            "--reasoning-format",
            "none",
            "--no-webui",
            "--no-slots",
            # A vision model's projector: llama-server then takes images.
            *(["--mmproj", str(self._projector)] if self._projector else []),
            # Speculative decoding (Modelfile DRAFT).
            *_speculative_args(
                exe, kwargs.get("draft_model_path"), _gpu_layers(kwargs.get("n_gpu_layers"))
            ),
        ]
        log_dir = config.home_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"llama-server-{Path(model_path).stem}.log"
        timeout = float(getattr(config, "model_load_timeout", 600) or 600)
        self._argv, self._template_args, self._timeout = base_argv, [], timeout
        self._launch([*base_argv, *self._lora_args()], model_path, timeout)
        fixed = self._template_with_bos(config.home_dir / "templates", Path(model_path).stem)
        if fixed is not None:
            # Once more with the corrected template (see
            # ``_template_with_bos``): llama-server reads it only at start.
            self._template_args = ["--chat-template-file", str(fixed)]
            self.unload()
            argv = [*base_argv, *self._template_args, *self._lora_args()]
            self._launch(argv, model_path, timeout)
        self._read_template()
        if not n_ctx:
            n_ctx = self._reported_ctx()
        self._model_path, self._n_ctx, self._slots = model_path, n_ctx, slots
        self._cpu_only = _gpu_layers(kwargs.get("n_gpu_layers")) == 0 or not has_gpu(exe)
        logger.info(
            "llama-server serving %s: %d-token context shared by %d parallel slots",
            Path(model_path).name,
            n_ctx,
            slots,
        )

    def _launch(self, base_argv: list[str], model_path: str, timeout: float) -> None:
        assert self._log_path is not None
        self._cache_dir = _prompt_cache_dir(base_argv, model_path, self._lora_scales())
        # Always a place for slot files: the prompt cache's, or a work folder
        # of this engine's own (KV snapshots go through it).
        base_argv = [*base_argv, "--slot-save-path", str(self._slot_dir(model_path))]
        self._proc, self._client = start_server(base_argv, model_path, self._log_path, timeout)
        self._set_lora_scales()
        if self._cache_dir is not None:
            self._restore_slots(_argv_slots(base_argv))

    def _slot_dir(self, model_path: str = "") -> Path:
        """Where llama-server reads and writes slot files."""
        if self._cache_dir is not None:
            return self._cache_dir
        if self._work_dir is None or not self._work_dir.is_dir():
            import tempfile

            root = _prompt_cache_root().parent / "llama-server-work"
            root.mkdir(parents=True, exist_ok=True)
            self._work_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{Path(model_path or self._model_path or 'model').stem}-", dir=root
                )
            )
        return self._work_dir

    # ------------------------------------------------------------ snapshots

    # KV snapshots are its slots' files (``hfl.engine.snapshot``).
    kv_snapshots_as_slot_files = True

    def save_slots_to(self, dest: Path) -> int:
        """Every slot's KV into ``dest`` as ``slot-N.bin`` (empty slots are
        left out); the tokens saved. For ``/api/snapshot/save``: the caller
        holds this model's queue exclusively, so no slot is generating."""
        import shutil

        slot_dir, total = self._slot_dir(), 0
        for slot in range(self._slots or _argv_slots(self._argv)):
            name = f"hfl-snapshot-{slot}.bin"
            done = self._post(f"/slots/{slot}?action=save", json={"filename": name}, timeout=600)
            done.raise_for_status()
            tokens = int(done.json().get("n_saved", 0))
            written = slot_dir / name
            if tokens and written.is_file():
                shutil.move(str(written), str(dest / f"slot-{slot}.bin"))
                total += tokens
            else:
                written.unlink(missing_ok=True)
        return total

    def load_slots_from(self, src: Path) -> int:
        """The slots saved by :meth:`save_slots_to` back in; tokens restored."""
        import shutil

        slot_dir, total = self._slot_dir(), 0
        for slot in range(self._slots or _argv_slots(self._argv)):
            saved = src / f"slot-{slot}.bin"
            if not saved.is_file():
                continue
            name = f"hfl-snapshot-{slot}.bin"
            shutil.copyfile(saved, slot_dir / name)
            try:
                done = self._post(
                    f"/slots/{slot}?action=restore", json={"filename": name}, timeout=600
                )
                done.raise_for_status()
                total += int(done.json().get("n_restored", 0))
            finally:
                (slot_dir / name).unlink(missing_ok=True)
        return total

    # --------------------------------------------------- prompt cache on disk

    def _restore_slots(self, slots: int) -> None:
        """Each slot's KV as the last unload of this exact model and
        configuration saved it (``HFL_PROMPT_CACHE_PERSIST``)."""
        assert self._cache_dir is not None
        restored = 0
        for slot in range(slots):
            saved = self._cache_dir / f"slot-{slot}.bin"
            if not saved.is_file():
                continue
            try:
                done = self._post(
                    f"/slots/{slot}?action=restore", json={"filename": saved.name}, timeout=300
                )
                done.raise_for_status()
                restored += int(done.json().get("n_restored", 0))
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("prompt cache: slot %d not restored (%s); dropped", slot, exc)
                saved.unlink(missing_ok=True)
        if restored:
            logger.info("prompt cache: %d tokens restored from disk", restored)

    def _save_slots(self) -> None:
        """Each slot's KV to disk before the process stops; empty ones and
        whatever the size budget cannot hold are not kept."""
        if self._cache_dir is None or self._client is None:
            return
        saved = 0
        for slot in range(_argv_slots(self._argv) or self._slots):
            name = f"slot-{slot}.bin"
            try:
                done = self._client.post(
                    f"/slots/{slot}?action=save", json={"filename": name}, timeout=300
                )
                done.raise_for_status()
                tokens = int(done.json().get("n_saved", 0))
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("prompt cache: slot %d not saved (%s)", slot, exc)
                tokens = 0
            if tokens:
                saved += tokens
            else:
                (self._cache_dir / name).unlink(missing_ok=True)
        if saved:
            logger.info("prompt cache: %d tokens saved to disk", saved)
        _trim_prompt_cache(self._cache_dir)

    def _props(self) -> dict[str, Any]:
        try:
            props = self._get("/props", timeout=10).json()
        except (httpx.HTTPError, ValueError):
            return {}
        return props if isinstance(props, dict) else {}

    def _template_with_bos(self, directory: Path, stem: str) -> Path | None:
        """A copy of the model's chat template that starts with BOS, written
        to ``directory``, when the vocabulary wants BOS and the template does
        not write it; else None.

        llama-server renders a template's prompt without adding BOS, as
        llama-cpp-python does: Hermes-3 3B then answered a tool prompt with
        ``】,\\n762\\n##...`` (measured). Whether BOS is wanted is asked of
        llama-server itself — its tokenizer, with special tokens on.
        """
        from hfl.models.chat_template import repair_chat_template

        props = self._props()
        template, bos = props.get("chat_template"), props.get("bos_token")
        if not isinstance(template, str):
            return None
        # Also the known mistakes of shipped templates (Qwen2.5-Coder's
        # doubled braces), corrected in the same copy.
        repaired = repair_chat_template(template)
        if repaired != template:
            logger.info("Chat template has a known mistake; HFL corrects it")
        needs_bos = (
            isinstance(bos, str)
            and bool(bos)
            and "bos_token" not in template
            and bos not in template
            and self._adds_bos()
        )
        if needs_bos:
            logger.info("Chat template does not start with BOS; HFL adds it")
        if repaired == template and not needs_bos:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.jinja"
        path.write_text(("{{ bos_token }}" if needs_bos else "") + repaired, encoding="utf-8")
        return path

    def _adds_bos(self) -> bool:
        """Whether this vocabulary wants BOS: asked of llama-server's own
        tokenizer, with special tokens on and off."""
        try:
            with_special = self._post(
                "/tokenize", json={"content": "a", "add_special": True}, timeout=10
            )
            without = self._post(
                "/tokenize", json={"content": "a", "add_special": False}, timeout=10
            )
            return len(with_special.json()["tokens"]) > len(without.json()["tokens"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return False

    def _read_template(self) -> None:
        """What the running template does with tools (llama.cpp's own
        probe of it, ``chat_template_caps``)."""
        props = self._props()
        template = props.get("chat_template")
        caps = props.get("chat_template_caps")
        self._chat_template = template if isinstance(template, str) else ""
        if isinstance(caps, dict) and "supports_tools" in caps:
            self._template_knows_tools = bool(caps["supports_tools"])
        else:
            from hfl.engine.llama_cpp import _template_renders_tools

            self._template_knows_tools = _template_renders_tools(self._chat_template, None)

    def _reported_ctx(self) -> int:
        """The context llama-server chose, from ``/props`` (0 if unknown)."""
        props = self._props()
        ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
        return int(ctx) if isinstance(ctx, int) else 0

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        stop_server(proc)

    def unload(self) -> None:
        if self._client is not None:
            try:
                self._save_slots()
            finally:
                self._client.close()
                self._client = None
        self._stop()
        if self._work_dir is not None:
            try:
                self._work_dir.rmdir()  # only when empty: it holds nothing kept
            except OSError:
                pass

    # ------------------------------------------------------------- requests

    def _http(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("llama-server engine is not loaded")
        if self._proc is not None and self._proc.poll() is not None:
            self._revive(self._proc)
        assert self._client is not None
        return self._client

    def _revive(self, dead: subprocess.Popen[bytes]) -> None:
        """Start llama-server again for the loaded model: it died under it
        (the OOM killer, a crash). Requests went on to its closed port — a
        500 each until keep_alive unloaded the model (measured, audit G1)."""
        with self._revive_lock:
            if self._proc is not dead or self._client is None:
                return  # another request started it again, or it was unloaded
            logger.warning(
                "llama-server for %s exited (code %s); starting it again",
                Path(self._model_path).name,
                dead.returncode,
            )
            old = self._client
            argv = [*self._argv, *self._template_args, *self._lora_args()]
            self._launch(argv, self._model_path, self._timeout)
            old.close()

    def _died(self) -> bool:
        """After a request got no reply: whether the process had died (and was
        started again). Its guard exits a moment after llama-server, so a
        request sent right after the death still saw it running."""
        proc = self._proc
        if proc is None or not self._model_path:
            return False
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            return False  # still running: the failure is something else
        self._revive(proc)
        return True

    def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http().post(path, **kwargs)
        except _BEFORE_A_REPLY:
            if not self._died():
                raise
            return self._http().post(path, **kwargs)

    def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http().get(path, **kwargs)
        except _BEFORE_A_REPLY:
            if not self._died():
                raise
            return self._http().get(path, **kwargs)

    @contextlib.contextmanager
    def _stream(self, method: str, path: str, **kwargs: Any) -> Iterator[httpx.Response]:
        """``httpx.Client.stream``, sent again once if the process had died
        before it answered. Once a reply has begun, a death ends it with an
        error: half a reply is never repeated."""
        try:
            opened = self._http().stream(method, path, **kwargs)
            response = opened.__enter__()
        except _BEFORE_A_REPLY:
            if not self._died():
                raise
            opened = self._http().stream(method, path, **kwargs)
            response = opened.__enter__()
        try:
            yield response
        except BaseException:
            if not opened.__exit__(*sys.exc_info()):
                raise
        else:
            opened.__exit__(None, None, None)

    def _chat_body(
        self, messages: list[ChatMessage], cfg: GenerationConfig, tools: list[dict] | None
    ) -> dict[str, Any]:
        from hfl.engine.llama_cpp import _history_for_template, _tools_as_text

        penalty = repeat_penalty_for(cfg, messages, tools)
        wire = _wire_messages(messages, vision=self._projector is not None)
        if self._template_knows_tools:
            wire = _history_for_template(wire, self._chat_template)
        else:
            # The same as the in-process backend: llama-server's own
            # handling of such a template did not get Hermes-3 or
            # DeepSeek-R1 to call a tool at all (measured).
            wire, tools = _tools_as_text(wire, tools), None
        body: dict[str, Any] = {
            "messages": wire,
            "max_tokens": cfg.max_tokens,
            **_sampling(cfg),
            "repeat_penalty": penalty,
        }
        template_vars = reasoning_template_vars(cfg.reasoning)
        if template_vars:
            # ``think: false`` & co. reach the template, not only the reply.
            body["chat_template_kwargs"] = template_vars
        if tools:
            body["tools"] = tools
        if cfg.logprobs is not None:
            body["logprobs"] = True
            # At 0 it sends none at all, not even the drawn token's: ask for
            # one alternative and keep none (``_logprob_entries``).
            body["top_logprobs"] = max(1, cfg.logprobs)
        response_format = _response_format(cfg.response_format)
        if response_format is not None:
            body["response_format"] = response_format
        return body

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()
        started = time.monotonic_ns()
        body = self._chat_body(messages, cfg, tools)
        if cancel.current() is not None:
            data = self._streamed_chat(body)
        else:
            response = self._post("/v1/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        result = self._result(
            self._answer(message.get("content") or "", cfg),
            data,
            started,
            choice.get("finish_reason"),
            _canonical_tool_calls(message.get("tool_calls")),
        )
        if cfg.logprobs is not None:
            content = (choice.get("logprobs") or {}).get("content")
            result.logprobs = _logprob_entries(content, cfg.logprobs)
        return result

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        cfg = config or GenerationConfig()
        counted = CountedStream()
        if tools:
            # llama-server streams tool calls as structured deltas, and this
            # interface carries text. The routes buffer tool-aware turns and
            # parse them at the end anyway, so answer in one piece and write
            # any call back as the marker their parsers read.
            def _whole() -> Iterator[str]:
                result = self.chat(messages, cfg, tools)
                counted.prompt_tokens = result.tokens_prompt
                counted.completion_tokens = result.tokens_generated
                if result.text:
                    yield result.text
                for call in result.tool_calls or []:
                    yield _as_marker(call)

            return counted.feed(_whole())
        body = {
            **self._chat_body(messages, cfg, None),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        def _stream() -> Iterator[str]:
            with self._stream("POST", "/v1/chat/completions", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    usage = event.get("usage")
                    if usage:
                        counted.prompt_tokens = usage.get("prompt_tokens")
                        counted.completion_tokens = usage.get("completion_tokens")
                    for choice in event.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield text

        if self._harmony and not cfg.expose_reasoning:
            from hfl.engine.llama_cpp import _filter_gemma4_stream

            return counted.feed(_filter_gemma4_stream(_stream(), harmony=True))
        return counted.feed(_stream())

    def _completion_body(self, prompt: str, cfg: GenerationConfig) -> dict[str, Any]:
        body = {"prompt": completion_prompt(prompt, cfg), "n_predict": cfg.max_tokens,
                **_sampling(cfg)}  # fmt: skip
        if cfg.logprobs is not None:
            body["n_probs"] = max(1, cfg.logprobs)  # the drawn token is one of them
        # A response format constrains the completion too, as on /v1/chat: it
        # used to reach only the chat body (local audit B14).
        rf = cfg.response_format
        if rf == "json":
            body["json_schema"] = {}  # any JSON value
        elif isinstance(rf, dict):
            body["json_schema"] = rf
        elif isinstance(rf, str) and rf.startswith("GBNF:"):
            body["grammar"] = rf[len("GBNF:") :]
        return body

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        cfg = config or GenerationConfig()
        started = time.monotonic_ns()
        body = self._completion_body(prompt, cfg)
        if cancel.current() is not None:
            data = self._streamed_completion(body)
        else:
            response = self._post("/completion", json=body)
            response.raise_for_status()
            data = response.json()
        # Current builds say ``stop_type: "limit"``; older ones ``stopped_limit``.
        hit_limit = data.get("stop_type") == "limit" or bool(data.get("stopped_limit"))
        stop = "length" if hit_limit else "stop"
        result = self._result(data.get("content") or "", data, started, stop, None)
        if cfg.logprobs is not None:
            result.logprobs = _logprob_entries(data.get("completion_probabilities"), cfg.logprobs)
        if cfg.keep_context:
            result.context_tokens = self._context_tokens(body["prompt"] + result.text)
        return result

    def _context_tokens(self, text: str) -> list[int]:
        """Ollama's ``context``: prompt and reply as tokens, the way the
        completion read them (special tokens parsed, BOS as the model wants)."""
        try:
            done = self._post(
                "/tokenize",
                json={"content": text, "add_special": True, "parse_special": True},
                timeout=60,
            )
            done.raise_for_status()
            return [int(t) for t in done.json()["tokens"]]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            logger.warning("keep_context requested but /tokenize failed", exc_info=True)
            return []

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        cfg = config or GenerationConfig()
        counted = CountedStream()
        body = {**self._completion_body(prompt, cfg), "stream": True}

        def _stream() -> Iterator[str]:
            with self._stream("POST", "/completion", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("stop") and "tokens_predicted" in event:
                        counted.prompt_tokens = event.get("tokens_evaluated")
                        counted.completion_tokens = event.get("tokens_predicted")
                    text = event.get("content")
                    if text:
                        yield text

        return counted.feed(_stream())

    # A dispatched request (run_dispatched gives it a cancellation signal) is
    # sent streamed and reassembled into the one-piece answer: between two
    # events the signal can be checked, and leaving the stream closes the
    # connection, which makes llama-server cancel the task ("stop: cancel
    # task"). A blocking request could be neither checked nor interrupted:
    # past its budget it ran to the end holding a slot (measured: closing the
    # client from another thread left the calling thread hung).

    def _events(self, path: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """The server-sent events of ``path`` streamed, until done or cancelled."""
        with self._stream("POST", path, json={**body, "stream": True}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if cancel.cancelled():
                    raise cancel.GenerationCancelled("request cancelled")
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                yield json.loads(line[6:])

    def _streamed_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """/v1/chat/completions streamed, reassembled as its blocking answer."""
        body = {**body, "stream_options": {"include_usage": True}}
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        logprobs: list[dict[str, Any]] = []
        data: dict[str, Any] = {}
        finish = None
        for event in self._events("/v1/chat/completions", body):
            for key in ("usage", "timings"):
                if event.get(key):
                    data[key] = event[key]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                for call in delta.get("tool_calls") or []:
                    merged = calls.setdefault(
                        int(call.get("index", len(calls))),
                        {"type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if call.get("id"):
                        merged["id"] = call["id"]
                    function = call.get("function") or {}
                    merged["function"]["name"] += function.get("name") or ""
                    merged["function"]["arguments"] += function.get("arguments") or ""
                logprobs.extend((choice.get("logprobs") or {}).get("content") or [])
                finish = choice.get("finish_reason") or finish
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if any(reasoning):
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        choice_out: dict[str, Any] = {"message": message, "finish_reason": finish}
        if logprobs:
            choice_out["logprobs"] = {"content": logprobs}
        return {**data, "choices": [choice_out]}

    def _streamed_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        """/completion streamed, reassembled as its blocking answer."""
        content: list[str] = []
        probabilities: list[dict[str, Any]] = []
        data: dict[str, Any] = {}
        for event in self._events("/completion", body):
            content.append(event.get("content") or "")
            probabilities.extend(event.get("completion_probabilities") or [])
            if event.get("stop"):
                data = {
                    k: v
                    for k, v in event.items()
                    if k not in ("content", "completion_probabilities")
                }
        return {**data, "content": "".join(content), "completion_probabilities": probabilities}

    @property
    def _harmony(self) -> bool:
        """gpt-oss's template: its replies carry Harmony channels, which
        ``--reasoning-format none`` leaves in the text."""
        return "<|channel|>" in self._chat_template

    def _answer(self, text: str, cfg: GenerationConfig) -> str:
        if self._harmony and not cfg.expose_reasoning:
            from hfl.engine.llama_cpp import _strip_harmony_channels

            return _strip_harmony_channels(text)
        return text

    def _result(
        self,
        text: str,
        data: dict[str, Any],
        started_ns: int,
        finish: str | None,
        tool_calls: list[dict] | None,
    ) -> GenerationResult:
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        n_prompt = int(usage.get("prompt_tokens") or data.get("tokens_evaluated") or 0)
        n_gen = int(usage.get("completion_tokens") or data.get("tokens_predicted") or 0)
        eval_ms = float(timings.get("predicted_ms") or 0.0)
        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=n_prompt,
            tokens_per_second=(n_gen / (eval_ms / 1000.0)) if eval_ms else 0.0,
            stop_reason="length" if finish == "length" else "stop",
            tool_calls=tool_calls,
            total_duration=time.monotonic_ns() - started_ns,
            prompt_eval_duration=int(float(timings.get("prompt_ms") or 0.0) * 1e6),
            eval_duration=int(eval_ms * 1e6),
        )

    # ----------------------------------------------------------------- LoRA

    def count_prompt_tokens(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> int:
        """The prompt ``chat`` sends, rendered by llama-server's own template
        (``/apply-template``, the same body) and tokenized as its chat
        endpoint does (special tokens parsed and added)."""
        body = self._chat_body(messages, config or GenerationConfig(), tools)
        request = {k: body[k] for k in ("messages", "tools", "chat_template_kwargs") if k in body}
        rendered = self._post("/apply-template", json=request, timeout=60)
        if rendered.status_code == 404:
            raise NotImplementedError("this llama-server has no /apply-template")
        rendered.raise_for_status()
        tokens = self._post(
            "/tokenize",
            json={"content": rendered.json()["prompt"], "add_special": True, "parse_special": True},
            timeout=60,
        )
        tokens.raise_for_status()
        return len(tokens.json()["tokens"])

    # llama-server reads adapter files only when it starts, so changing them
    # starts it again with the new set: the model loads again (seconds for a
    # small one), and nothing may be decoding meanwhile — the route drains
    # every request first (``restarts_for_lora``).
    restarts_for_lora = True

    @staticmethod
    def _check_adapter_path(path: str) -> None:
        if "," in path:
            # ``--lora`` lists adapters separated by commas.
            raise ValueError(f"a LoRA adapter path cannot contain a comma: {path}")

    def _lora_args(self) -> list[str]:
        # ``--lora``, each at scale 1, then ``_set_lora_scales``: the
        # ``--lora-scaled FNAME:SCALE`` form splits at every colon, and a
        # Windows path has one after its drive letter (``C:\...``).
        if not self._loras:
            return []
        return ["--lora", ",".join(path for _, path, _ in self._loras)]

    def _lora_scales(self) -> list[float]:
        """Each adapter's scale, in order; empty when all are 1."""
        scales = [scale for _, _, scale in self._loras]
        return scales if any(scale != 1.0 for scale in scales) else []

    def _set_lora_scales(self) -> None:
        scales = self._lora_scales()
        if not scales:
            return
        try:
            done = self._post(
                "/lora-adapters",
                json=[{"id": i, "scale": scale} for i, scale in enumerate(scales)],
                timeout=30,
            )
            done.raise_for_status()
        except httpx.HTTPError as exc:
            self._http().close()
            self._client = None
            self._stop()
            raise RuntimeError(f"llama-server did not take the LoRA scales: {exc}") from exc

    def _relaunch(self) -> None:
        self.unload()
        argv = [*self._argv, *self._template_args, *self._lora_args()]
        self._launch(argv, self._model_path, self._timeout)

    def apply_lora(self, path: str, scale: float, adapter_id: str | None = None) -> None:
        """Serve the model with this adapter too, on top of any applied."""
        if not self._argv:
            raise RuntimeError("no model loaded")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self._check_adapter_path(path)
        self._loras.append((adapter_id or path, path, float(scale)))
        try:
            self._relaunch()
        except (RuntimeError, TimeoutError) as exc:
            # llama-server refused it (another model's adapter, not an
            # adapter at all): back to serving the model as it was.
            self._loras.pop()
            self._relaunch()
            raise ValueError(
                f"{Path(path).name} is not a LoRA adapter llama.cpp can load for this model"
            ) from exc

    def remove_lora(self, adapter_id: str) -> None:
        """Serve the model without this adapter."""
        found = next((entry for entry in self._loras if entry[0] == adapter_id), None)
        if found is None:
            raise RuntimeError(f"adapter {adapter_id!r} is not applied to this model")
        self._loras.remove(found)
        self._relaunch()

    # ----------------------------------------------------------- properties

    @property
    def model_name(self) -> str:
        return Path(self._model_path).name if self._model_path else ""

    @property
    def is_loaded(self) -> bool:
        # Loaded until unloaded: a process that died under the model is
        # started again by the next request (``_revive``), not reloaded
        # through the server's residency, which would load it twice.
        return self._client is not None and bool(self._model_path)

    @property
    def context_size(self) -> int:
        return self._n_ctx

    @property
    def supports_concurrent_inference(self) -> bool:
        return True

    @property
    def generates_on_all_cpu_cores(self) -> bool:
        # llama-server starts as many threads as the machine has cores.
        return self._cpu_only

    @property
    def supports_structured_output(self) -> bool:
        return True  # json_schema / grammar on both /v1/chat and /completion

    @property
    def parallel_slots(self) -> int:
        return self._slots

    @property
    def acceleration(self) -> str | None:
        return f"llama-server · {self._slots} parallel slots" if self._slots else None

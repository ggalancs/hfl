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
else keeps its usual backend). ``HFL_NUM_PARALLEL`` sets the slots when
above 1; the default is 4. The slots share one KV buffer sized to the
model's context (``--kv-unified``), so memory is what a single-slot load
would use.

The process listens on a random loopback port with a random API key, no web
UI and no ``/slots`` endpoint (it would show other requests' prompts): HFL is
its only client, and HFL's own authentication and limits stay in front.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from hfl.engine.base import (
    ChatMessage,
    CountedStream,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
    reasoning_template_vars,
    repeat_penalty_for,
)

logger = logging.getLogger(__name__)

DEFAULT_SLOTS = 4


def binary() -> str | None:
    """The ``llama-server`` to run: ``HFL_LLAMA_SERVER_BIN`` or the PATH's."""
    configured = os.environ.get("HFL_LLAMA_SERVER_BIN")
    if configured:
        return configured if Path(configured).is_file() else None
    return shutil.which("llama-server")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _slots() -> int:
    from hfl.config import config

    configured = int(getattr(config, "queue_max_inflight", 1) or 1)
    return configured if configured > 1 else DEFAULT_SLOTS


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


def _as_marker(call: dict) -> str:
    """A structured call written back as the ``<tool_call>`` text HFL's
    parsers read, for the streaming path that only carries text."""
    fn = call["function"]
    return (
        f"<tool_call>{json.dumps({'name': fn['name'], 'arguments': fn['arguments']})}</tool_call>"
    )


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
        guarded = [
            sys.executable,
            "-m",
            "hfl.engine._child_guard",
            str(os.getpid()),
            "--",
            *argv,
        ]
        proc = subprocess.Popen(
            guarded,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "LLAMA_API_KEY": key},
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
    )
    return proc, client


def stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    """Stop a llama-server started by ``start_server`` (SIGTERM, then kill)."""
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


class LlamaServerEngine(InferenceEngine):
    """One ``llama-server`` child process serving one GGUF model."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._client: httpx.Client | None = None
        self._model_path = ""
        self._n_ctx = 0
        self._slots = 0
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

    # ------------------------------------------------------------------ life

    def load(self, model_path: str, **kwargs: Any) -> None:
        from hfl.config import config
        from hfl.engine.llama_cpp import _read_gguf_model_info, resolve_n_ctx

        exe = binary()
        if exe is None:
            raise RuntimeError(
                "llama-server was not found: install llama.cpp (e.g. `brew install llama.cpp`) "
                "or set HFL_LLAMA_SERVER_BIN"
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
            "--jinja",
            "--reasoning-format",
            "none",
            "--no-webui",
            "--no-slots",
            # A vision model's projector: llama-server then takes images.
            *(["--mmproj", str(self._projector)] if self._projector else []),
        ]
        log_dir = config.home_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"llama-server-{Path(model_path).stem}.log"
        timeout = float(getattr(config, "model_load_timeout", 600) or 600)
        self._argv, self._template_args, self._timeout = base_argv, [], timeout
        self._launch([*base_argv, *self._lora_args()], model_path, timeout)
        fixed = self._template_with_bos(config.home_dir / "templates", Path(model_path).stem)
        if fixed is not None:
            # Once more with the template that writes BOS (see
            # ``_template_with_bos``): llama-server reads it only at start.
            logger.info("Chat template does not start with BOS; HFL adds it")
            self._template_args = ["--chat-template-file", str(fixed)]
            self.unload()
            argv = [*base_argv, *self._template_args, *self._lora_args()]
            self._launch(argv, model_path, timeout)
        self._read_template()
        if not n_ctx:
            n_ctx = self._reported_ctx()
        self._model_path, self._n_ctx, self._slots = model_path, n_ctx, slots
        logger.info(
            "llama-server serving %s: %d-token context shared by %d parallel slots",
            Path(model_path).name,
            n_ctx,
            slots,
        )

    def _launch(self, base_argv: list[str], model_path: str, timeout: float) -> None:
        assert self._log_path is not None
        self._proc, self._client = start_server(base_argv, model_path, self._log_path, timeout)

    def _props(self) -> dict[str, Any]:
        try:
            props = self._http().get("/props", timeout=10).json()
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
        props = self._props()
        template, bos = props.get("chat_template"), props.get("bos_token")
        if not isinstance(template, str) or not isinstance(bos, str) or not bos:
            return None
        if "bos_token" in template or bos in template:
            return None
        try:
            with_special = self._http().post(
                "/tokenize", json={"content": "a", "add_special": True}, timeout=10
            )
            without = self._http().post(
                "/tokenize", json={"content": "a", "add_special": False}, timeout=10
            )
            adds_bos = len(with_special.json()["tokens"]) > len(without.json()["tokens"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return None
        if not adds_bos:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{stem}.jinja"
        path.write_text("{{ bos_token }}" + template, encoding="utf-8")
        return path

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
            self._client.close()
            self._client = None
        self._stop()

    # ------------------------------------------------------------- requests

    def _http(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("llama-server engine is not loaded")
        return self._client

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
        response = self._http().post(
            "/v1/chat/completions", json=self._chat_body(messages, cfg, tools)
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        return self._result(
            self._answer(message.get("content") or "", cfg),
            data,
            started,
            choice.get("finish_reason"),
            _canonical_tool_calls(message.get("tool_calls")),
        )

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
            with self._http().stream("POST", "/v1/chat/completions", json=body) as response:
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
        return {"prompt": prompt, "n_predict": cfg.max_tokens, **_sampling(cfg)}

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        cfg = config or GenerationConfig()
        started = time.monotonic_ns()
        response = self._http().post("/completion", json=self._completion_body(prompt, cfg))
        response.raise_for_status()
        data = response.json()
        stop = "length" if data.get("stopped_limit") else "stop"
        return self._result(data.get("content") or "", data, started, stop, None)

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        cfg = config or GenerationConfig()
        counted = CountedStream()
        body = {**self._completion_body(prompt, cfg), "stream": True}

        def _stream() -> Iterator[str]:
            with self._http().stream("POST", "/completion", json=body) as response:
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

    # llama-server reads adapter files only when it starts, so changing them
    # starts it again with the new set: the model loads again (seconds for a
    # small one), and nothing may be decoding meanwhile — the route drains
    # every request first (``restarts_for_lora``).
    restarts_for_lora = True

    @staticmethod
    def _check_adapter_path(path: str) -> None:
        if "," in path:
            # ``--lora-scaled`` lists adapters separated by commas.
            raise ValueError(f"a LoRA adapter path cannot contain a comma: {path}")

    def _lora_args(self) -> list[str]:
        if not self._loras:
            return []
        return ["--lora-scaled", ",".join(f"{path}:{scale}" for _, path, scale in self._loras)]

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
        return self._proc is not None and self._proc.poll() is None

    @property
    def context_size(self) -> int:
        return self._n_ctx

    @property
    def supports_concurrent_inference(self) -> bool:
        return True

    @property
    def parallel_slots(self) -> int:
        return self._slots

    @property
    def acceleration(self) -> str | None:
        return f"llama-server · {self._slots} parallel slots" if self._slots else None

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Inference backend based on llama-cpp-python.

This is the main backend for GGUF models.
Supports CPU, CUDA, Metal, and Vulkan.
"""

import contextlib
import functools
import logging
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, cast

from hfl.engine.base import (
    ChatMessage,
    CountedStream,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
    held,
    reasoning_template_vars,
    repeat_penalty_for,
)

# ``llama_cpp`` is an optional dependency (HFL ``[llama]`` extra). The
# import is wrapped in try/except so the rest of the module — including
# the GGUF chat-format detection helper, the architecture map, and any
# test that monkeypatches ``hfl.engine.llama_cpp.Llama`` — can be
# imported and exercised in environments without llama-cpp-python (CI
# default, doc generators, type checkers).
if TYPE_CHECKING:
    # Keep the real ``Llama`` type visible to type checkers regardless of
    # whether the optional backend is installed in the checking env.
    from llama_cpp import Llama
else:
    try:
        from llama_cpp import Llama
    except ImportError:  # pragma: no cover — exercised by the [dev] CI matrix
        Llama = None

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_stderr():
    """Temporarily suppresses stderr (to silence Metal/CUDA logs)."""
    # File descriptor 2 itself: that is where the C library writes. Python's
    # ``sys.stderr`` is not always on it (a test runner, a redirected
    # logger), and silencing whatever it points at let the noise through.
    stderr_fd = 2
    with contextlib.suppress(Exception):
        sys.stderr.flush()
    saved_fd = os.dup(stderr_fd)
    try:
        # Redirect stderr to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restore stderr
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


@contextmanager
def _capture_llama_log(sink: list[str]):
    """Divert ggml/llama.cpp's log into ``sink`` for the duration of a load.

    Replaces the library's global log callback rather than redirecting the
    stderr file descriptor. Redirecting the fd does not work here: with
    ``verbose=False`` llama-cpp-python's own callback checks the level of the
    ``llama-cpp-python`` logger and drops INFO lines *before* they ever reach
    stderr — and raising that level is futile because ``Llama.__init__``
    calls ``set_verbose()`` itself, resetting it mid-construction. Swapping
    the callback is immune to both.

    The user's terminal stays as quiet as it was with the old
    stderr-to-/dev/null suppression; the difference is that the one line that
    matters ("offloaded 41/41 layers to GPU") is now kept instead of thrown
    away, which is why a fully Metal-accelerated load used to look identical
    to a CPU-only one.

    Capture is strictly best-effort: this is diagnostics, never a reason for
    a load to fail. Any missing symbol, stubbed module or changed API shape
    degrades to a no-op (empty ``sink`` -> "acceleration unknown"). Tests that
    swap ``llama_cpp.llama_cpp`` for a fake module exercise exactly that path.
    """
    import ctypes

    installed: Any = None
    lcpp: Any = None
    pkg: Any = None
    try:
        # Import under a second name and assign: binding an *annotated* name
        # directly from an ``import`` is a redefinition for mypy when
        # llama_cpp is absent (the CI venv omits the [llama] extra), while
        # the annotation is what lets the ``except`` branch leave it None.
        import llama_cpp as _pkg_module
        from llama_cpp import llama_cpp as _lcpp_module

        pkg = _pkg_module
        lcpp = _lcpp_module
        installed = pkg.llama_log_callback(_make_sink_callback(sink))
        # Keep a strong reference for the whole window: ctypes callbacks are
        # garbage-collectable, and letting one die while C still holds the
        # pointer is a segfault, not an exception.
        _ACTIVE_LOG_CALLBACKS.append(installed)
        lcpp.llama_log_set(installed, ctypes.c_void_p(0))
    except Exception:
        logger.debug("llama.cpp log capture unavailable", exc_info=True)
        if installed is not None:
            with contextlib.suppress(ValueError):
                _ACTIVE_LOG_CALLBACKS.remove(installed)
        yield
        return

    try:
        yield
    finally:
        try:
            # Restore llama-cpp-python's own callback so normal logging
            # behaviour resumes for anything outside the load.
            lcpp.llama_log_set(getattr(pkg._logger, "llama_log_callback", None), ctypes.c_void_p(0))
        except Exception:  # pragma: no cover — defensive
            pass
        with contextlib.suppress(ValueError):
            _ACTIVE_LOG_CALLBACKS.remove(installed)


# Strong references to in-flight ctypes callbacks (see _capture_llama_log).
_ACTIVE_LOG_CALLBACKS: list[Any] = []


def _make_sink_callback(sink: list[str]):
    """Build the plain Python function a ggml log callback wraps."""

    def _cb(level: int, text: bytes, user_data: Any) -> None:
        try:
            sink.append(text.decode("utf-8", errors="replace"))
        except Exception:  # pragma: no cover — never raise into C
            pass

    return _cb


# ``load_tensors: offloaded 41/41 layers to GPU``
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
# ``load_tensors:  MTL0_Mapped model buffer size =  8579.06 MiB`` — the device
# prefix identifies the backend (MTL0 = Metal, CUDA0, ROCm0, Vulkan0, ...).
_DEVICE_BUFFER_RE = re.compile(
    r"^\s*\S*?:\s+(\w+?)(?:_Mapped)?\s+model buffer size\s*=\s*([\d.]+)\s*MiB",
    re.MULTILINE,
)
# ``ggml_metal_device_init: GPU name:   MTL0 (Apple M3 Max)``
_GPU_NAME_RE = re.compile(r"GPU name:\s*(.+?)\s*$", re.MULTILINE)


def _summarize_acceleration(log_text: str) -> str | None:
    """Turn llama.cpp's loader dump into one human-readable INFO line.

    Returns ``None`` when the text carries no offload information (an old
    llama.cpp, a captured log we failed to read, or ``verbose=True``, where
    the output went to the terminal instead).
    """
    if not log_text:
        return None

    parts: list[str] = []

    gpu_name = _GPU_NAME_RE.search(log_text)
    if gpu_name:
        parts.append(gpu_name.group(1))

    offload = _OFFLOAD_RE.search(log_text)
    if offload:
        done, total = offload.group(1), offload.group(2)
        parts.append(f"{done}/{total} layers on GPU")

    # Sum the per-device weight buffers, skipping the CPU one, so the line
    # states how much actually landed on the accelerator.
    device_mib = 0.0
    devices: set[str] = set()
    for device, mib in _DEVICE_BUFFER_RE.findall(log_text):
        if device.upper() == "CPU":
            continue
        devices.add(device)
        device_mib += float(mib)
    if device_mib > 0:
        parts.append(f"{device_mib / 1024:.1f} GiB in {'/'.join(sorted(devices))} buffers")

    if not parts:
        return None
    return " · ".join(parts)


def _perf_reset(model: Any) -> None:
    """Zero llama.cpp's per-context perf counters before a generation.

    They accumulate across calls, so without a reset the "this request"
    numbers would be lifetime totals. Best-effort: silently does nothing on
    builds that don't expose the API.
    """
    try:
        from llama_cpp import llama_cpp as _lcpp

        _lcpp.llama_perf_context_reset(model._ctx.ctx)
    except Exception:  # pragma: no cover — optional/older backend
        logger.debug("llama_perf_context_reset unavailable", exc_info=True)


def _perf_read(model: Any) -> tuple[int, int, int, int] | None:
    """Read llama.cpp's *measured* prompt-eval and eval timings.

    Returns ``(prompt_ns, n_prompt_eval, eval_ns, n_eval)`` or ``None`` when
    the backend doesn't expose the counters.

    This exists because the previous approach — splitting total wall-clock in
    proportion to the token counts — is not a measurement, it is an
    assumption that prefill and generation run at the same tokens/second.
    They do not: prefill is compute-bound and roughly an order of magnitude
    faster per token than memory-bound generation. On a 72B with a
    4161-token prompt and 250 generated tokens, the proportional split
    attributed 94% of the time to the prompt and reported generation at
    44.8 tok/s — a physically impossible figure on hardware whose ceiling is
    about 9 tok/s. llama.cpp measures both phases properly; use its numbers.
    """
    try:
        from llama_cpp import llama_cpp as _lcpp

        data = _lcpp.llama_perf_context(model._ctx.ctx)
        return (
            int(data.t_p_eval_ms * 1_000_000),
            int(data.n_p_eval),
            int(data.t_eval_ms * 1_000_000),
            int(data.n_eval),
        )
    except Exception:  # pragma: no cover — optional/older backend
        logger.debug("llama_perf_context unavailable", exc_info=True)
        return None


def _split_durations(model: Any, total_ns: int, n_prompt: int, n_gen: int) -> tuple[int, int]:
    """Return ``(prompt_eval_ns, eval_ns)`` for a finished generation.

    Prefers llama.cpp's measured counters (see :func:`_perf_read`). Falls
    back to the old token-proportional estimate only when the backend can't
    report them — that estimate is documented as approximate precisely
    because it assumes both phases run at the same rate, which is wrong by
    roughly an order of magnitude.
    """
    perf = _perf_read(model)
    if perf is not None:
        prompt_ns, _, eval_ns, n_eval = perf
        # Trust the counters only when they actually recorded this call;
        # a zeroed struct means the build reports nothing useful.
        if eval_ns > 0 or prompt_ns > 0:
            return prompt_ns, eval_ns
        del n_eval
    total_tokens = max(1, n_prompt + n_gen)
    prompt_eval_ns = int(total_ns * n_prompt / total_tokens)
    return prompt_eval_ns, total_ns - prompt_eval_ns


def _warn_if_on_battery() -> None:
    """Log once when a macOS host is running inference on battery.

    Worth a WARNING rather than INFO: measured 8x slower token generation
    on an M3 Max (see :mod:`hfl.engine.power`), and the symptom — every
    layer offloaded to the GPU and still crawling — otherwise reads as a
    broken install.
    """
    global _BATTERY_WARNED
    if _BATTERY_WARNED:
        return
    try:
        from hfl.engine.power import on_battery
    except Exception:  # pragma: no cover — defensive
        return
    if on_battery():
        _BATTERY_WARNED = True
        logger.warning(
            "Running on battery power: macOS clocks the GPU down hard — "
            "measured 8x slower token generation on an M3 Max. Plug in the "
            "charger for full speed."
        )


_BATTERY_WARNED = False


@contextmanager
def _nullcontext():
    """Context manager that does nothing (for when verbose=True)."""
    yield


# Map from GGUF ``general.architecture`` to the chat_format string that
# llama-cpp-python ships built-in. We only override when the GGUF lacks a
# usable ``tokenizer.chat_template`` AND when llama-cpp-python's own
# auto-detection guesses wrong (it falls back to a Llama-2 [INST] format
# for unknown architectures, which silently destroys the chat quality of
# Gemma family models).
_ARCHITECTURE_CHAT_FORMAT: dict[str, str] = {
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "gemma4": "gemma",
}


# Per-architecture safe cap for ``n_ctx`` when the caller doesn't pass
# an explicit value. The Gemma 3/4 family advertises 131072-token
# contexts in GGUF metadata; the resulting fp16 KV cache (tens of GB
# even at 9B, >140GB at 27B) pins unified memory on Apple Silicon and
# can kernel-panic the host. Cap to a safe default unless the caller
# explicitly opts in via ``n_ctx=`` (or ``--ctx`` at the CLI).
_ARCHITECTURE_CTX_CAP: dict[str, int] = {
    "gemma3": 8192,
    "gemma4": 8192,
}

# Architectures where llama-cpp-python's flash-attention path is not
# yet safe. For these we force ``flash_attn=False`` unless the caller
# explicitly passes ``flash_attn=True``.
_ARCHITECTURE_NO_FLASH_ATTN: set[str] = {"gemma4"}


def _image_data_uri(image_bytes: bytes) -> str:
    """An image as the ``data:image/...;base64,...`` URI that llama.cpp's
    multimodal handlers (and llama-server's OpenAI endpoint) take on
    ``image_url.url``, its MIME type sniffed from the magic bytes."""
    import base64

    if image_bytes.startswith(b"\x89PNG"):
        mime = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif image_bytes.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/png"  # Best-effort fallback.
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")


_TOOL_PROBE = "hfl_probe_tool_7f3a"
_SYSTEM_PROBE = "hfl-probe-system-7f3a"


@functools.lru_cache(maxsize=64)
def _probe_template(template: str) -> tuple[bool, bool]:
    """``(lists the tools, takes a system message)``, found by rendering the
    template with a stand-in tool and system message — as llama.cpp does —
    rather than by reading it.

    Reading was wrong both ways: Phi-4-mini's template names ``tools`` but
    reads them from the system message, SmolLM3's takes ``xml_tools``, so
    they were handed tools they never showed the model; Mistral's rejects a
    system message outright. A template that cannot be rendered here is
    taken not to show tools (HFL then writes them in: at worst the model
    sees them twice, instead of never) and to take a system message.
    """
    import json as _json
    from datetime import datetime

    try:
        import jinja2
        import jinja2.ext
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:  # not a core dependency; every backend that renders
        return False, True  # templates in-process brings it

    def _raise(message: str) -> None:
        raise jinja2.TemplateError(message)

    def _tojson(value: Any, ensure_ascii: bool = False, indent: Any = None, **_: Any) -> str:
        return _json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)

    class _IgnoreGeneration(jinja2.ext.Extension):
        """``{% generation %}`` (SmolLM3's template): its content, as
        llama-cpp-python and Transformers render it."""

        tags = {"generation"}

        def parse(self, parser: Any) -> Any:
            next(parser.stream)
            return parser.parse_statements(("name:endgeneration",), drop_needle=True)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols, _IgnoreGeneration]
    )
    env.filters["tojson"] = _tojson
    env.globals.update(raise_exception=_raise, strftime_now=lambda f: datetime.now().strftime(f))
    tool = {
        "type": "function",
        "function": {
            "name": _TOOL_PROBE,
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    user = {"role": "user", "content": "hi"}

    def render(messages: list[dict]) -> str | None:
        try:
            compiled = env.from_string(template)
            return str(
                compiled.render(
                    messages=messages,
                    tools=[tool],
                    add_generation_prompt=True,
                    bos_token="",
                    eos_token="",
                )
            )
        except Exception:
            return None

    with_system = render([{"role": "system", "content": _SYSTEM_PROBE}, user])
    takes_system = with_system is not None and _SYSTEM_PROBE in with_system
    out = with_system if takes_system else render([user])
    if out is None:
        return False, True
    return _TOOL_PROBE in out, takes_system


def _template_renders_tools(template: str, chat_format: str | None) -> bool:
    """Whether the prompt the model gets will list the request's tools.

    Many templates do not — Hermes-3 ships plain ChatML, DeepSeek's render
    past calls but never the tool list, Phi-4-mini's and SmolLM3's want
    them elsewhere. A static ``chat_format`` does only if it is a
    function-calling one.
    """
    if chat_format:
        return "function" in chat_format
    return bool(template) and _probe_template(template)[0]


def _template_takes_system(template: str) -> bool:
    """Whether a system message survives the template (Mistral's rejects
    one); unknown or no template: assumed to."""
    return not template or _probe_template(template)[1]


def _fold_system(msgs: list[dict]) -> list[dict]:
    """System messages as the start of the first user message, for a
    template that takes none (Mistral: "Only user and assistant roles")."""
    system = "\n\n".join(
        str(m.get("content") or "") for m in msgs if m.get("role") == "system"
    ).strip()
    rest = [m for m in msgs if m.get("role") != "system"]
    if not system:
        return rest
    for i, m in enumerate(rest):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            rest[i] = {**m, "content": f"{system}\n\n{m['content']}"}
            return rest
    return [{"role": "user", "content": system}, *rest]


# The tool prompt HFL writes in when the model's template has none: the
# Hermes function-calling prompt, word for word — the convention most open
# models were trained on, and whose ``<tool_call>`` replies
# ``hfl.api.tool_parsers`` reads. (A paraphrase of it was measured to fail:
# Hermes-3 3B then garbled the opening marker.)
_TOOLS_PROMPT = (
    "You are a function calling AI model. You are provided with function "
    "signatures within <tools></tools> XML tags. You may call one or more "
    "functions to assist with the user query. Don't make assumptions about "
    "what values to plug into functions. Here are the available tools: "
    "<tools> {tools} </tools> "
    "Use the following pydantic model json schema for each tool call you will "
    'make: {{"properties": {{"arguments": {{"title": "Arguments", "type": '
    '"object"}}, "name": {{"title": "Name", "type": "string"}}}}, "required": '
    '["arguments", "name"], "title": "FunctionCall", "type": "object"}} '
    "For each function call return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags as follows:\n"
    '<tool_call>\n{{"arguments": <args-dict>, "name": <function-name>}}\n</tool_call>'
)


def _install_template_formatters(model: Any) -> tuple[list[Any], bool]:
    """Serve every GGUF chat template of ``model`` through HFL's formatter.

    Two things llama-cpp-python's own handler cannot do:

    - **Variables per request.** It renders a template with the messages and
      tools only, so ``enable_thinking`` / ``reasoning_effort`` never reached
      it: ``think: false`` hid the reasoning but the model still reasoned.
      HFL's formatter adds the variables set on it (``template_vars``).
    - **BOS.** It tokenizes the prompt without adding BOS, so a template
      that does not write ``{{ bos_token }}`` — Hermes-3's plain ChatML — ran
      with none, and Hermes-3 3B answered a tool prompt with broken JSON
      (measured). Such a template gets BOS when the vocabulary wants one.

    Returns the formatters installed and whether any template got BOS.
    """
    try:
        from llama_cpp import llama_chat_format

        vocab = model._model
        bos_id = vocab.token_bos()
        bos = vocab.token_get_text(bos_id) if bos_id >= 0 else ""
        wants_bos = bool(bos) and bool(vocab.add_bos_token())
        eos_id = vocab.token_eos()
        eos = vocab.token_get_text(eos_id) if eos_id >= 0 else ""
    except Exception:  # an older llama-cpp-python, or a stand-in in tests
        return [], False

    class _Formatter(llama_chat_format.Jinja2ChatFormatter):
        template_vars: dict[str, Any] = {}
        hfl_name = ""  # the name llama-cpp-python registers its handler under

        def __call__(self, **kwargs: Any) -> Any:
            return super().__call__(**{**self.template_vars, **kwargs})

    formatters: list[Any] = []
    added_bos = False
    for key, template in list((getattr(model, "metadata", None) or {}).items()):
        if key != "tokenizer.chat_template" and not key.startswith("tokenizer.chat_template."):
            continue
        if not isinstance(template, str):
            continue
        if wants_bos and "bos_token" not in template and bos not in template:
            template = "{{ bos_token }}" + template
            added_bos = True
        formatter = _Formatter(
            template=template, eos_token=eos, bos_token=bos, stop_token_ids=[eos_id]
        )
        formatter.template_vars = {}
        # The names llama-cpp-python registers them under.
        name = "chat_template.default" if key == "tokenizer.chat_template" else key[10:]
        formatter.hfl_name = name
        model._chat_handlers[name] = formatter.to_chat_handler()
        formatters.append(formatter)
    return formatters, added_bos


def _render_special_tokens(model: Any, on: bool) -> None:
    """Make ``model``'s completions keep (or drop, the library's default)
    control tokens in their text.

    A tool call is often written with them — Hermes' ``<tool_call>``,
    gpt-oss's ``<|channel|>``/``<|call|>``, DeepSeek's ``<｜tool▁call▁begin｜>``
    — and llama-cpp-python drops them when it turns tokens into text, so
    the parsers would never see the call. Only requests with tools turn it
    on, and every request sets it, so a stream abandoned half-way cannot
    leave it on for the next one.
    """
    import functools

    detokenize = getattr(type(model), "detokenize", None)
    if on and detokenize is not None:
        model.detokenize = functools.partial(detokenize, model, special=True)
    elif "detokenize" in getattr(model, "__dict__", {}):
        del model.detokenize


def _history_for_template(msgs: list[dict], template: str) -> list[dict]:
    """Past calls and results in the shape the template renders.

    GLM-4-0414's template drops ``tool_calls`` and ``role: "tool"``: it
    renders a call as an assistant turn whose ``metadata`` is the function
    and whose content its JSON arguments, and a result as an
    ``observation`` turn. Without this the model never saw the result and
    called the tool again (measured). Other templates get the history as is.
    """
    if not ("metadata" in template and "observation" in template):
        return msgs
    import json as _json

    out: list[dict] = []
    for msg in msgs:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            if msg.get("content"):
                out.append({"role": "assistant", "content": msg["content"]})
            for call in msg["tool_calls"]:
                fn = call.get("function", call)
                args = fn.get("arguments", {})
                text = args if isinstance(args, str) else _json.dumps(args, ensure_ascii=False)
                out.append({"role": "assistant", "metadata": fn.get("name", ""), "content": text})
        elif msg.get("role") == "tool":
            out.append({"role": "observation", "content": msg.get("content") or ""})
        else:
            out.append(msg)
    return out


def _tools_as_text(msgs: list[dict], tools: list[dict] | None) -> list[dict]:
    """Write the tools, past calls and their results into plain messages,
    for a template that would drop them: the tool list goes into the
    system message, an assistant's calls become ``<tool_call>`` text and
    tool results a user turn of ``<tool_response>`` blocks (consecutive
    results merged, so turns still alternate for strict templates).

    Also for a template that renders past calls but not the tool list
    (DeepSeek's): its own rendering ends the prompt after the tool output
    without reopening the assistant turn, and DeepSeek-R1-0528 8B then
    answered nothing; with this text it answered from the result (measured).
    """
    import json as _json

    out: list[dict] = []
    merged_results = False  # the last message is a user turn of tool results
    for msg in msgs:
        role = msg.get("role")
        if role == "tool":
            block = f"<tool_response>\n{msg.get('content') or ''}\n</tool_response>"
            if merged_results:
                out[-1]["content"] += "\n" + block
            else:
                out.append({"role": "user", "content": block})
            merged_results = True
            continue
        merged_results = False
        if role == "assistant" and msg.get("tool_calls"):
            calls = []
            for call in msg["tool_calls"]:
                fn = call.get("function", call)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except ValueError:
                        pass
                payload = _json.dumps({"name": fn.get("name", ""), "arguments": args})
                calls.append(f"<tool_call>\n{payload}\n</tool_call>")
            text = msg.get("content") or ""
            out.append({"role": "assistant", "content": "\n".join([text, *calls]).strip()})
        else:
            out.append(msg)
    if tools:
        listing = " ".join(_json.dumps(t, ensure_ascii=False) for t in tools)
        prompt = _TOOLS_PROMPT.format(tools=listing)
        if out and out[0].get("role") == "system" and isinstance(out[0].get("content"), str):
            out[0] = {**out[0], "content": f"{out[0]['content']}\n\n{prompt}".strip()}
        else:
            out.insert(0, {"role": "system", "content": prompt})
    return out


# Architectures whose vocabulary contains split-pipe channel/think/turn
# markers (``<|channel>...<channel|>`` etc.) that need post-filtering
# on chat output. This is a separate set from ``_ARCHITECTURE_CHAT_FORMAT``
# because the override to ``gemma`` (= Gemma 2 format) is not enough to
# stop the model from emitting its reasoning-format markers when the
# GGUF doesn't ship a proper ``tokenizer.chat_template``.
_ARCHITECTURE_CHANNEL_FILTER: set[str] = {"gemma4", "gpt-oss"}

# Regexes that strip Gemma 4-family split-pipe markers from chat output.
# Applied to the text in ``chat()`` and to the stream in ``chat_stream()``
# (with a small buffering filter so markers split across chunks still
# match). Stripping logic:
#
#   1. Entire ``thought`` / ``think`` channels are suppressed, content
#      included — the user asked for a clean answer, not the CoT.
#   2. Orphan opening markers like ``<|channel>final`` (with optional
#      channel/turn name + optional newline) are dropped, keeping the
#      content that follows.
#   3. Orphan closing markers like ``<channel|>`` / ``<turn|>`` are
#      dropped.
#
# Tool markers (``<|tool>``, ``<|tool_call>``, ``<|tool_response>`` and
# their closers) are deliberately NOT stripped by this filter — they
# carry the payload that :func:`hfl.api.tool_parsers.parse_gemma4`
# extracts at the route layer. The route runs ``parse_tool_calls`` on
# the engine's output to recover structured ``tool_calls``, and would
# find nothing if we pre-stripped the markers here.
#
# The exact tag names come from the model's vocabulary (tokens 98–218
# on the bartowski/google_gemma-4-31B GGUF used in production).
_GEMMA4_THOUGHT_BLOCK = re.compile(r"<\|channel>thought[\s\S]*?<channel\|>")
_GEMMA4_THINK_BLOCK = re.compile(r"<\|think>[\s\S]*?<think\|>")
_GEMMA4_OPEN_MARKER = re.compile(r"<\|(?:channel|turn)>[a-z_]*\n?")
_GEMMA4_CLOSE_MARKER = re.compile(r"<(?:channel|turn|think)\|>")

# Maximum length of any single marker — used by the streaming filter
# to decide how much of the tail it must hold back to avoid emitting a
# partially-seen marker. ``<|tool_response>`` is the longest at 16
# chars, plus room for a short channel name and a newline.
_GEMMA4_MAX_MARKER_LEN = 32


def _strip_gemma4_channel_markers(text: str) -> str:
    """Remove split-pipe channel/think/turn markers from a finished
    chat response. See ``_ARCHITECTURE_CHANNEL_FILTER`` for the
    rationale. Safe to call on text that doesn't contain any markers
    — it's a no-op."""
    text = _GEMMA4_THOUGHT_BLOCK.sub("", text)
    text = _GEMMA4_THINK_BLOCK.sub("", text)
    text = _GEMMA4_OPEN_MARKER.sub("", text)
    text = _GEMMA4_CLOSE_MARKER.sub("", text)
    return text


# Ordered list of known markers used by the streaming filter. Must be
# longest-prefix-first so the matcher prefers ``<|channel>thought`` over
# the shorter ``<|channel>`` when both could apply.
#
# Tuples are ``(marker_string, kind)`` with kind in:
#   - ``thought_open`` / ``think_open``: switch the filter to suppress
#     state until the matching close marker arrives
#   - ``final_open``:                    strip the marker (keep content
#     that follows)
#   - ``open``:                           strip the marker (no state
#     change)
#   - ``close``:                          strip the marker and exit
#     suppress state if we were in one
_GEMMA4_STREAM_MARKERS: list[tuple[str, str]] = [
    ("<|channel>thought", "thought_open"),
    ("<|channel>final", "final_open"),
    ("<|channel>", "open"),
    ("<channel|>", "close"),
    ("<|think>", "think_open"),
    ("<think|>", "close"),
    ("<|turn>", "open"),
    ("<turn|>", "close"),
    # ``<|tool*>`` markers are deliberately omitted — they carry
    # tool-call payload that ``hfl.api.tool_parsers.parse_gemma4``
    # needs to extract. The streaming filter lets them through as
    # plain text (character-by-character fallthrough) so the route
    # can re-parse the accumulated stream at the end.
]


# gpt-oss writes every reply in OpenAI's Harmony format — its reasoning in
# an ``analysis`` channel, the answer in ``final``, tool calls in
# ``commentary`` — and its GGUF keeps those markers in the text, so a plain
# ``hfl run gpt-oss`` showed the whole chain of thought as the answer
# (measured on unsloth/gpt-oss-20b-GGUF). The reasoning is dropped as
# Gemma 4's is; the ``commentary`` tool-call markers are left for
# ``hfl.api.tool_parsers.parse_harmony``.
_HARMONY_ANALYSIS_BLOCK = re.compile(r"<\|channel\|>analysis<\|message\|>[\s\S]*?(?:<\|end\|>|$)")
_HARMONY_MARKERS = re.compile(
    r"<\|start\|>assistant|<\|channel\|>final<\|message\|>|<\|end\|>|<\|return\|>"
)

_HARMONY_STREAM_MARKERS: list[tuple[str, str]] = [
    ("<|channel|>analysis<|message|>", "thought_open"),
    ("<|channel|>final<|message|>", "final_open"),
    ("<|start|>assistant", "open"),
    ("<|end|>", "close"),
    ("<|return|>", "close"),
]


def _strip_harmony_channels(text: str) -> str:
    """The answer of a finished Harmony reply: reasoning out, channel
    markers out, tool-call markers kept."""
    return _HARMONY_MARKERS.sub("", _HARMONY_ANALYSIS_BLOCK.sub("", text))


def _strip_channel_markers(text: str, architecture: str | None) -> str:
    if architecture == "gpt-oss":
        return _strip_harmony_channels(text)
    return _strip_gemma4_channel_markers(text)


class _Gemma4StreamFilter:
    """Stateful char-level filter that strips Gemma 4 channel/think/
    turn markers from a token stream.

    Handles markers split across chunks (a single token like
    ``<|channel>`` frequently arrives on its own boundary) by
    holding the buffer until either a known marker completes, the
    tail becomes too long to be a partial marker, or the stream
    ends. Trade-off vs. the whole-buffer regex approach: the filter
    is stateful across ``feed`` calls, so it correctly suppresses
    multi-chunk thought blocks of arbitrary length.
    """

    def __init__(self, harmony: bool = False) -> None:
        self._buffer: str = ""
        # ``True`` while we're inside a thought/think block that
        # should be fully suppressed (content included).
        self._suppress: bool = False
        # The same filter reads gpt-oss's Harmony markers.
        self._markers = _HARMONY_STREAM_MARKERS if harmony else _GEMMA4_STREAM_MARKERS
        self._strip = _strip_harmony_channels if harmony else _strip_gemma4_channel_markers

    def feed(self, chunk: str) -> str:
        """Feed a new chunk, return whatever text is safe to emit now."""
        if not chunk:
            return ""
        self._buffer += chunk
        out: list[str] = []
        while self._buffer:
            c = self._buffer[0]
            if c != "<":
                if not self._suppress:
                    out.append(c)
                self._buffer = self._buffer[1:]
                continue
            # Potential marker start. Walk through each known marker
            # tracking both the longest full match and whether any
            # (necessarily longer) marker could still grow into the
            # buffer once more data arrives. The "could_grow" branch
            # is what prevents us from committing to a short match
            # like ``<|channel>`` when ``<|channel>thought`` is still
            # in flight — without it, the streaming filter would
            # consume the prefix and silently leak the ``thought\n``
            # content that arrives in the next chunk.
            matched: tuple[str, str] | None = None
            could_grow = False
            for marker, kind in self._markers:
                if self._buffer.startswith(marker):
                    if matched is None or len(marker) > len(matched[0]):
                        matched = (marker, kind)
                elif len(self._buffer) < len(marker) and marker.startswith(self._buffer):
                    could_grow = True
            if could_grow:
                # Some longer marker might still match. Wait for more
                # data regardless of whether we already have a shorter
                # tentative match.
                return "".join(out)
            if matched is None:
                # Bare ``<`` that can't be any known marker. Emit it.
                if not self._suppress:
                    out.append("<")
                self._buffer = self._buffer[1:]
                continue
            # Found a complete marker. Apply its side effect and
            # consume it from the buffer.
            marker, kind = matched
            if kind in ("thought_open", "think_open"):
                self._suppress = True
            elif kind == "close" and self._suppress:
                self._suppress = False
            self._buffer = self._buffer[len(marker) :]
            # Open markers are normally followed by an immediate
            # newline (``<|channel>final\n``). Consume it so it
            # doesn't leak into the emitted output.
            if kind in ("thought_open", "think_open", "final_open", "open"):
                if self._buffer.startswith("\n"):
                    self._buffer = self._buffer[1:]
        return "".join(out)

    def flush(self) -> str:
        """Finalise the stream: emit what's left, or drop it if we're
        still inside an unclosed thought/think block."""
        if self._suppress:
            # Incomplete suppressed block: drop the remainder rather
            # than leaking partial reasoning text.
            self._buffer = ""
            return ""
        # Any bytes left over at EOF are plain text (possibly with
        # orphan markers); run the strip one last time to clean
        # those up before emitting.
        leftover = self._buffer
        self._buffer = ""
        return self._strip(leftover)


def _filter_gemma4_stream(iterator: Iterator[str], harmony: bool = False) -> Iterator[str]:
    """Stream wrapper around :class:`_Gemma4StreamFilter`."""
    filt = _Gemma4StreamFilter(harmony=harmony)
    for chunk in iterator:
        piece = filt.feed(chunk)
        if piece:
            yield piece
    piece = filt.flush()
    if piece:
        yield piece


# Refuse to load when estimated (weights + KV cache) would exceed this
# fraction of available system RAM. On Apple Silicon (unified memory)
# and on discrete GPUs loading via ``n_gpu_layers=-1`` this is the
# same budget.
_MEMORY_SAFETY_FRACTION = 0.85

# Floor for the memory-aware auto-sizing in ``_fit_ctx_to_memory``. We
# never shrink an auto-selected context below this — a model that can't
# hold 2048 tokens is unusable, and the preflight check is the right
# place to reject that load with real numbers rather than silently
# serving a crippled context.
_MIN_AUTO_CTX = 2048


def _detect_chat_format_from_gguf(model_path: str) -> str | None:
    """Read ``general.architecture`` from a GGUF and map it to a
    llama-cpp-python ``chat_format``.

    Returns ``None`` when:

    - the optional ``gguf`` package isn't installed (HFL ships it in the
      ``[convert]`` extra, not in the core deps);
    - the file isn't readable;
    - the architecture isn't in our mapping table — in which case we
      defer to llama-cpp-python's own auto-detection.

    This helper exists because newer Gemma family GGUFs (released after
    Gemma 4) ship without an embedded ``tokenizer.chat_template``, so
    llama-cpp-python's fallback chooses the Llama-2 ``[INST]`` format,
    which silently destroys output quality. Detecting the architecture
    from the GGUF header lets us pick the correct format ahead of time.
    """
    try:
        import gguf
    except ImportError:
        logger.debug(
            "gguf package not installed; skipping chat-format auto-detection. "
            "Install hfl[convert] for full support."
        )
        return None

    try:
        reader = gguf.GGUFReader(model_path)
        arch_field = reader.fields.get("general.architecture")
        if arch_field is None:
            return None
        # The architecture value is stored as a UTF-8 string in the last
        # ``parts`` chunk of the field record.
        arch_bytes = bytes(arch_field.parts[-1])
        arch = arch_bytes.decode("utf-8", errors="replace").strip()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("could not read GGUF metadata for %s: %s", model_path, exc)
        return None

    fmt = _ARCHITECTURE_CHAT_FORMAT.get(arch)
    if fmt is not None:
        logger.info("Detected GGUF architecture %r → using chat_format=%r", arch, fmt)
    return fmt


def _read_gguf_model_info(model_path: str) -> dict | None:
    """Read layout metadata from a GGUF header for memory estimation.

    Returns a dict with keys ``architecture``, ``block_count``,
    ``embedding_length`` and ``max_context``. Any field that isn't
    present in the GGUF (or that fails to decode) is set to ``None``.

    Returns ``None`` if the optional ``gguf`` package isn't installed
    or the file isn't readable — in that case the caller should fall
    back to whatever safety nets don't require metadata (arch-based
    caps and user-supplied ``n_ctx`` still apply).
    """
    try:
        import gguf
    except ImportError:
        logger.debug(
            "gguf package not installed; skipping GGUF model info probe. "
            "Install hfl[convert] for full support."
        )
        return None

    try:
        reader = gguf.GGUFReader(model_path)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("could not open GGUF %s: %s", model_path, exc)
        return None

    def _read_str(field_name: str) -> str | None:
        field = reader.fields.get(field_name)
        if field is None:
            return None
        try:
            return bytes(field.parts[-1]).decode("utf-8", errors="replace").strip()
        except Exception:  # pragma: no cover — defensive
            return None

    def _read_int(field_name: str) -> int | None:
        field = reader.fields.get(field_name)
        if field is None:
            return None
        try:
            value = field.parts[-1]
            if isinstance(value, (bytes, bytearray, memoryview)):
                return int.from_bytes(bytes(value), "little", signed=False)
            # numpy array with a single scalar, or plain int
            import numpy as np

            arr = np.asarray(value)
            return int(arr.flat[0])
        except Exception:
            return None

    def _read_bool(field_name: str) -> bool | None:
        field = reader.fields.get(field_name)
        if field is None:
            return None
        try:
            value = field.parts[-1]
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bool(int.from_bytes(bytes(value), "little", signed=False))
            import numpy as np

            arr = np.asarray(value)
            return bool(int(arr.flat[0]))
        except Exception:
            return None

    arch = _read_str("general.architecture")
    if arch is None:
        return None

    return {
        "architecture": arch,
        "block_count": _read_int(f"{arch}.block_count"),
        "embedding_length": _read_int(f"{arch}.embedding_length"),
        "max_context": _read_int(f"{arch}.context_length"),
        "head_count": _read_int(f"{arch}.attention.head_count"),
        "head_count_kv": _read_int(f"{arch}.attention.head_count_kv"),
        # Presence of an embedded Jinja chat template. When True we
        # must NOT override ``chat_format`` with our static map —
        # the embedded template is always more accurate than any
        # preset llama-cpp-python ships, especially for new arches
        # like Gemma 4 whose prompt format differs from Gemma 2.
        "has_chat_template": "tokenizer.chat_template" in reader.fields,
        # Phase 11 P1 — V2 row 39. Gemma 4 models are shipped with
        # ``tokenizer.ggml.add_bos_token = false`` because their chat
        # template handles the BOS token explicitly. Engines that
        # also auto-prepend BOS double-insert it and mis-predict the
        # first few tokens. We surface this flag so the tokenize /
        # generate paths can respect it.
        "add_bos_token": _read_bool("tokenizer.ggml.add_bos_token"),
    }


def _kv_bytes_per_token(info: dict | None) -> int:
    """Bytes of fp16 KV cache consumed by a single context token.

    ``2 (K+V) * n_layers * n_kv_heads * head_dim * 2 bytes``. Falls back
    to the non-GQA upper bound (``embedding_length`` in place of
    ``n_kv_heads * head_dim``) when the GGUF header omits the attention
    head counts. Returns ``0`` when the header gives us nothing to work
    with — callers must treat that as "unknown", not as "free".
    """
    if not info:
        return 0

    block_count = info.get("block_count") or 0
    embedding_length = info.get("embedding_length") or 0
    head_count = info.get("head_count") or 0
    head_count_kv = info.get("head_count_kv") or 0

    if block_count and head_count and head_count_kv and embedding_length:
        head_dim = embedding_length // head_count
        kv_dim = head_count_kv * head_dim
        return 2 * block_count * kv_dim * 2
    if block_count and embedding_length:
        return 2 * block_count * embedding_length * 2
    return 0


def _fit_ctx_to_memory(model_path: str, info: dict | None, n_ctx: int) -> int:
    """Shrink an auto-selected ``n_ctx`` until weights + KV cache fit.

    The VRAM tier in :mod:`hfl.engine.vram` sizes the context from the
    machine's memory alone — it knows nothing about how much of that
    memory the *weights* already claim. On a 128 GB Apple Silicon host
    the top tier hands back 262144 tokens, which for a 72B model is an
    80 GiB KV cache on top of 44 GiB of weights: the load is refused by
    :func:`_preflight_memory_check` (or, without psutil, thrashes the
    host). Ollama sizes the context to what is left after the weights;
    this does the same.

    Returns the largest power-of-two context ``<= n_ctx`` whose weights
    + KV estimate stays inside ``_MEMORY_SAFETY_FRACTION`` of available
    memory, floored at ``_MIN_AUTO_CTX``. Returns ``n_ctx`` unchanged
    when we can't measure either side (no psutil, no GGUF header) —
    the preflight check remains the backstop.
    """
    from pathlib import Path

    if n_ctx <= _MIN_AUTO_CTX:
        return n_ctx

    kv_per_token = _kv_bytes_per_token(info)
    if kv_per_token <= 0:
        return n_ctx

    from hfl.engine.memory import HAS_PSUTIL, get_memory_snapshot

    if not HAS_PSUTIL:
        return n_ctx

    try:
        weights_bytes = Path(model_path).stat().st_size
    except OSError:
        return n_ctx

    available_bytes = get_memory_snapshot().system_available_gb * (1024**3)
    kv_budget = available_bytes * _MEMORY_SAFETY_FRACTION - weights_bytes
    if kv_budget <= 0:
        # The weights alone blow the budget — nothing we do to the
        # context saves this load. Leave it to the preflight check,
        # which reports the real numbers to the user.
        return _MIN_AUTO_CTX

    fitted = n_ctx
    while fitted > _MIN_AUTO_CTX and fitted * kv_per_token > kv_budget:
        fitted //= 2

    if fitted < n_ctx:
        logger.info(
            "Auto-sized n_ctx %d → %d to fit weights (%.1fGB) + KV cache "
            "in %.0f%% of %.1fGB available memory",
            n_ctx,
            fitted,
            weights_bytes / (1024**3),
            _MEMORY_SAFETY_FRACTION * 100,
            available_bytes / (1024**3),
        )
    return fitted


def _count(counted: CountedStream, model: Any, first: int | None, finish: str | None) -> None:
    """Token counts of a stream that ran to its end.

    llama-cpp-python's streams carry no ``usage`` and their chunks merge
    tokens (an unfinished UTF-8 character, a possible stop sequence), so the
    counts come from the context instead — measured against the
    non-streaming ``usage``, including a cached prompt prefix and emoji:
    when the first chunk arrives the prompt has been evaluated and nothing
    else (``n_tokens`` = prompt); at the end ``n_tokens`` = prompt +
    completion, less the last token when generation hit ``max_tokens``
    (sampled, never evaluated).
    """
    final = getattr(model, "n_tokens", None)
    if not isinstance(first, int) or not isinstance(final, int):
        return  # a backend (or a stub) that does not expose the context
    counted.prompt_tokens = first
    counted.completion_tokens = final - first + (1 if finish == "length" else 0)


def resolve_n_ctx(
    model_path: str,
    gguf_info: dict | None,
    n_ctx: int,
    explicit_n_ctx: bool,
) -> int:
    """The context window to open ``model_path`` with.

    ``n_ctx`` is the caller's value (explicit) or the configured default
    (``0`` = auto). An explicit value is returned untouched; otherwise the
    architecture cap, the VRAM tier, the model's advertised maximum and the
    memory left after the weights decide. Shared by every GGUF backend so
    they open the same model with the same window.
    """
    architecture = gguf_info.get("architecture") if gguf_info else None
    # Architecture-based safe cap on n_ctx. Gemma 3/4 GGUFs advertise
    # 131072-token contexts; the fp16 KV cache for that window can
    # trivially exceed available unified memory on macOS and crash the
    # host. Only apply when the caller did NOT pass an explicit n_ctx.
    if not explicit_n_ctx and architecture in _ARCHITECTURE_CTX_CAP:
        cap = _ARCHITECTURE_CTX_CAP[architecture]
        if n_ctx == 0 or n_ctx > cap:
            logger.warning(
                "Capping n_ctx to %d for architecture %r (was %s). "
                "Advertised context length would exceed the safe "
                "memory budget. Override with n_ctx=<N> or set "
                "HFL_DEFAULT_CTX_SIZE.",
                cap,
                architecture,
                n_ctx if n_ctx else "auto",
            )
            n_ctx = cap

    # Phase 11 P1 — V2 row 13. When neither the caller nor the
    # architecture pinned a value, fall back to a VRAM-tier
    # recommendation (4 k / 32 k / 256 k). No-op on Gemma 3/4
    # because the arch cap above already set n_ctx.
    if not explicit_n_ctx and n_ctx == 0:
        try:
            from hfl.engine.vram import pick_ctx_size

            tier = pick_ctx_size()
            n_ctx = tier.ctx
            if tier.vram_gib is not None:
                logger.info(
                    "VRAM probe saw %.1f GiB → num_ctx=%d",
                    tier.vram_gib,
                    n_ctx,
                )
            else:
                logger.info("VRAM probe inconclusive → defaulting num_ctx=%d", n_ctx)
        except Exception:
            logger.debug("VRAM auto-sizing failed", exc_info=True)

    # Two clamps that only apply to an auto-selected context — an
    # explicit ``n_ctx=`` is the caller's call and is left alone
    # (the preflight check below still guards the host).
    #
    #   1. Never exceed the context the model was actually trained
    #      for. The VRAM tier is derived from the machine, not the
    #      model, so on a large host it happily returns 262144 for
    #      a model whose GGUF advertises 32768.
    #   2. Never size the KV cache past what is left after the
    #      weights. Without this a 72B Q4_K_M on a 128 GB Mac
    #      auto-selects 262144 tokens = 80 GiB of KV on top of
    #      44 GiB of weights, and every load 500s with
    #      "Insufficient memory ... requires ~124.2GB".
    if not explicit_n_ctx and n_ctx > 0:
        advertised = (gguf_info or {}).get("max_context") or 0
        if advertised and n_ctx > advertised:
            logger.info(
                "Clamping auto n_ctx %d → %d (model's advertised context length)",
                n_ctx,
                advertised,
            )
            n_ctx = advertised
        n_ctx = _fit_ctx_to_memory(model_path, gguf_info, n_ctx)

    return n_ctx


def _estimate_memory_required_gb(model_path: str, info: dict | None, n_ctx: int) -> float:
    """Conservative upper bound for the RAM / unified memory a load will take.

    Adds two components:

    - **Weights**: the GGUF file size on disk. For mmap'd GGUFs the
      steady-state footprint can be lower than this (the kernel can
      page out unused pages) but under load the whole file tends to
      fault in, so using the file size is a safe upper bound.
    - **KV cache**: ``2 (K+V) * n_layers * n_ctx * n_kv_heads *
      head_dim * 2 bytes (fp16)``. When ``head_count`` /
      ``head_count_kv`` aren't in the GGUF header we fall back to the
      non-GQA upper bound ``embedding_length`` (= ``n_heads *
      head_dim``), which over-estimates heavy-GQA models like Gemma 4
      (32/4 ratio → 8× over-estimate). The GQA-aware branch makes the
      preflight precise enough that we don't reject models that would
      actually fit.

    Returns ``0.0`` when neither component can be measured.
    """
    from pathlib import Path

    try:
        weights_bytes = Path(model_path).stat().st_size
    except OSError:
        weights_bytes = 0
    weights_gb = weights_bytes / (1024**3)

    kv_gb = 0.0
    if info is not None and n_ctx > 0:
        kv_gb = (_kv_bytes_per_token(info) * n_ctx) / (1024**3)

    return weights_gb + kv_gb


def _preflight_memory_check(
    model_path: str,
    info: dict | None,
    n_ctx: int,
    architecture: str | None,
) -> None:
    """Refuse to load when we can already tell the model won't fit.

    Raises :class:`hfl.exceptions.OutOfMemoryError` with concrete
    required/available numbers when the estimate exceeds
    ``_MEMORY_SAFETY_FRACTION`` of available system memory.

    Returns silently (after logging a warning) when ``psutil`` isn't
    installed — we simply can't make the call, so we fall back to
    trusting the arch-specific caps that were applied upstream.

    Can be disabled with ``HFL_DISABLE_MEMORY_PREFLIGHT=1`` for
    advanced users who accept the risk (e.g. loading onto a discrete
    GPU whose VRAM we don't measure).
    """
    if os.environ.get("HFL_DISABLE_MEMORY_PREFLIGHT", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.debug("Memory preflight disabled via HFL_DISABLE_MEMORY_PREFLIGHT")
        return

    from hfl.engine.memory import HAS_PSUTIL, get_memory_snapshot
    from hfl.exceptions import OutOfMemoryError

    if not HAS_PSUTIL:
        logger.warning(
            "psutil not installed; skipping memory preflight. Install it "
            "('pip install psutil') to catch oversized model loads before "
            "they crash the host."
        )
        return

    required_gb = _estimate_memory_required_gb(model_path, info, n_ctx)
    if required_gb <= 0:
        # Nothing useful to compare against — either the file is
        # missing (load() will raise FileNotFoundError shortly anyway)
        # or we couldn't read any metadata. Don't block the load.
        return

    snapshot = get_memory_snapshot()
    available_gb = snapshot.system_available_gb
    budget_gb = available_gb * _MEMORY_SAFETY_FRACTION

    logger.info(
        "Memory preflight: required≈%.1fGB, available=%.1fGB, budget=%.1fGB (%.0f%%)",
        required_gb,
        available_gb,
        budget_gb,
        _MEMORY_SAFETY_FRACTION * 100,
    )

    if required_gb > budget_gb:
        # Split the footprint so the error can tell the user whether the
        # load is impossible on this hardware (weights don't fit) or just
        # over-contexted (weights fit, KV cache doesn't) — and, in the
        # latter case, the largest context that would have worked.
        weights_gb = _estimate_memory_required_gb(model_path, info, 0)
        fitting_ctx: int | None = None
        if n_ctx > 0 and weights_gb < budget_gb:
            candidate = _fit_ctx_to_memory(model_path, info, n_ctx)
            # Only advertise a context that genuinely fits — at the floor
            # ``_fit_ctx_to_memory`` gives up rather than going lower.
            if candidate < n_ctx and (
                _estimate_memory_required_gb(model_path, info, candidate) <= budget_gb
            ):
                fitting_ctx = candidate
        err = OutOfMemoryError(
            required_gb=required_gb,
            available_gb=available_gb,
            weights_gb=weights_gb,
            n_ctx=n_ctx or None,
            fitting_ctx=fitting_ctx,
        )
        if architecture and architecture.startswith("gemma"):
            err.details = (
                f"{err.details}\n\n"
                f"The {architecture} family advertises very large context "
                f"windows (often 131072 tokens); the KV cache for that "
                f"context dominates memory usage. Retry with --ctx 4096 "
                f"(or a smaller quantisation like Q3_K_M), or set "
                f"HFL_DEFAULT_CTX_SIZE=4096. Set "
                f"HFL_DISABLE_MEMORY_PREFLIGHT=1 only if you know the "
                f"model will fit via paths we can't measure (e.g. a "
                f"discrete GPU)."
            )
        raise err


def _build_vision_chat_handler(
    *,
    architecture: str | None,
    clip_model_path: str,
    verbose: bool = False,
) -> object | None:
    """Instantiate the right multimodal chat handler for this arch.

    Phase 4 P0-6. llama-cpp-python ships one ``chat_handler``
    subclass per vision family; the arch name we detected from the
    GGUF header picks which one. Returns ``None`` (with a warning
    log) when the local llama-cpp-python install is too old to
    expose the handlers, so load falls back to text-only mode
    instead of crashing.

    Arguments:
        architecture: ``general.architecture`` value from the GGUF
            header (e.g. ``gemma3``, ``llama4``, ``qwen2vl``,
            ``llava``).
        clip_model_path: Absolute path to the CLIP projector GGUF
            (typically ``mmproj-*.gguf``).
        verbose: Passed through to the handler for logging.
    """
    arch = (architecture or "").lower()

    try:
        from llama_cpp import llama_chat_format as _lcf
        from llama_cpp.llama_chat_format import (
            Llava15ChatHandler,
            Llava16ChatHandler,
            MoondreamChatHandler,
            Qwen25VLChatHandler,
        )
    except ImportError:
        logger.warning(
            "llama-cpp-python installed here lacks multimodal chat handlers; "
            "loading %s without vision support. Upgrade with: "
            "pip install -U 'llama-cpp-python>=0.3.20'",
            arch or "model",
        )
        return None

    # The Gemma vision handler was renamed ``Gemma3ChatHandler`` ->
    # ``Gemma4ChatHandler`` across llama-cpp-python releases. Resolve it
    # dynamically so a version bump neither crashes the import (which
    # would take *all* vision handlers down with it) nor pins us to one
    # spelling. ``None`` when this build ships neither.
    gemma_handler = getattr(_lcf, "Gemma4ChatHandler", None) or getattr(
        _lcf, "Gemma3ChatHandler", None
    )

    # Arch → handler. Order matters: most-specific substring first
    # so e.g. ``llava-v1.6`` doesn't match the v1.5 handler.
    if "gemma" in arch and ("3" in arch or "4" in arch):
        if gemma_handler is None:
            logger.warning(
                "this llama-cpp-python build has no Gemma vision handler; "
                "loading %s without vision support.",
                arch or "model",
            )
            return None
        handler_cls = gemma_handler
    elif "qwen" in arch and "vl" in arch:
        handler_cls = Qwen25VLChatHandler
    elif "moondream" in arch:
        handler_cls = MoondreamChatHandler
    elif "llava-v1.6" in arch or "llava_1.6" in arch or "llava16" in arch:
        handler_cls = Llava16ChatHandler
    elif "llava" in arch:
        handler_cls = Llava15ChatHandler
    else:
        # Unknown architecture but we have a CLIP projector — try
        # the most common (LLaVA 1.5) and let the model complain at
        # first inference if it mismatches.
        logger.warning(
            "Unknown vision architecture %r; falling back to Llava15 handler",
            architecture,
        )
        handler_cls = Llava15ChatHandler

    return cast("object | None", handler_cls(clip_model_path=clip_model_path, verbose=verbose))


# V4 F5 — adapter that lets a second ``Llama`` instance act as a
# draft model for speculative decoding. ``llama_cpp.Llama`` itself
# does NOT implement the ``LlamaDraftModel`` protocol — calling it
# directly returns a completion dict, not a token-ids ndarray.
# Subclassing ``LlamaDraftModel`` and forwarding to the inner Llama
# closes that gap.
class _LlamaModelDraftAdapter:
    """Bridge a small ``Llama`` instance into the ``LlamaDraftModel``
    protocol expected by llama-cpp-python's ``Llama(draft_model=...)``.

    On every step the target asks the adapter to predict ``num_pred``
    tokens past the current ``input_ids``; we feed *only the new
    tokens* to the draft Llama (preserving its KV cache across calls)
    and return a numpy int array of greedy candidates.

    Why incremental: a naive implementation that ``reset()``s the
    draft on every call pays the full prefill cost N times across a
    single response, which inverts the intended speedup (speculative
    decoding becomes 2-3× *slower* than plain). Reusing the cache
    across calls keeps the draft cost O(1) per step.

    The adapter detects when the target's ``input_ids`` is a prefix
    extension of the previous call (the common case during a single
    response) and only evaluates the suffix; if the sequence
    diverges (a fresh request, a cancelled completion) it falls back
    to a full reset + prefill.
    """

    def __init__(self, draft: "Llama", num_pred_tokens: int = 10) -> None:
        self._draft = draft
        self._num_pred = max(1, int(num_pred_tokens))
        # Tokens already evaluated by the draft's KV cache (target
        # prompt + accepted target tokens + draft predictions whose
        # eval was performed but rejected by the target). We only
        # add to this; ``_aligned`` controls how much overlaps with
        # the next call's ``input_ids``.
        self._processed: list[int] = []

    def _align_and_eval_suffix(self, input_ids_list: list[int]) -> bool:
        """Make the draft's KV cache align with ``input_ids_list``.

        Common path: ``input_ids_list`` starts with ``self._processed``
        (the target accepted some draft tokens and is now asking for
        the next round) — we evaluate only the tail.

        Divergent path: a fresh request, or the target rejected our
        last predictions — reset the draft and replay the whole
        sequence.

        Returns ``True`` on success, ``False`` if any eval raised
        (the caller short-circuits to "no candidates").
        """
        # Find the longest common prefix with the previous state.
        common = 0
        for a, b in zip(self._processed, input_ids_list):
            if a == b:
                common += 1
            else:
                break

        try:
            if common < len(self._processed):
                # Divergence — reset and replay from scratch.
                self._draft.reset()
                self._draft.eval(input_ids_list)
                self._processed = list(input_ids_list)
                return True
            suffix = input_ids_list[common:]
            if suffix:
                self._draft.eval(suffix)
                self._processed.extend(suffix)
            return True
        except Exception:  # pragma: no cover — ABI defence
            self._processed = []
            return False

    def __call__(self, input_ids, /, **kwargs):
        import numpy as np

        if input_ids is None or len(input_ids) == 0:
            return np.zeros((0,), dtype=np.intc)

        ids_list = [int(t) for t in input_ids]
        if not self._align_and_eval_suffix(ids_list):
            return np.zeros((0,), dtype=np.intc)

        # Greedy-sample ``num_pred`` tokens. We feed each prediction
        # back into the draft so the next sample sees the updated
        # KV state. The predictions ARE recorded into ``_processed``
        # so the next call can correctly compute the divergence
        # point (the target may accept all, some, or none).
        out: list[int] = []
        try:
            sample_fn = getattr(self._draft, "sample", None)
            if sample_fn is None:
                return np.zeros((0,), dtype=np.intc)
            for _ in range(self._num_pred):
                tok_id = int(sample_fn(temp=0.0))
                out.append(tok_id)
                self._draft.eval([tok_id])
                self._processed.append(tok_id)
        except Exception:
            # Fall back to whatever we managed to predict; never
            # propagate so the target keeps running.
            return np.array(out, dtype=np.intc)

        return np.array(out, dtype=np.intc)


class LlamaCppEngine(InferenceEngine):
    """llama.cpp inference engine."""

    # One request at a time here; the same GGUF can serve several at once
    # through llama-server. The dispatcher says so when a request waits.
    parallel_hint = True

    def __init__(self):
        # ``Llama`` is an untyped optional-dependency handle (resolved
        # to ``Any`` when llama-cpp-python isn't installed). Annotating
        # the attribute as ``Any`` lets mypy type-check the call sites
        # without spurious ``union-attr`` / ``None not callable`` errors
        # while preserving the runtime ``None`` sentinel for "unloaded".
        self._model: Any = None
        self._model_path: str = ""
        self._architecture: str | None = None
        # Effective context the model was opened with, after the
        # explicit/arch/VRAM/memory resolution in ``load``. Surfaced via
        # ``context_size`` so the API layer can reload when a request
        # asks for a different ``num_ctx``.
        self._n_ctx: int = 0
        # One-line summary of what the load did with the hardware, mined
        # from llama.cpp's loader output. ``None`` = unknown / CPU-only.
        self._acceleration: str | None = None
        # V4 F5 — companion draft model for speculative decoding.
        # Held here so ``unload()`` can free its memory alongside
        # the target.
        self._draft_model: Any = None
        # True when the model was loaded with a CLIP projector and
        # accepts images in ``create_chat_completion`` messages.
        # Phase 4 P0-6.
        self._is_multimodal: bool = False
        # Whether the prompt lists the request's tools by itself; when not,
        # ``_tools_as_text`` writes them in. Set at load.
        self._template_knows_tools: bool = True
        # The GGUF chat template in use ("" for a static format or a vision
        # handler): how it renders past tool calls decides how HFL passes
        # them (``_tool_messages``).
        self._chat_template: str = ""
        # HFL's formatters for the GGUF's templates (``_install_template_
        # formatters``): the per-request template variables are set on them.
        self._formatters: list[Any] = []
        # Whether the template takes a system message (Mistral's does not:
        # HFL folds system text into the first user turn).
        self._template_takes_system: bool = True
        # Held by whichever thread is inside llama.cpp for this model (see
        # ``_holding``). A plain Lock, not an RLock: a stream may be closed
        # on a different thread than the one that started it.
        self._native = threading.Lock()
        # LoRA adapters on the context, in the order applied: (adapter id,
        # path, scale, llama.cpp handle). See ``apply_lora``.
        self._loras: list[tuple[str, str, float, Any]] = []

    def load(self, model_path: str, **kwargs) -> None:
        """
        Loads a GGUF model.

        Args:
            model_path: Path to the .gguf file
            **kwargs: Additional parameters:
                n_ctx: Context size (default 4096)
                n_gpu_layers: GPU layers (-1 = all)
                n_threads: CPU threads (0 = auto)
                verbose: Show llama.cpp logs
                flash_attn: Use Flash Attention (default True)
                chat_format: Chat format (auto-detected)

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the path is invalid or not a GGUF file.
        """
        from pathlib import Path

        # Validate model path for security and correctness
        path = Path(model_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not path.is_file():
            raise ValueError(f"Model path is not a file: {model_path}")

        if not path.suffix.lower() == ".gguf":
            raise ValueError(f"Model file must be a .gguf file, got: {path.suffix}")

        # Use resolved path to prevent path traversal issues
        model_path = str(path)

        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with: pip install 'hfl[llama]'"
            )

        from hfl.config import config as hfl_config

        verbose = kwargs.get("verbose", False)
        user_n_ctx = kwargs.get("n_ctx")
        explicit_n_ctx = user_n_ctx is not None and user_n_ctx > 0
        # ``explicit_n_ctx`` guarantees ``user_n_ctx`` is a positive int
        # at runtime, but mypy can't carry that narrowing through the
        # separate boolean — cast to the runtime-correct ``int``.
        n_ctx: int = cast(int, user_n_ctx if explicit_n_ctx else hfl_config.default_ctx_size)
        # Phase 11 P1 — V2 row 13 VRAM auto-sizing moved *after* the
        # architecture cap below so that arch-specific safe defaults
        # (Gemma's 8192 cap) always take precedence. We only reach
        # the VRAM path when neither the caller nor the architecture
        # pinned a value.
        n_gpu_layers = kwargs.get("n_gpu_layers", hfl_config.default_n_gpu_layers)

        # Read the GGUF header once and use it for BOTH chat-format
        # detection AND the memory-safety gates below. This replaces the
        # old ``_detect_chat_format_from_gguf`` call site — that helper
        # is kept around for its own unit tests, but the load path now
        # reads the header through a single shared probe.
        gguf_info = _read_gguf_model_info(model_path)
        architecture = gguf_info.get("architecture") if gguf_info else None

        # Chat format selection: the decision tree is
        #
        #   1. Explicit ``chat_format=`` from the caller — always wins.
        #   2. GGUF ships an embedded ``tokenizer.chat_template`` — leave
        #      ``chat_format=None`` so llama-cpp-python uses the embedded
        #      Jinja template. This is the correct path for any well-
        #      packaged GGUF (bartowski, unsloth, lmstudio-community,
        #      official Google exports, …).
        #   3. Static override from ``_ARCHITECTURE_CHAT_FORMAT`` — only
        #      as a last resort, for community GGUFs that forgot to
        #      embed the template (which otherwise downgrade to
        #      llama-cpp-python's Llama-2 ``[INST]`` fallback and
        #      silently ruin chat quality).
        #
        # Overriding ``chat_format`` when the GGUF already has a Jinja
        # template is worse than leaving it alone: the static presets
        # llama-cpp-python ships are frozen at Gemma 2 and don't know
        # about Gemma 4's ``<|turn>`` / ``<|channel>`` delimiter
        # scheme, so forcing them breaks the prompt side of the chat.
        chat_format = kwargs.get("chat_format")
        has_embedded_template = bool(gguf_info and gguf_info.get("has_chat_template"))
        if chat_format is None and architecture is not None and not has_embedded_template:
            chat_format = _ARCHITECTURE_CHAT_FORMAT.get(architecture)
            if chat_format is not None:
                logger.info(
                    "Detected GGUF architecture %r with no embedded "
                    "chat_template → using static chat_format=%r",
                    architecture,
                    chat_format,
                )
        elif has_embedded_template:
            logger.debug(
                "GGUF for architecture %r ships an embedded "
                "tokenizer.chat_template; deferring to it instead of "
                "the static _ARCHITECTURE_CHAT_FORMAT override.",
                architecture,
            )

        n_ctx = resolve_n_ctx(model_path, gguf_info, n_ctx, explicit_n_ctx)

        # Flash-attention is not safe for every architecture: llama-cpp-
        # python's flash-attn path has been historically crash-prone for
        # new arches. Force-disable for known-bad arches unless the
        # caller explicitly opted in.
        #
        # Resolution order:
        #   1. Per-load ``flash_attn=`` kwarg (e.g. from a Modelfile).
        #   2. Global server toggle ``HFL_FLASH_ATTENTION`` /
        #      ``OLLAMA_FLASH_ATTENTION`` — accepts ``"1"``/``"0"``,
        #      ``"true"``/``"false"`` (case-insensitive). When set to a
        #      falsy value, flash-attn is forced off across the board.
        #      When truthy, the per-architecture safety list still
        #      applies — operators turning the global on don't get
        #      crashes on known-bad arches for free.
        #   3. Architecture-aware default (False for known-unsafe arches,
        #      True otherwise).
        global_flash_env = os.environ.get("HFL_FLASH_ATTENTION") or os.environ.get(
            "OLLAMA_FLASH_ATTENTION"
        )
        global_flash: bool | None
        if global_flash_env is None or global_flash_env == "":
            global_flash = None
        else:
            global_flash = global_flash_env.strip().lower() in ("1", "true", "yes", "on")

        if "flash_attn" in kwargs:
            flash_attn = kwargs["flash_attn"]
        elif global_flash is False:
            flash_attn = False
        elif architecture in _ARCHITECTURE_NO_FLASH_ATTN:
            logger.info(
                "Disabling flash_attn for architecture %r "
                "(known unsafe in current llama-cpp-python). "
                "Pass flash_attn=True to override.",
                architecture,
            )
            flash_attn = False
        else:
            flash_attn = True

        # Preflight memory check: refuse to load when we can already
        # tell the model + KV cache won't fit. When n_ctx is still 0
        # at this point we're letting llama-cpp auto-detect from the
        # GGUF metadata max — use that same value for the estimate so
        # we catch oversized auto-detected contexts too.
        preflight_ctx = n_ctx
        if preflight_ctx <= 0 and gguf_info is not None:
            preflight_ctx = gguf_info.get("max_context") or 0
        _preflight_memory_check(
            model_path=model_path,
            info=gguf_info,
            n_ctx=preflight_ctx,
            architecture=architecture,
        )

        logger.info("Loading GGUF model: %s", path.name)
        logger.debug(
            "Model path: %s, n_ctx=%s, n_gpu_layers=%s, chat_format=%s, "
            "flash_attn=%s, architecture=%s",
            model_path,
            n_ctx,
            n_gpu_layers,
            chat_format,
            flash_attn,
            architecture,
        )

        # Phase 4 P0-6: vision / multimodal. Vision-capable GGUF
        # models ship a paired CLIP/vision projector file (usually
        # ``mmproj-*.gguf``). When the caller passes one — either as
        # an explicit ``clip_model_path`` kwarg or as a path
        # adjacent to ``model_path`` — build the matching multimodal
        # chat handler so ``create_chat_completion`` accepts
        # ``images`` in its messages.
        clip_model_path: str | None = kwargs.get("clip_model_path")
        if clip_model_path is None:
            from hfl.engine.projector import find_projector

            candidate = find_projector(path)
            if candidate is not None:
                clip_model_path = str(candidate)
                logger.info("Auto-detected CLIP projector: %s", candidate.name)

        chat_handler = None
        if clip_model_path:
            chat_handler = _build_vision_chat_handler(
                architecture=architecture,
                clip_model_path=clip_model_path,
                verbose=verbose,
            )

        start_time = time.perf_counter()
        try:
            # Suppress Metal/CUDA initialization messages if verbose=False
            # Capture llama.cpp's loader dump instead of discarding it, so the
            # acceleration summary can be logged at INFO. With verbose=True the
            # user already sees everything on the terminal, so we don't capture.
            captured: list[str] = []
            context = _suppress_stderr if not verbose else _nullcontext
            with context(), _capture_llama_log(captured):
                llama_kwargs: dict = {
                    "model_path": model_path,
                    "n_ctx": n_ctx,
                    "n_gpu_layers": n_gpu_layers,
                    "n_threads": kwargs.get("n_threads", hfl_config.default_threads) or None,
                    "verbose": verbose,
                    "flash_attn": flash_attn,
                    "chat_format": chat_format,
                }
                # Phase 11 P1: KV cache quantisation. Maps
                # ``"q4_0"`` / ``"q8_0"`` strings to llama-cpp's
                # ``type_k`` / ``type_v`` integer enum. ``"f16"`` is
                # the default and leaves the fields unset so the
                # library picks its own default.
                kv_type = kwargs.get("kv_cache_type") or hfl_config.kv_cache_type
                if kv_type and kv_type != "f16":
                    # Import under a second name and assign: binding the
                    # *annotated* name directly from an ``import`` is a
                    # redefinition for mypy when llama_cpp is absent (the CI
                    # venv omits the [llama] extra), while the annotation is
                    # what lets the ``except`` branch assign ``None`` when it
                    # is present. Splitting the two satisfies both.
                    _lcpp: Any
                    try:
                        from llama_cpp import llama_cpp as _lcpp_module

                        _lcpp = _lcpp_module
                    except Exception:
                        _lcpp = None
                    type_map = {}
                    if _lcpp is not None:
                        type_map = {
                            "q4_0": getattr(_lcpp, "GGML_TYPE_Q4_0", None),
                            "q8_0": getattr(_lcpp, "GGML_TYPE_Q8_0", None),
                            "f32": getattr(_lcpp, "GGML_TYPE_F32", None),
                            "f16": getattr(_lcpp, "GGML_TYPE_F16", None),
                        }
                    code = type_map.get(kv_type.lower())
                    if code is not None:
                        llama_kwargs["type_k"] = code
                        llama_kwargs["type_v"] = code
                        logger.info("KV cache quantised to %s", kv_type)
                    else:
                        logger.warning(
                            "kv_cache_type=%r unsupported by this llama-cpp build, "
                            "falling back to f16",
                            kv_type,
                        )
                if chat_handler is not None:
                    # When a multimodal chat_handler is supplied,
                    # ``chat_format`` must be None so llama-cpp-python
                    # doesn't try to install a conflicting text-only
                    # template.
                    llama_kwargs["chat_handler"] = chat_handler
                    llama_kwargs.pop("chat_format", None)
                # Phase 8 P3-2: LoRA adapters (a Modelfile's ADAPTER lines)
                # are applied after the load, all of them, through the same
                # list as hot-applied ones (``apply_lora``): llama.cpp sets
                # a context's adapters as one set, so an adapter handed to
                # ``Llama(lora_path=...)`` would be dropped by the first
                # hot-apply.
                lora_paths = list(kwargs.get("lora_paths") or [])
                # V4 F5 — speculative decoding.
                #
                # Two modes are supported through the same kwarg:
                #
                #   draft_model_path = "prompt-lookup"
                #       Use llama-cpp-python's
                #       ``LlamaPromptLookupDecoding``. Zero VRAM cost,
                #       reliable 1.3-2× speedup on prompts with
                #       repetitive patterns (RAG, code, structured
                #       output). The default for ``HFL_DRAFT_DEFAULT=
                #       lookup``.
                #
                #   draft_model_path = "<path/to/draft.gguf>"
                #       Load a second small ``Llama`` and route it
                #       through :class:`_LlamaModelDraftAdapter`. Only
                #       safe with a draft from the SAME tokenizer
                #       family as the target (Qwen3-14B ↔ Qwen3-0.6B,
                #       Llama-3.1-70B ↔ Llama-3.2-1B). Acceptance
                #       rates and net speedup are workload-dependent —
                #       benchmark before relying on it. The
                #       per-callback overhead of llama-cpp-python's
                #       Python ``draft_model`` API can in some cases
                #       cancel out the savings; use prompt-lookup as
                #       a known-good baseline.
                draft_spec = kwargs.get("draft_model_path") or None
                draft_llama: Llama | None = None
                # Either a prompt-lookup decoder or a draft-model adapter,
                # both duck-typed as llama-cpp's ``draft_model``.
                draft_callable: Any = None
                if draft_spec == "prompt-lookup":
                    try:
                        from llama_cpp.llama_speculative import (
                            LlamaPromptLookupDecoding,
                        )

                        draft_callable = LlamaPromptLookupDecoding(
                            num_pred_tokens=10, max_ngram_size=2
                        )
                        logger.info("Speculative decoding: prompt-lookup mode")
                    except Exception as exc:
                        logger.warning(
                            "prompt-lookup decoding unavailable (%s); "
                            "continuing without speculation",
                            exc,
                        )
                elif draft_spec:
                    logger.info("Loading speculative-decoding draft: %s", draft_spec)
                    try:
                        draft_llama = Llama(
                            model_path=str(draft_spec),
                            n_ctx=n_ctx,
                            n_gpu_layers=n_gpu_layers,
                            verbose=verbose,
                        )
                        draft_callable = _LlamaModelDraftAdapter(draft_llama)
                    except Exception as exc:
                        logger.warning(
                            "draft model load failed (%s); continuing without speculative decoding",
                            exc,
                        )
                        draft_llama = None
                if draft_callable is not None:
                    llama_kwargs["draft_model"] = draft_callable
                self._model = Llama(**llama_kwargs)
                # Track the draft so ``unload`` releases its memory too.
                self._draft_model = draft_llama
            self._model_path = model_path
            self._architecture = architecture
            # ``n_ctx`` may still be 0 here when the caller left it to
            # llama-cpp-python's own metadata default — read back what
            # the library actually opened so ``context_size`` never
            # reports a value the model isn't running with.
            try:
                self._n_ctx = int(self._model.n_ctx())
            except Exception:  # pragma: no cover — defensive
                self._n_ctx = n_ctx
            # Phase 11 P1 — V2 row 39. Remember the tokenizer's BOS
            # preference so downstream ``tokenize()`` calls don't
            # double-prepend BOS on Gemma 4 and friends. Default True
            # mirrors llama-cpp-python's old behaviour.
            self._tokenizer_add_bos: bool = bool((gguf_info or {}).get("add_bos_token", True))
            self._is_multimodal = chat_handler is not None
            self._loras = []
            for index, lora in enumerate(lora_paths):
                logger.info("Loading LoRA adapter: %s", lora)
                self.apply_lora(lora, 1.0, adapter_id=f"modelfile-{index}")
            self._formatters = []
            if chat_handler is None:
                self._formatters, added_bos = _install_template_formatters(self._model)
                if added_bos:
                    logger.info("Chat template does not start with BOS; HFL adds it")
                if (
                    chat_format is None
                    and self._formatters
                    and "chat_template.default" in self._model._chat_handlers
                ):
                    # The GGUF's own template, through HFL's formatter — not
                    # the built-in format llama-cpp-python swaps in when it
                    # recognises the template (Mistral's): that one dropped
                    # system messages, the tools HFL wrote there with them,
                    # and bypassed the BOS fix and the reasoning switch.
                    self._model.chat_format = "chat_template.default"
            # A vision chat handler has its own fixed format, never tools.
            template = (getattr(self._model, "metadata", None) or {}).get(
                "tokenizer.chat_template", ""
            )
            if not isinstance(template, str) or chat_handler is not None or chat_format:
                template = ""  # a static format or a vision handler is used instead
            self._chat_template = template
            self._template_knows_tools = chat_handler is None and _template_renders_tools(
                template, chat_format
            )
            self._template_takes_system = _template_takes_system(template)
            elapsed = time.perf_counter() - start_time
            mm_note = " (multimodal)" if self._is_multimodal else ""
            logger.info("Model loaded in %.2fs%s: %s", elapsed, mm_note, path.name)

            # Report what the load actually did with the hardware. Without
            # this the only way to know whether Metal/CUDA picked up the
            # weights was to re-run with verbose=True and read llama.cpp's
            # raw dump — so a fully accelerated load was indistinguishable
            # from a CPU-only one.
            self._acceleration = _summarize_acceleration("".join(captured))
            if self._acceleration:
                logger.info("Acceleration: %s", self._acceleration)
            elif not verbose:
                logger.info(
                    "Acceleration: no GPU offload reported by llama.cpp "
                    "(running on CPU). Check 'hfl doctor'."
                )
            _warn_if_on_battery()
        except Exception as e:
            logger.error("Failed to load model %s: %s", path.name, e)
            raise

    def _holding(self, chunks: Iterator[str]) -> Iterator[str]:
        """``chunks`` read with the model held (``hfl.engine.base.held``)."""
        return held(self._native, chunks)

    # ------------------------------------------------------------------ LoRA

    def _set_loras(self) -> None:
        """Put the adapter list on the context, as one set, and forget the
        cached prompt: its KV was computed with the previous weights."""
        import ctypes

        from llama_cpp import llama_cpp as _lcpp

        count = len(self._loras)
        handles = (_lcpp.llama_adapter_lora_p_ctypes * count)(*[h for *_, h in self._loras])
        scales = (ctypes.c_float * count)(*[scale for _, _, scale, _ in self._loras])
        if _lcpp.llama_set_adapters_lora(self._model.ctx, handles, count, scales) != 0:
            raise RuntimeError("llama.cpp refused the LoRA adapter set")
        self._model.reset()

    def apply_lora(self, path: str, scale: float, adapter_id: str | None = None) -> None:
        with self._native:
            self._apply_lora(path, scale, adapter_id)

    def _apply_lora(self, path: str, scale: float, adapter_id: str | None = None) -> None:
        """Apply a LoRA adapter to the loaded model, on top of any already
        applied (``POST /api/lora/apply``, and a Modelfile's ADAPTER lines at
        load). llama-cpp-python has no API for it; llama.cpp's own has."""
        if self._model is None:
            raise RuntimeError("no model loaded")
        from llama_cpp import llama_cpp as _lcpp

        handle = _lcpp.llama_adapter_lora_init(self._model.model, str(path).encode())
        if not handle:
            raise ValueError(
                f"{os.path.basename(path)} is not a LoRA adapter llama.cpp can load for this model"
            )
        self._loras.append((adapter_id or str(path), str(path), float(scale), handle))
        try:
            self._set_loras()
        except Exception:
            self._loras.pop()
            _lcpp.llama_adapter_lora_free(handle)
            raise

    def remove_lora(self, adapter_id: str) -> None:
        with self._native:
            self._remove_lora(adapter_id)

    def count_prompt_tokens(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> int:
        """The prompt ``chat`` sends, counted as llama-cpp-python's chat
        handler does: the template's formatter renders it, and it is
        tokenized with special tokens, BOS unless the template wrote it."""
        with self._native:
            if self._model is None:
                raise RuntimeError("no model loaded")
            tokens, _, _ = self._template_prompt(messages, config or GenerationConfig(), tools)
            return len(tokens)

    def _template_prompt(
        self, messages: list[ChatMessage], cfg: GenerationConfig, tools: list[dict] | None
    ) -> tuple[list[int], list[dict] | None, bool]:
        """The prompt tokens ``chat`` feeds the model, rendered by the GGUF
        template's formatter as llama-cpp-python's handler renders it, and
        the tools and markers ``_tool_messages`` settled on."""
        named = {getattr(f, "hfl_name", ""): f for f in self._formatters}
        default = named.get("chat_template.default")
        if default is None:
            raise NotImplementedError("this model has no chat template in its GGUF")
        msgs, tools, markers = self._tool_messages(messages, tools)
        default.template_vars = reasoning_template_vars(cfg.reasoning)
        rendered = default(messages=msgs, tools=tools) if tools else default(messages=msgs)
        tokens = self._model.tokenize(
            rendered.prompt.encode("utf-8"),
            add_bos=not rendered.added_special,
            special=True,
        )
        return list(tokens), tools, markers

    def _sample_with_logprobs(
        self, tokens: list[int], cfg: GenerationConfig, penalty: float, special: bool
    ) -> tuple[str, list[dict], int, str]:
        """Generate from ``tokens`` with each token's log-probability and its
        ``cfg.logprobs`` best alternatives: ``(text, entries, n, finish)``.

        llama-cpp-python gives logprobs only to a model opened with
        ``logits_all`` — the logits of every position of the context, about
        5 GB for an 8k context and a 152k vocabulary — and without it
        ``/api/generate`` with ``logprobs`` answered 500. Here a logits
        processor keeps one row, the distribution the next token is drawn
        from (before sampling reshapes it), and the generator says which
        token was drawn.
        """
        import numpy as np
        from llama_cpp import LogitsProcessorList
        from llama_cpp import llama_cpp as _lcpp

        model = self._model
        n_vocab = model.n_vocab()
        seen: dict[str, Any] = {}

        def capture(input_ids: Any, logits: Any) -> Any:
            if len(logits) != n_vocab:  # not the full, id-ordered row
                raise NotImplementedError("logprobs: unexpected logits layout")
            seen["row"] = np.array(logits, dtype=np.float64)
            return logits

        if cfg.seed >= 0:
            model.set_seed(cfg.seed)
        top = max(0, min(20, int(cfg.logprobs or 0)))
        stops = [stop for stop in (cfg.stop or []) if stop]
        vocab = model._model.vocab
        out: list[int] = []
        entries: list[dict] = []
        text = b""
        finish = "length"

        def piece(token: int) -> tuple[bytes, dict]:
            raw = model.detokenize([token], prev_tokens=out, special=special)
            return raw, {"token": raw.decode("utf-8", errors="replace"), "bytes": list(raw)}

        steps = model.generate(
            tokens,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            temp=cfg.temperature,
            repeat_penalty=penalty,
            logits_processor=LogitsProcessorList([capture]),
        )
        try:
            for token in steps:
                row = seen.pop("row")
                if _lcpp.llama_vocab_is_eog(vocab, token):
                    finish = "stop"
                    break
                peak = row.max()
                logprobs = row - (peak + np.log(np.exp(row - peak).sum()))
                raw, entry = piece(token)
                entry["logprob"] = float(logprobs[token])
                best: list[int] = []
                if top:
                    best = list(np.argpartition(-logprobs, top)[:top])
                    best.sort(key=lambda i: -logprobs[i])
                entry["top_logprobs"] = [
                    {**piece(int(i))[1], "logprob": float(logprobs[i])} for i in best
                ]
                entries.append(entry)
                out.append(token)
                text += raw
                decoded = text.decode("utf-8", errors="replace")
                cuts = [decoded.find(stop) for stop in stops if stop in decoded]
                if cuts:
                    text, finish = decoded[: min(cuts)].encode("utf-8"), "stop"
                    break
                if cfg.max_tokens and len(out) >= cfg.max_tokens:
                    break
        finally:
            steps.close()
        return text.decode("utf-8", errors="replace"), entries, len(out), finish

    def _remove_lora(self, adapter_id: str) -> None:
        """Take an applied adapter off the model."""
        found = next((entry for entry in self._loras if entry[0] == adapter_id), None)
        if found is None:
            raise RuntimeError(f"adapter {adapter_id!r} is not applied to this model")
        from llama_cpp import llama_cpp as _lcpp

        self._loras.remove(found)
        self._set_loras()
        _lcpp.llama_adapter_lora_free(found[3])

    def unload(self) -> None:
        with self._native:
            self._unload()

    def _unload(self) -> None:
        if self._model:
            model_name = self.model_name
            self._loras = []  # llama.cpp frees them with the model
            logger.info("Unloading model: %s", model_name)
            del self._model
            self._model = None
            self._architecture = None
            self._n_ctx = 0
            # V4 F5 — release the speculative-decoding draft alongside
            # the target so a subsequent load doesn't double up.
            if self._draft_model is not None:
                logger.info("Unloading speculative draft model")
                del self._draft_model
                self._draft_model = None
            # Force garbage collection to free GPU memory
            import gc

            gc.collect()
            logger.debug("Model unloaded: %s", model_name)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        with self._native:
            return self._generate(prompt, config)

    def _generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()
        logger.debug("Generating with max_tokens=%s, temp=%s", cfg.max_tokens, cfg.temperature)

        # OLLAMA_PARITY_PLAN P2-3 + Phase 11 P1 (V2 row 23). When
        # ``template_override`` is set and ``raw`` is False, render
        # the Modelfile's Go-template against the caller's prompt via
        # the proper Go-template evaluator (hfl.converter.go_template).
        # Covers ``{{ range }}``, ``{{ if }}``, ``{{- -}}`` trimming
        # and nested field access — not just the ``{{ .Prompt }}`` /
        # ``{{ .System }}`` placeholders the old regex handled. The
        # evaluator falls back to the literal template on parse
        # errors so a user's typo never crashes generation.
        effective_prompt = prompt
        if cfg.template_override and not cfg.raw:
            from hfl.converter.go_template import render_go_template

            effective_prompt = render_go_template(
                cfg.template_override,
                {"Prompt": prompt, "System": "", "Messages": []},
            )

        if cfg.logprobs is not None:
            start_ns = time.monotonic_ns()
            add_bos = bool(getattr(self, "_tokenizer_add_bos", True))
            encoded = effective_prompt.encode("utf-8")
            prompt_tokens = list(self._model.tokenize(encoded, add_bos=add_bos, special=True))
            text, entries, n_gen, finish = self._sample_with_logprobs(
                prompt_tokens, cfg, cfg.repeat_penalty, special=False
            )
            total_ns = time.monotonic_ns() - start_ns
            return GenerationResult(
                text=text,
                tokens_generated=n_gen,
                tokens_prompt=len(prompt_tokens),
                tokens_per_second=n_gen / (total_ns / 1e9) if total_ns else 0,
                stop_reason=finish,
                total_duration=total_ns,
                logprobs=entries,
            )

        call_kwargs: dict = {
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty,
            "stop": cfg.stop,
            "seed": cfg.seed if cfg.seed >= 0 else None,
        }
        # llama.cpp's perf counters accumulate per context; zero them so the
        # prompt-eval / eval split below describes THIS call.
        _perf_reset(self._model)
        start_ns = time.monotonic_ns()
        output = self._model(effective_prompt, **call_kwargs)
        total_ns = time.monotonic_ns() - start_ns
        elapsed = total_ns / 1e9

        text = output["choices"][0]["text"]
        usage = output.get("usage", {})
        n_gen = usage.get("completion_tokens", 0)
        n_prompt = usage.get("prompt_tokens", 0)

        # Guard the division: two monotonic_ns() reads can be equal on a fast
        # completion / low-resolution clock, making elapsed == 0. The argument is
        # evaluated regardless of log level, so an unguarded divide turns a
        # successful generation into a 500 whenever DEBUG logging is enabled.
        logger.debug(
            "Generated %s tokens in %.2fs (%.1f tok/s)",
            n_gen,
            elapsed,
            (n_gen / elapsed if elapsed > 0 else 0.0),
        )

        prompt_eval_ns, eval_ns = _split_durations(self._model, total_ns, n_prompt, n_gen)

        # Phase 7 P2-4: populate the legacy ``context`` array when
        # the caller opted in via ``keep_context=True``. llama-cpp's
        # ``tokenize`` on the concatenated prompt + response is the
        # cheapest way to obtain the integer sequence clients feed
        # back on the next turn.
        context_tokens: list[int] | None = None
        if cfg.keep_context:
            try:
                full = (effective_prompt + text).encode("utf-8", errors="replace")
                # Phase 11 P1 — V2 row 39. When the model's GGUF
                # says ``tokenizer.ggml.add_bos_token=false`` we
                # must not re-prepend BOS during tokenisation (Gemma
                # 4 is the canonical offender). Default True to
                # match llama-cpp's prior behaviour.
                add_bos = getattr(self, "_tokenizer_add_bos", True)
                context_tokens = list(self._model.tokenize(full, add_bos=bool(add_bos)))
            except Exception:  # noqa: BLE001
                logger.warning("keep_context requested but tokenize() failed", exc_info=True)
                context_tokens = []

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=n_prompt,
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
            stop_reason=output["choices"][0].get("finish_reason", "stop"),
            total_duration=total_ns,
            load_duration=0,
            prompt_eval_duration=prompt_eval_ns,
            eval_duration=eval_ns,
            context_tokens=context_tokens,
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        cfg = config or GenerationConfig()
        counted = CountedStream()
        model = self._model

        def _chunks() -> Iterator[str]:
            first: int | None = None
            finish: str | None = None
            for chunk in model(
                prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repeat_penalty=cfg.repeat_penalty,
                stop=cfg.stop,
                seed=cfg.seed if cfg.seed >= 0 else None,
                stream=True,
            ):
                if first is None:
                    first = getattr(model, "n_tokens", None)
                choice = chunk["choices"][0]
                finish = choice.get("finish_reason") or finish
                text = choice["text"]
                if text:
                    yield text
            _count(counted, model, first, finish)

        return counted.feed(self._holding(_chunks()))

    def _build_stop_list(
        self, caller_stop: list[str] | None, tools: list[dict] | None
    ) -> list[str]:
        """Compose the stop list passed to ``create_chat_completion``.

        The caller's own stop strings are always preserved. For Gemma 4
        models with ``tools`` supplied, we additionally append
        ``<tool_call|>`` so the model halts immediately after emitting
        a tool call instead of hallucinating the tool's response and
        continuing with a fabricated answer (observed in the wild on
        the bartowski/google_gemma-4-31B GGUF — without a stop, the
        model emits ``<|tool_response>`` tokens and fakes JSON output
        as if the tool had already run).

        Returns a list even when the caller passed ``None`` so the
        downstream kwargs pass a consistent type.
        """
        stop: list[str] = list(caller_stop) if caller_stop else []
        if tools and self._architecture == "gemma4":
            if "<tool_call|>" not in stop:
                stop.append("<tool_call|>")
        # GLM hands the turn to the tool with ``<|observation|>``, which its
        # GGUFs do not mark as end of generation: GLM-4-0414 went on to
        # invent the tool's reply and answer from it (measured). No other
        # family writes that string.
        if tools and "<|observation|>" not in stop:
            stop.append("<|observation|>")
        return stop

    def _tool_messages(
        self, messages: list[ChatMessage], tools: list[dict] | None
    ) -> tuple[list[dict], list[dict] | None, bool]:
        """The messages for ``create_chat_completion``, the tools to hand the
        template (None when HFL wrote them in itself), and whether to keep
        the model's control tokens in the text for the tool parsers."""
        msgs = self._messages_to_llama_cpp(messages)
        markers = bool(tools)
        if self._template_knows_tools:
            msgs = _history_for_template(msgs, self._chat_template)
        else:
            msgs, tools = _tools_as_text(msgs, tools), None
        if not self._template_takes_system:
            msgs = _fold_system(msgs)
        return msgs, tools, markers

    @staticmethod
    def _messages_to_llama_cpp(messages: list[ChatMessage]) -> list[dict]:
        """Convert internal ChatMessage list to llama-cpp-python format.

        Preserves ``tool_calls`` on assistant turns and ``name`` on tool
        turns so the underlying chat template can render them
        correctly. When a message carries ``images``, convert the
        content to llama-cpp's list-of-parts form
        (``[{"type":"text", "text": "..."}, {"type":"image_url",
        "image_url":{"url":"data:image/png;base64,..."}}]``) which
        the multimodal chat handlers recognise.
        """
        out: list[dict] = []
        for m in messages:
            # Text-only fast path
            if not m.images:
                entry: dict = {"role": m.role, "content": m.content or ""}
            else:
                parts: list[dict] = []
                if m.content:
                    parts.append({"type": "text", "text": m.content})
                for image_bytes in m.images:
                    uri = _image_data_uri(image_bytes)
                    parts.append({"type": "image_url", "image_url": {"url": uri}})
                entry = {"role": m.role, "content": parts}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            if m.name:
                entry["name"] = m.name
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            out.append(entry)
        return out

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        with self._native:
            return self._chat(messages, config, tools)

    def _chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()

        penalty = repeat_penalty_for(cfg, messages, tools)
        if cfg.logprobs is not None:
            if cfg.response_format is not None:
                raise NotImplementedError("logprobs together with a response format")
            start_ns = time.monotonic_ns()
            prompt, tools, markers = self._template_prompt(messages, cfg, tools)
            text, entries, n_gen, finish = self._sample_with_logprobs(
                prompt, cfg, penalty, special=markers
            )
            total_ns = time.monotonic_ns() - start_ns
            return GenerationResult(
                text=text,
                tokens_generated=n_gen,
                tokens_prompt=len(prompt),
                tokens_per_second=n_gen / (total_ns / 1e9) if total_ns else 0,
                stop_reason=finish,
                total_duration=total_ns,
                logprobs=entries,
            )
        msgs, tools, markers = self._tool_messages(messages, tools)
        # Set on every request, so one never inherits the last one's.
        for formatter in self._formatters:
            formatter.template_vars = reasoning_template_vars(cfg.reasoning)

        kwargs: dict = {
            "messages": msgs,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repeat_penalty": penalty,
            "stop": self._build_stop_list(cfg.stop, tools),
            "seed": cfg.seed if cfg.seed >= 0 else None,
        }
        if tools:
            # llama-cpp-python >= 0.3.0 forwards ``tools`` into the chat
            # template and parses tool calls back into the response.
            kwargs["tools"] = tools

        # OLLAMA_PARITY_PLAN P0-5: structured outputs.
        # Compile the request's response_format into a GBNF grammar
        # that llama-cpp enforces at sampling time. We use the dict
        # ``response_format`` kwarg that create_chat_completion accepts
        # natively (maps to OpenAI's JSON mode for free-form JSON, or
        # to a compiled schema grammar for strict conformance).
        _rf = cfg.response_format
        if _rf is not None:
            if _rf == "json":
                kwargs["response_format"] = {"type": "json_object"}
            elif isinstance(_rf, dict):
                kwargs["response_format"] = {
                    "type": "json_object",
                    "schema": _rf,
                }
            # ``GBNF:`` raw-grammar passthrough: build LlamaGrammar
            # directly so advanced users can ship custom grammars.
            elif isinstance(_rf, str) and _rf.startswith("GBNF:"):
                try:
                    from llama_cpp import LlamaGrammar

                    kwargs["grammar"] = LlamaGrammar.from_string(_rf[len("GBNF:") :])
                except ImportError:  # pragma: no cover — optional dep
                    pass

        # Nanosecond timings (Ollama-parity P1-3). ``monotonic_ns``
        # is the right clock for wall-clock deltas — perf_counter_ns
        # has the same resolution but ``monotonic_ns`` is what Ollama
        # uses internally.
        # See generate(): zero the per-context perf counters first.
        _perf_reset(self._model)
        start_ns = time.monotonic_ns()
        _render_special_tokens(self._model, markers)
        try:
            output = self._model.create_chat_completion(**kwargs)
        except TypeError:
            # Older llama-cpp-python without ``tools`` / ``response_format``
            # support — strip them and retry so the caller's text-based
            # parser can still extract calls.
            kwargs.pop("tools", None)
            kwargs.pop("response_format", None)
            kwargs.pop("grammar", None)
            output = self._model.create_chat_completion(**kwargs)
        finally:
            _render_special_tokens(self._model, False)
        total_ns = time.monotonic_ns() - start_ns
        elapsed = total_ns / 1e9  # seconds, for the tokens/s ratio

        message = output["choices"][0].get("message", {})
        text = message.get("content") or ""
        # Post-filter channel/think/turn markers for architectures
        # whose GGUFs don't ship a proper ``tokenizer.chat_template``
        # and whose vocab contains split-pipe reasoning delimiters.
        # No-op for architectures not in the filter set. When the
        # caller requested ``expose_reasoning`` (Phase 5 P1-1, Ollama
        # ``think=true``) we leave the markers IN the text so the
        # route layer can separate reasoning from answer.
        if self._architecture in _ARCHITECTURE_CHANNEL_FILTER and not cfg.expose_reasoning:
            text = _strip_channel_markers(text, self._architecture)
        tool_calls = message.get("tool_calls")

        # Normalise tool_calls shape: llama-cpp-python may return
        # [{"id": ..., "type": "function", "function": {"name", "arguments"}}]
        # with ``arguments`` as a JSON string. We want a parsed dict.
        normalised_tool_calls: list[dict] | None = None
        if tool_calls:
            import json as _json

            normalised_tool_calls = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except (ValueError, TypeError):
                        args = {}
                normalised_tool_calls.append(
                    {
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": args or {},
                        }
                    }
                )

        usage = output.get("usage", {})
        n_gen = usage.get("completion_tokens", 0)
        n_prompt = usage.get("prompt_tokens", 0)

        # Estimate prompt_eval / eval split from token counts.
        # llama-cpp-python doesn't surface the pre-first-token delta
        # natively, so we apportion total_ns proportionally to
        prompt_eval_ns, eval_ns = _split_durations(self._model, total_ns, n_prompt, n_gen)

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=n_prompt,
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
            # "length" when max_tokens cut the reply; the routes report it as
            # OpenAI's finish_reason, Ollama's done_reason, Anthropic's
            # stop_reason. (Tool calls are reported by the routes themselves.)
            stop_reason="length"
            if output["choices"][0].get("finish_reason") == "length"
            else "stop",
            tool_calls=normalised_tool_calls,
            total_duration=total_ns,
            load_duration=0,  # Model was already loaded — this is chat, not load()
            prompt_eval_duration=prompt_eval_ns,
            eval_duration=eval_ns,
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        cfg = config or GenerationConfig()
        penalty = repeat_penalty_for(cfg, messages, tools)
        msgs, tools, markers = self._tool_messages(messages, tools)
        # Set on every request, so one never inherits the last one's.
        for formatter in self._formatters:
            formatter.template_vars = reasoning_template_vars(cfg.reasoning)

        kwargs: dict = {
            "messages": msgs,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repeat_penalty": penalty,
            "stop": self._build_stop_list(cfg.stop, tools),
            "seed": cfg.seed if cfg.seed >= 0 else None,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            iterator = self._model.create_chat_completion(**kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            iterator = self._model.create_chat_completion(**kwargs)

        counted = CountedStream()
        model = self._model

        def _raw_chunks() -> Iterator[str]:
            first: int | None = None
            finish: str | None = None
            # The stream detokenizes as it is read, so the markers stay on
            # for as long as it is.
            _render_special_tokens(model, markers)
            try:
                for chunk in iterator:
                    if first is None:
                        first = getattr(model, "n_tokens", None)
                    choice = chunk["choices"][0]
                    finish = choice.get("finish_reason") or finish
                    text = choice.get("delta", {}).get("content", "")
                    if text:
                        yield text
            finally:
                _render_special_tokens(model, False)
            _count(counted, model, first, finish)

        if self._architecture in _ARCHITECTURE_CHANNEL_FILTER and not cfg.expose_reasoning:
            harmony = self._architecture == "gpt-oss"
            # Held outermost, so closing the stream releases it at once.
            return counted.feed(
                self._holding(_filter_gemma4_stream(_raw_chunks(), harmony=harmony))
            )
        # ``expose_reasoning=True`` (Phase 5 P1-1) → let the raw
        # chunks through so the caller sees the reasoning channel.
        return counted.feed(self._holding(_raw_chunks()))

    @property
    def model_name(self) -> str:
        return self._model_path.split("/")[-1] if self._model_path else ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def context_size(self) -> int:
        return self._n_ctx

    @property
    def acceleration(self) -> str | None:
        return self._acceleration

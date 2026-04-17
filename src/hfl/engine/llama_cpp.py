# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Inference backend based on llama-cpp-python.

This is the main backend for GGUF models.
Supports CPU, CUDA, Metal, and Vulkan.
"""

import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)

# ``llama_cpp`` is an optional dependency (HFL ``[llama]`` extra). The
# import is wrapped in try/except so the rest of the module — including
# the GGUF chat-format detection helper, the architecture map, and any
# test that monkeypatches ``hfl.engine.llama_cpp.Llama`` — can be
# imported and exercised in environments without llama-cpp-python (CI
# default, doc generators, type checkers).
try:
    from llama_cpp import Llama  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — exercised by the [dev] CI matrix
    Llama = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_stderr():
    """Temporarily suppresses stderr (to silence Metal/CUDA logs)."""
    # Save the original descriptor
    stderr_fd = sys.stderr.fileno()
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


# Architectures whose vocabulary contains split-pipe channel/think/turn
# markers (``<|channel>...<channel|>`` etc.) that need post-filtering
# on chat output. This is a separate set from ``_ARCHITECTURE_CHAT_FORMAT``
# because the override to ``gemma`` (= Gemma 2 format) is not enough to
# stop the model from emitting its reasoning-format markers when the
# GGUF doesn't ship a proper ``tokenizer.chat_template``.
_ARCHITECTURE_CHANNEL_FILTER: set[str] = {"gemma4"}

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

    def __init__(self) -> None:
        self._buffer: str = ""
        # ``True`` while we're inside a thought/think block that
        # should be fully suppressed (content included).
        self._suppress: bool = False

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
            for marker, kind in _GEMMA4_STREAM_MARKERS:
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
        return _strip_gemma4_channel_markers(leftover)


def _filter_gemma4_stream(iterator: Iterator[str]) -> Iterator[str]:
    """Stream wrapper around :class:`_Gemma4StreamFilter`."""
    filt = _Gemma4StreamFilter()
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
        import gguf  # type: ignore[import-not-found]
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
        import gguf  # type: ignore[import-not-found]
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
            import numpy as np  # type: ignore[import-not-found]

            arr = np.asarray(value)
            return int(arr.flat[0])
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
    }


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
        block_count = info.get("block_count") or 0
        embedding_length = info.get("embedding_length") or 0
        head_count = info.get("head_count") or 0
        head_count_kv = info.get("head_count_kv") or 0

        if block_count and head_count and head_count_kv and embedding_length:
            # GQA-aware: n_kv_heads * head_dim, where head_dim =
            # embedding_length / n_heads.
            head_dim = embedding_length // head_count
            kv_dim = head_count_kv * head_dim
            kv_bytes = 2 * block_count * n_ctx * kv_dim * 2
            kv_gb = kv_bytes / (1024**3)
        elif block_count and embedding_length:
            # Fallback: no GQA metadata → assume n_kv_heads == n_heads
            # (i.e. non-GQA worst case). Over-estimates GQA models but
            # is still the right thing to do when we can't measure.
            kv_bytes = 2 * block_count * n_ctx * embedding_length * 2
            kv_gb = kv_bytes / (1024**3)

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
        err = OutOfMemoryError(required_gb=required_gb, available_gb=available_gb)
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
        from llama_cpp.llama_chat_format import (
            Gemma3ChatHandler,
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

    # Arch → handler. Order matters: most-specific substring first
    # so e.g. ``llava-v1.6`` doesn't match the v1.5 handler.
    if "gemma" in arch and ("3" in arch or "4" in arch):
        handler_cls = Gemma3ChatHandler
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

    return handler_cls(clip_model_path=clip_model_path, verbose=verbose)


class LlamaCppEngine(InferenceEngine):
    """llama.cpp inference engine."""

    def __init__(self):
        self._model: "Llama | None" = None
        self._model_path: str = ""
        self._architecture: str | None = None
        # True when the model was loaded with a CLIP projector and
        # accepts images in ``create_chat_completion`` messages.
        # Phase 4 P0-6.
        self._is_multimodal: bool = False

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
        user_n_ctx = kwargs.get("n_ctx", None)
        explicit_n_ctx = user_n_ctx is not None and user_n_ctx > 0
        n_ctx = user_n_ctx if explicit_n_ctx else hfl_config.default_ctx_size
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)

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

        # Flash-attention is not safe for every architecture: llama-cpp-
        # python's flash-attn path has been historically crash-prone for
        # new arches. Force-disable for known-bad arches unless the
        # caller explicitly opted in.
        if "flash_attn" in kwargs:
            flash_attn = kwargs["flash_attn"]
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
            # Convention: ``mmproj-*.gguf`` in the same directory.
            for candidate in path.parent.glob("mmproj-*.gguf"):
                clip_model_path = str(candidate)
                logger.info("Auto-detected CLIP projector: %s", candidate.name)
                break

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
            context = _suppress_stderr if not verbose else _nullcontext
            with context():
                llama_kwargs: dict = dict(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=kwargs.get("n_threads", 0) or None,
                    verbose=verbose,
                    flash_attn=flash_attn,
                    chat_format=chat_format,
                )
                # Phase 11 P1: KV cache quantisation. Maps
                # ``"q4_0"`` / ``"q8_0"`` strings to llama-cpp's
                # ``type_k`` / ``type_v`` integer enum. ``"f16"`` is
                # the default and leaves the fields unset so the
                # library picks its own default.
                kv_type = kwargs.get("kv_cache_type") or hfl_config.kv_cache_type
                if kv_type and kv_type != "f16":
                    try:
                        from llama_cpp import llama_cpp as _lcpp  # type: ignore
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
                # Phase 8 P3-2: LoRA adapters. llama-cpp-python accepts
                # a single ``lora_path`` at load time; we honour the
                # first path the Modelfile declared. Stacking more
                # than one requires a post-load ``apply_lora_from_file``
                # call which landed in llama-cpp 0.3+. For portability
                # we only wire the first path here and log the rest.
                lora_paths = kwargs.get("lora_paths") or []
                if lora_paths:
                    primary = lora_paths[0]
                    logger.info("Loading LoRA adapter: %s", primary)
                    llama_kwargs["lora_path"] = primary
                    if len(lora_paths) > 1:
                        logger.warning(
                            "%d additional LoRA adapter(s) ignored — "
                            "llama-cpp's ``lora_path`` accepts only one",
                            len(lora_paths) - 1,
                        )
                self._model = Llama(**llama_kwargs)
            self._model_path = model_path
            self._architecture = architecture
            self._is_multimodal = chat_handler is not None
            elapsed = time.perf_counter() - start_time
            mm_note = " (multimodal)" if self._is_multimodal else ""
            logger.info("Model loaded in %.2fs%s: %s", elapsed, mm_note, path.name)
        except Exception as e:
            logger.error("Failed to load model %s: %s", path.name, e)
            raise

    def unload(self) -> None:
        if self._model:
            model_name = self.model_name
            logger.info("Unloading model: %s", model_name)
            del self._model
            self._model = None
            self._architecture = None
            # Force garbage collection to free GPU memory
            import gc

            gc.collect()
            logger.debug("Model unloaded: %s", model_name)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()
        logger.debug("Generating with max_tokens=%s, temp=%s", cfg.max_tokens, cfg.temperature)

        # OLLAMA_PARITY_PLAN P2-3. When ``template_override`` is set
        # and ``raw`` is False, render the Modelfile-style template
        # against the caller's prompt before feeding it to the
        # model. We handle the two placeholders that cover the vast
        # majority of real Modelfiles — ``{{ .Prompt }}`` and
        # ``{{ .System }}`` — without pulling in a full Go-template
        # parser. Anything else passes through the template as a
        # literal; sophisticated templates should go through
        # ``/api/chat`` where llama-cpp's Jinja renderer is used.
        effective_prompt = prompt
        if cfg.template_override and not cfg.raw:
            tmpl = cfg.template_override
            tmpl = re.sub(r"\{\{\s*\.Prompt\s*\}\}", prompt, tmpl)
            tmpl = re.sub(r"\{\{\s*\.System\s*\}\}", "", tmpl)
            effective_prompt = tmpl

        start_ns = time.monotonic_ns()
        output = self._model(
            effective_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop,
            seed=cfg.seed if cfg.seed >= 0 else None,
        )
        total_ns = time.monotonic_ns() - start_ns
        elapsed = total_ns / 1e9

        text = output["choices"][0]["text"]
        usage = output.get("usage", {})
        n_gen = usage.get("completion_tokens", 0)
        n_prompt = usage.get("prompt_tokens", 0)

        logger.debug("Generated %s tokens in %.2fs (%.1f tok/s)", n_gen, elapsed, n_gen / elapsed)

        # See chat() for the rationale behind this split.
        total_tokens = max(1, n_prompt + n_gen)
        prompt_eval_ns = int(total_ns * n_prompt / total_tokens)
        eval_ns = total_ns - prompt_eval_ns

        # Phase 7 P2-4: populate the legacy ``context`` array when
        # the caller opted in via ``keep_context=True``. llama-cpp's
        # ``tokenize`` on the concatenated prompt + response is the
        # cheapest way to obtain the integer sequence clients feed
        # back on the next turn.
        context_tokens: list[int] | None = None
        if cfg.keep_context:
            try:
                full = (effective_prompt + text).encode("utf-8", errors="replace")
                context_tokens = list(self._model.tokenize(full, add_bos=False))
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

        for chunk in self._model(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if text:
                yield text

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
        return stop

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
        import base64 as _base64

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
                    # llama-cpp's multimodal handlers accept
                    # ``data:image/...;base64,...`` URIs on the
                    # ``image_url.url`` field. Sniff the MIME from
                    # magic bytes so the URI is correct even if we
                    # came in as raw bytes.
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
                    uri = f"data:{mime};base64," + _base64.b64encode(image_bytes).decode("ascii")
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
        cfg = config or GenerationConfig()

        msgs = self._messages_to_llama_cpp(messages)

        kwargs: dict = dict(
            messages=msgs,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=self._build_stop_list(cfg.stop, tools),
        )
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
        start_ns = time.monotonic_ns()
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
            text = _strip_gemma4_channel_markers(text)
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
        # prompt vs. generated tokens — the ratio is what matters
        # for monitoring (tokens/sec stays consistent).
        total_tokens = max(1, n_prompt + n_gen)
        prompt_eval_ns = int(total_ns * n_prompt / total_tokens)
        eval_ns = total_ns - prompt_eval_ns

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=n_prompt,
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
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
        msgs = self._messages_to_llama_cpp(messages)

        kwargs: dict = dict(
            messages=msgs,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=self._build_stop_list(cfg.stop, tools),
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools

        try:
            iterator = self._model.create_chat_completion(**kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            iterator = self._model.create_chat_completion(**kwargs)

        def _raw_chunks() -> Iterator[str]:
            for chunk in iterator:
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text

        if self._architecture in _ARCHITECTURE_CHANNEL_FILTER and not cfg.expose_reasoning:
            yield from _filter_gemma4_stream(_raw_chunks())
        else:
            # ``expose_reasoning=True`` (Phase 5 P1-1) → let the raw
            # chunks through so the caller sees the reasoning channel.
            yield from _raw_chunks()

    @property
    def model_name(self) -> str:
        return self._model_path.split("/")[-1] if self._model_path else ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

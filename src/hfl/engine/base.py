# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Abstract interface for inference engines."""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from types import TracebackType

# =============================================================================
# LLM (Text Generation) Types
# =============================================================================


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    # Tool-calling extensions (only populated when relevant):
    # - ``tool_calls``: assistant turn requesting one or more function calls.
    #   Each entry is the canonical Ollama shape:
    #   ``{"function": {"name": str, "arguments": dict}}``.
    # - ``name``: for role=tool, the function name whose result this carries.
    # - ``tool_call_id``: optional id linking a tool result to a prior call.
    tool_calls: list[dict] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    # Vision / multimodal extension (Phase 4, P0-6).
    # Each entry is raw, already-decoded image bytes (PNG / JPEG /
    # WEBP / GIF). The router decodes base64 / data-URIs and runs
    # ``image_validator.validate_image`` before populating this
    # list, so engines can assume every entry is a bounded, sniffed
    # raster image. ``None`` means "text-only message" — the vast
    # majority of messages.
    images: list[bytes] | None = None


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stop: list[str] | None = None
    repeat_penalty: float = 1.1
    seed: int = -1
    # Structured-output constraint (OLLAMA_PARITY_PLAN P0-5).
    # Backends that support grammar-based constrained decoding
    # (llama-cpp-python GBNF, vLLM GuidedDecodingParams, Transformers
    # + outlines) honour this to force the model's output to conform
    # to a JSON schema or a raw GBNF grammar. ``None`` means
    # unconstrained generation.
    #
    # - ``"json"`` → free-form JSON object (value is the literal
    #   string; backends compile it to their "any JSON" grammar).
    # - ``dict`` → JSON Schema (validated at the route boundary).
    # - ``str`` starting with ``"GBNF:"`` → raw GBNF grammar body
    #   (advanced users; bypasses schema validation).
    response_format: str | dict | None = None
    # Expose the model's internal reasoning / thinking channel in the
    # output. When False (default), architecture-specific channel
    # filters (e.g. Gemma 4's split-pipe ``<|channel>thought...``
    # markers) strip the reasoning from ``content``. When True, the
    # filter is disabled and engines emit the raw channel text. The
    # route layer is responsible for post-processing the raw text
    # into a separate ``thinking`` field on the response envelope
    # (OLLAMA_PARITY_PLAN P1-1, Ollama 2026 ``think=true``).
    expose_reasoning: bool = False
    # Multi-level thinking intensity (Phase 10 P1).
    # - ``"off"``: reasoning suppressed (same as legacy ``expose_reasoning=False``).
    # - ``"low"``: shortest viable chain; honoured by GPT-OSS-class models.
    # - ``"medium"``: default "thinking on" behaviour (same as ``expose_reasoning=True``).
    # - ``"high"``: full-depth reasoning channel.
    # Engines that can't differentiate the levels treat ``"low"`` /
    # ``"medium"`` / ``"high"`` all as "expose reasoning"; the
    # route still honours ``"off"``.
    thinking_level: str = "off"
    # Whether the model should reason at all, as the client asked (Ollama
    # ``think``, OpenAI ``reasoning_effort`` / ``reasoning.effort``,
    # Anthropic ``thinking``): ``None`` leaves the model's own default,
    # ``"off"`` switches it off where the template can, ``"low"`` /
    # ``"medium"`` / ``"high"`` switch it on. Unlike ``thinking_level``,
    # which only decides what the reply shows, this reaches the prompt —
    # see ``reasoning_template_vars``.
    reasoning: str | None = None
    # Per-request chat-template override (OLLAMA_PARITY_PLAN P2-3).
    # When set, the engine uses this Jinja template in place of the
    # model's default for this request only. Ignored by engines that
    # don't expose a pluggable template (vLLM). ``None`` means "use
    # the model's default template".
    template_override: str | None = None
    # Raw prompt mode (OLLAMA_PARITY_PLAN P2-3, Ollama ``raw=true``).
    # When True the engine forwards the prompt to the model verbatim,
    # bypassing the chat template and the BOS token. Meant for
    # evaluation harnesses and manual prompt engineering. Only
    # meaningful on ``/api/generate``; chat routes ignore it.
    raw: bool = False
    # Return the encoded ``prompt + response`` token array so clients
    # can feed it back via Ollama's legacy ``context`` field for
    # multi-turn continuation (OLLAMA_PARITY_PLAN P2-4). Default off
    # because keeping token arrays in memory costs RAM and most
    # clients now use /api/chat with role-tagged messages instead.
    keep_context: bool = False
    # Per-token log probabilities: ``None`` (default) for none; ``0`` for
    # the drawn token's only; 1-20 for that many best alternatives too.
    # An engine that cannot give them raises ``NotImplementedError`` rather
    # than answer without them.
    logprobs: int | None = None
    # Speculative decoding (Phase 15 P2 — V2 row 11). Path or
    # registry name of a smaller "draft" model. Empty / None
    # disables speculation (the default). Engines that don't expose
    # the knob silently ignore it.
    draft_model: str | None = None
    # Whether the client set ``repeat_penalty`` itself (the OpenAI and
    # Anthropic APIs have no such parameter). See ``repeat_penalty_for``.
    repeat_penalty_chosen: bool = False


def held(lock: "threading.Lock", chunks: Iterator[str]) -> Iterator[str]:
    """``chunks`` read with ``lock`` held, from the first item until the
    stream is exhausted or closed — by the thread doing the native work.

    For engines that keep one model instance (llama.cpp, MLX,
    Transformers): the dispatcher's slot alone was not enough. When a client
    drops a stream the slot is released while the worker thread may still
    be inside the model (a long prefill cannot be interrupted), and the next
    request then ran on the same model at the same time — ``llama_decode
    returned -3`` and a Metal segfault under Claude Code (measured). The
    lock must be a plain ``Lock``: a stream can be closed on a different
    thread than the one that started it, which an ``RLock`` forbids.
    """
    with lock:
        yield from chunks


def reasoning_template_vars(reasoning: str | None) -> dict[str, Any]:
    """The chat-template variables that turn a model's reasoning on or off.

    Each family reads its own: Qwen3 and GLM ``enable_thinking``, DeepSeek
    V3.1 ``thinking``, gpt-oss ``reasoning_effort`` (which cannot be off:
    ``"low"`` is its least). A template ignores the ones it does not read.
    Nothing is set when the client did not ask, so the model's default
    stands.
    """
    if reasoning is None:
        return {}
    on = reasoning != "off"
    return {
        "enable_thinking": on,
        "thinking": on,
        "reasoning_effort": reasoning if on else "low",
    }


def repeat_penalty_for(
    cfg: GenerationConfig, messages: list[ChatMessage], tools: list[dict] | None
) -> float:
    """The repetition penalty to sample with.

    A turn with tools, or answering from a tool's result, has to repeat
    what it was just given — argument names, the values in the result — and
    the default penalty (1.1) pushes against exactly those tokens:
    DeepSeek-R1-0528 8B, handed a weather result, answered "I cannot fulfill
    your request" with it and correctly without it (measured). So such turns
    use none (1.0, llama.cpp's own default) unless the client chose one.
    """
    if cfg.repeat_penalty_chosen:
        return cfg.repeat_penalty
    if tools or any(m.role == "tool" or m.tool_calls for m in messages):
        return 1.0
    return cfg.repeat_penalty


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int = 0
    tokens_prompt: int = 0
    tokens_per_second: float = 0.0
    stop_reason: str = "stop"
    # Populated when the engine (or a downstream parser) produced structured
    # tool calls. Shape: list of ``{"function": {"name", "arguments": dict}}``.
    tool_calls: list[dict] | None = None
    # Nanosecond-precision timings (Ollama-parity P1-3).
    # Engines populate as many as they can measure cleanly; unknown
    # values stay at 0. The invariant intended by Ollama is:
    #   total_duration ≈ load_duration + prompt_eval_duration + eval_duration
    # with some slop for overhead. Callers should treat the field as
    # advisory rather than exact.
    total_duration: int = 0  # Wall-clock of the entire request (ns)
    load_duration: int = 0  # Time to load the model (0 if already warm) (ns)
    prompt_eval_duration: int = 0  # Time to process the prompt (ns)
    eval_duration: int = 0  # Time spent generating tokens (ns)
    # Encoded prompt + response tokens, for Ollama's legacy ``context``
    # multi-turn continuation (P2-4). Only populated when
    # ``GenerationConfig.keep_context`` is True; otherwise stays
    # ``None`` so default payloads don't carry large int arrays.
    context_tokens: list[int] | None = None
    # Per-token logprobs (Phase 12 P1 — V2 row 7). Shape mirrors the
    # OpenAI response: ``[{"token", "logprob", "top_logprobs": [{"token",
    # "logprob"}, ...]}, ...]``. ``None`` when the caller didn't
    # request them.
    logprobs: list[dict] | None = None


class CountedStream(Iterator[str]):
    """A token stream that knows, once exhausted, how many tokens it cost.

    ``prompt_tokens`` / ``completion_tokens`` stay ``None`` until the
    engine has counted them — and for good if it cannot; callers must not
    make a number up in their place. One object per request, so concurrent
    streams never share their counts.
    """

    def __init__(self) -> None:
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self._it: Iterator[str] = iter(())

    def feed(self, source: Iterator[str]) -> "CountedStream":
        self._it = source
        return self

    def __iter__(self) -> "CountedStream":
        return self

    def __next__(self) -> str:
        return next(self._it)

    def close(self) -> None:
        close = getattr(self._it, "close", None)
        if close is not None:
            close()


def stream_counts(stream: object) -> tuple[int | None, int | None]:
    """``(prompt_tokens, completion_tokens)`` of an exhausted engine stream,
    or ``(None, None)`` when the engine does not count them."""
    counts = (getattr(stream, "prompt_tokens", None), getattr(stream, "completion_tokens", None))
    return tuple(  # type: ignore[return-value]
        value if isinstance(value, int) and not isinstance(value, bool) else None
        for value in counts
    )


def refuse_logprobs(config: "GenerationConfig | None", backend: str) -> None:
    """For a backend that cannot give per-token logprobs: an error when they
    are asked for, rather than an answer without them."""
    if config is not None and config.logprobs is not None:
        raise NotImplementedError(f"the {backend} backend cannot return logprobs")


class InferenceEngine(ABC):
    """Interface that all backends must implement."""

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Loads the model into memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Releases the model from memory."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generates text synchronously."""
        ...

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Generates text token by token (streaming)."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> GenerationResult:
        """Synchronous chat with message format.

        Args:
            messages: Chat history, possibly including ``role=tool`` results
                and prior assistant ``tool_calls``.
            config: Sampling configuration.
            tools: Optional list of OpenAI/Ollama-shaped tool definitions
                (``[{"type": "function", "function": {...}}, ...]``). If
                provided and supported by the backend, the model's native
                chat template is applied with tool awareness so that
                structured tool calls can be emitted.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        """Streaming chat token by token.

        Same semantics as :meth:`chat` for the ``tools`` argument.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the loaded model."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is in memory."""
        ...

    @property
    def context_size(self) -> int:
        """Context window (tokens) the loaded model was opened with.

        ``0`` means "unknown / not tracked by this backend". Callers use
        it to decide whether a request asking for a different ``num_ctx``
        needs a reload, so an engine that can't report its context is
        simply never reloaded on that account.
        """
        return 0

    def count_prompt_tokens(
        self,
        messages: list["ChatMessage"],
        config: "GenerationConfig | None" = None,
        tools: list[dict] | None = None,
    ) -> int:
        """The tokens ``chat(messages, config, tools)`` would feed the model:
        the prompt as this engine renders it (template, tools, reasoning
        switch) and tokenizes it, with nothing generated.

        Raises ``NotImplementedError`` where that cannot be known exactly —
        a count that is only close would be a wrong answer.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot count a prompt's tokens")

    @property
    def supports_structured_output(self) -> bool:
        """Whether a response format (JSON, a JSON schema, a GBNF grammar)
        constrains this engine's sampling.

        ``False`` by default: MLX and Transformers ignored ``format`` /
        ``response_format`` silently and answered prose where JSON was asked.
        A route refuses such a request instead (400, a fixed sentence)."""
        return False

    @property
    def supports_concurrent_inference(self) -> bool:
        """Whether two inferences may run against this engine at once.

        ``False`` for every backend that drives a single non-reentrant model
        instance — llama.cpp and Transformers-GPU both keep one KV cache, so
        overlapping calls corrupt each other's state and produce garbage
        rather than an error. Only a backend with its own internal batching
        scheduler (vLLM) may report ``True``.

        The dispatcher's ``max_inflight`` is clamped to 1 when the loaded
        engine reports ``False``, so an operator who sets
        ``HFL_NUM_PARALLEL`` / ``OLLAMA_NUM_PARALLEL`` out of habit gets a
        warning and a safe server instead of silent corruption.
        """
        return False

    @property
    def acceleration(self) -> str | None:
        """Human-readable summary of the hardware the model is running on.

        E.g. ``"MTL0 (Apple M3 Max) · 41/41 layers on GPU · 8.4 GiB in
        MTL0 buffers"``. ``None`` means the backend doesn't report it —
        which is not the same as "no acceleration", so callers should
        present it as unknown rather than as CPU-only.
        """
        return None

    # Context manager support for automatic resource cleanup
    def __enter__(self) -> "InferenceEngine":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit context manager - automatically unload model."""
        if self.is_loaded:
            self.unload()

    async def __aenter__(self) -> "InferenceEngine":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit async context manager - automatically unload model."""
        if self.is_loaded:
            await asyncio.to_thread(self.unload)


# =============================================================================
# TTS (Text-to-Speech) Types
# =============================================================================


@dataclass
class TTSConfig:
    """Configuration for text-to-speech synthesis."""

    voice: str = "default"
    speed: float = 1.0
    language: str = "en"
    sample_rate: int = 22050
    format: str = "wav"  # wav, mp3, ogg


@dataclass
class AudioResult:
    """Result of audio synthesis."""

    audio: bytes
    sample_rate: int
    duration: float
    format: str
    metadata: dict = field(default_factory=dict)


class AudioEngine(ABC):
    """Abstract base class for audio synthesis engines (TTS)."""

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Loads the TTS model into memory.

        Args:
            model_path: Path to the model directory or file
            **kwargs: Backend-specific options (device, dtype, etc.)
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Releases the model from memory."""
        ...

    @abstractmethod
    def synthesize(self, text: str, config: TTSConfig | None = None) -> AudioResult:
        """Synthesizes text to audio.

        Args:
            text: Text to synthesize
            config: TTS configuration (voice, speed, language, etc.)

        Returns:
            AudioResult with audio bytes and metadata
        """
        ...

    @abstractmethod
    def synthesize_stream(self, text: str, config: TTSConfig | None = None) -> Iterator[bytes]:
        """Synthesizes text to audio in streaming chunks.

        Args:
            text: Text to synthesize
            config: TTS configuration

        Yields:
            Audio data chunks (raw PCM or encoded)
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is in memory."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the loaded model."""
        ...

    @property
    def supported_voices(self) -> list[str]:
        """List of supported voice identifiers."""
        return ["default"]

    @property
    def supported_languages(self) -> list[str]:
        """List of supported language codes."""
        return ["en"]

    # Context manager support for automatic resource cleanup
    def __enter__(self) -> "AudioEngine":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit context manager - automatically unload model."""
        if self.is_loaded:
            self.unload()

    async def __aenter__(self) -> "AudioEngine":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Exit async context manager - automatically unload model."""
        if self.is_loaded:
            await asyncio.to_thread(self.unload)


def completion_prompt(prompt: str, cfg: GenerationConfig) -> str:
    """``prompt`` through the request's (or the Modelfile's) Go template,
    unless raw — for every engine's plain completion, streaming or not (the
    llama.cpp stream and llama-server used to skip it)."""
    if cfg.template_override and not cfg.raw:
        from hfl.converter.go_template import render_go_template

        return render_go_template(
            cfg.template_override, {"Prompt": prompt, "System": "", "Messages": []}
        )
    return prompt

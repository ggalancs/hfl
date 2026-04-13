# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Inference backend based on llama-cpp-python.

This is the main backend for GGUF models.
Supports CPU, CUDA, Metal, and Vulkan.
"""

import logging
import os
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


class LlamaCppEngine(InferenceEngine):
    """llama.cpp inference engine."""

    def __init__(self):
        self._model: "Llama | None" = None
        self._model_path: str = ""

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
        n_ctx = kwargs.get("n_ctx", None)
        if n_ctx is None or n_ctx == 0:
            n_ctx = hfl_config.default_ctx_size  # 0 = auto from GGUF metadata
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)

        # Chat format selection: prefer an explicit caller override; otherwise
        # try to detect a known architecture from the GGUF header (fixes the
        # Gemma family which llama-cpp-python's fallback misidentifies as
        # Llama-2 ``[INST]``). ``None`` means "let llama-cpp-python decide".
        chat_format = kwargs.get("chat_format")
        if chat_format is None:
            chat_format = _detect_chat_format_from_gguf(model_path)

        logger.info("Loading GGUF model: %s", path.name)
        logger.debug(
            "Model path: %s, n_ctx=%s, n_gpu_layers=%s, chat_format=%s",
            model_path,
            n_ctx,
            n_gpu_layers,
            chat_format,
        )

        start_time = time.perf_counter()
        try:
            # Suppress Metal/CUDA initialization messages if verbose=False
            context = _suppress_stderr if not verbose else _nullcontext
            with context():
                self._model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=kwargs.get("n_threads", 0) or None,
                    verbose=verbose,
                    flash_attn=kwargs.get("flash_attn", True),
                    chat_format=chat_format,
                )
            self._model_path = model_path
            elapsed = time.perf_counter() - start_time
            logger.info("Model loaded in %.2fs: %s", elapsed, path.name)
        except Exception as e:
            logger.error("Failed to load model %s: %s", path.name, e)
            raise

    def unload(self) -> None:
        if self._model:
            model_name = self.model_name
            logger.info("Unloading model: %s", model_name)
            del self._model
            self._model = None
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

        t0 = time.perf_counter()
        output = self._model(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            stop=cfg.stop,
            seed=cfg.seed if cfg.seed >= 0 else None,
        )
        elapsed = time.perf_counter() - t0

        text = output["choices"][0]["text"]
        usage = output.get("usage", {})
        n_gen = usage.get("completion_tokens", 0)

        logger.debug("Generated %s tokens in %.2fs (%.1f tok/s)", n_gen, elapsed, n_gen / elapsed)

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
            stop_reason=output["choices"][0].get("finish_reason", "stop"),
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

    @staticmethod
    def _messages_to_llama_cpp(messages: list[ChatMessage]) -> list[dict]:
        """Convert internal ChatMessage list to llama-cpp-python format.

        Preserves ``tool_calls`` on assistant turns and ``name`` on tool
        turns so the underlying chat template can render them correctly.
        """
        out: list[dict] = []
        for m in messages:
            entry: dict = {"role": m.role, "content": m.content or ""}
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
            stop=cfg.stop,
        )
        if tools:
            # llama-cpp-python >= 0.3.0 forwards ``tools`` into the chat
            # template and parses tool calls back into the response.
            kwargs["tools"] = tools

        t0 = time.perf_counter()
        try:
            output = self._model.create_chat_completion(**kwargs)
        except TypeError:
            # Older llama-cpp-python without ``tools`` support — fall back
            # so the caller's text-based parser can still extract calls.
            kwargs.pop("tools", None)
            output = self._model.create_chat_completion(**kwargs)
        elapsed = time.perf_counter() - t0

        message = output["choices"][0].get("message", {})
        text = message.get("content") or ""
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

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
            tool_calls=normalised_tool_calls,
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
            stop=cfg.stop,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools

        try:
            iterator = self._model.create_chat_completion(**kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            iterator = self._model.create_chat_completion(**kwargs)

        for chunk in iterator:
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                yield text

    @property
    def model_name(self) -> str:
        return self._model_path.split("/")[-1] if self._model_path else ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

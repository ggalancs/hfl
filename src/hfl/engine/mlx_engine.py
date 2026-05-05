# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""MLX backend for Apple Silicon (Phase 13 P1 — V2 row 14).

Apple's MLX framework hits raw Metal directly; for Llama-family
architectures it outperforms llama-cpp's Metal path on M3/M4 Pro/Max
(3-10% on prompt processing, 15-25% on decode at fp16) and supports
mixed-precision (q4 / q5 / q8) quantisation without conversion.

This module implements ``InferenceEngine`` over ``mlx-lm``. The
dependency is behind the ``[mlx]`` extra and only loaded on
``darwin-arm64``; on every other platform the import is a no-op
and ``is_available()`` returns False.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any, Iterator

from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)

logger = logging.getLogger(__name__)

__all__ = ["MLXEngine", "is_available"]


def is_available() -> bool:
    """Return True iff the ``mlx-lm`` SDK is importable on the host.

    Also gates on ``platform.system() == 'Darwin'`` and
    ``machine() in {'arm64', 'aarch64'}`` so a wayward Linux
    container pinning ``hfl[mlx]`` in its Dockerfile fails fast.
    """
    if platform.system() != "Darwin":
        return False
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return False
    try:
        import mlx_lm  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


class MLXEngine(InferenceEngine):
    """Inference engine wrapping mlx-lm's ``generate`` helper.

    This engine is intentionally thin: mlx-lm's own API is stable
    and does the heavy lifting. We just adapt to HFL's
    ``ChatMessage`` / ``GenerationConfig`` / ``GenerationResult``
    contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_path: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, model_path: str, **kwargs: Any) -> None:
        if not is_available():
            raise RuntimeError(
                "MLX engine requires Darwin-arm64 with `pip install 'hfl[mlx]'` installed."
            )
        from mlx_lm import load  # type: ignore

        start = time.perf_counter()
        self._model, self._tokenizer = load(model_path)
        self._model_path = model_path
        logger.info("MLX model loaded from %s in %.2fs", model_path, time.perf_counter() - start)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_path = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_name(self) -> str:
        """Return the loaded model path (used for diagnostics)."""
        return self._model_path or "mlx-engine"

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Render messages via the tokenizer's chat template.

        mlx-lm's tokenizers wrap HF's ``AutoTokenizer``, which has
        ``apply_chat_template``. Fall back to a role-tagged
        concatenation when the tokenizer lacks a template.
        """
        if self._tokenizer is None:
            raise RuntimeError("tokenizer not loaded")
        dicts = [
            {
                "role": m.role,
                "content": m.content,
            }
            for m in messages
        ]
        apply = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return apply(dicts, tokenize=False, add_generation_prompt=True)
            except Exception:
                logger.debug("apply_chat_template failed; falling back to manual", exc_info=True)
        parts: list[str] = []
        for m in messages:
            parts.append(f"{m.role}: {m.content}")
        parts.append("assistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_sampling(self, cfg: GenerationConfig) -> dict[str, Any]:
        """Translate HFL's GenerationConfig to mlx-lm 0.31+ kwargs.

        mlx-lm 0.30+ moved sampling parameters off the top-level
        ``generate()`` signature and onto a ``sampler`` callable plus a
        list of ``logits_processors``. We build both here and return
        them as a kwargs dict ready to splat into ``generate`` /
        ``stream_generate``.
        """
        from mlx_lm.sample_utils import (  # type: ignore
            make_logits_processors,
            make_sampler,
        )

        sampler = make_sampler(
            temp=cfg.temperature,
            top_p=cfg.top_p if cfg.top_p else 0.0,
            top_k=cfg.top_k if cfg.top_k else 0,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=cfg.repeat_penalty if cfg.repeat_penalty != 1.0 else None,
        )
        return {
            "max_tokens": cfg.max_tokens,
            "sampler": sampler,
            "logits_processors": logits_processors,
        }

    def _run_generate(self, prompt: str, cfg: GenerationConfig) -> tuple[str, int, int, int]:
        from mlx_lm import generate  # type: ignore

        start_ns = time.monotonic_ns()
        kwargs = self._build_sampling(cfg)
        try:
            text = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                **kwargs,
            )
        except Exception:
            logger.exception("MLX generate failed")
            raise
        total_ns = time.monotonic_ns() - start_ns
        # mlx-lm returns only the completion text; we still need
        # token counts for the response envelope. Compute via the
        # tokenizer (cheap: ≤ a few k tokens per request).
        try:
            n_prompt = len(self._tokenizer.encode(prompt))
        except Exception:
            n_prompt = 0
        try:
            n_gen = len(self._tokenizer.encode(text)) if text else 0
        except Exception:
            n_gen = 0
        return text, n_prompt, n_gen, total_ns

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig()
        if not self.is_loaded:
            raise RuntimeError("MLX engine is not loaded")
        text, n_prompt, n_gen, total_ns = self._run_generate(prompt, cfg)
        elapsed = max(total_ns, 1) / 1e9
        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=n_prompt,
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
            stop_reason="stop",
            total_duration=total_ns,
            load_duration=0,
            prompt_eval_duration=int(total_ns * n_prompt / max(1, n_prompt + n_gen)),
            eval_duration=int(total_ns * n_gen / max(1, n_prompt + n_gen)),
        )

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        **_kwargs: Any,
    ) -> GenerationResult:
        prompt = self._messages_to_prompt(messages)
        return self.generate(prompt, config)

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Token-by-token streaming via mlx-lm's ``stream_generate``."""
        cfg = config or GenerationConfig()
        if not self.is_loaded:
            raise RuntimeError("MLX engine is not loaded")
        from mlx_lm import stream_generate  # type: ignore

        kwargs = self._build_sampling(cfg)
        for token in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            **kwargs,
        ):
            if hasattr(token, "text"):
                yield token.text  # newer mlx-lm versions wrap in a GenStep
            else:
                yield str(token)

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        **_kwargs: Any,
    ) -> Iterator[str]:
        prompt = self._messages_to_prompt(messages)
        yield from self.generate_stream(prompt, config)

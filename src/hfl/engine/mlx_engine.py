# SPDX-License-Identifier: Apache-2.0
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
from typing import Any, Generator, Iterator, cast

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
        import mlx_lm  # noqa: F401
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
        # KV of recent prompts (``mlx_lm``'s LRUPromptCache), or None when
        # disabled or when the installed mlx-lm predates it. None means the
        # engine runs exactly the pre-cache code paths.
        self._prompt_store: Any = None
        #: Prompt tokens the last request took from the cache instead of
        #: evaluating. Diagnostic only — the proof lives in the timings.
        self.last_prompt_tokens_reused: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, model_path: str, **kwargs: Any) -> None:
        if not is_available():
            raise RuntimeError(
                "MLX engine requires Darwin-arm64 with `pip install 'hfl[mlx]'` installed."
            )
        from mlx_lm import load

        start = time.perf_counter()
        # ``load`` returns ``(model, tokenizer)`` by default and
        # ``(model, tokenizer, config)`` when ``return_config=True``; the
        # starred target accepts either arity (we only want the first two).
        self._model, self._tokenizer, *_ = load(model_path)
        self._model_path = model_path
        self._prompt_store = self._new_prompt_store()
        logger.info("MLX model loaded from %s in %.2fs", model_path, time.perf_counter() - start)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_path = None
        # The cached KV belongs to the model being dropped; keeping it would
        # pin memory and could never be matched against another model.
        self._prompt_store = None
        self._release_memory()

    @staticmethod
    def _release_memory() -> None:
        """Hand the dropped model's memory back to the system.

        Dropping the references is not enough on MLX: freed Metal buffers go
        to MLX's own cache for reuse, so the process keeps them. Measured on
        a 17 GB model: resident memory stayed at 16.7 GB after ``unload``.
        With several models resident that made eviction a no-op — unloading
        one to make room for another freed nothing. ``clear_cache`` only
        returns buffers nothing uses, so a model still loaded elsewhere in
        the process keeps its weights.
        """
        import gc

        gc.collect()
        try:
            import mlx.core as mx
        except ImportError:
            return
        clear = getattr(mx, "clear_cache", None)
        if clear is None:  # mlx < 0.22 kept it under mx.metal
            clear = getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear is not None:
            try:
                clear()
            except Exception:  # pragma: no cover - backend-specific failure
                logger.debug("mlx clear_cache failed", exc_info=True)

    @staticmethod
    def _new_prompt_store() -> Any:
        """A bounded prompt cache, or None to run without one.

        llama.cpp reuses the KV prefix between turns inside
        ``Llama.generate``; ``mlx_lm.generate`` does not, so every chat turn
        on the MLX backend re-evaluated the whole conversation — measured on
        Qwen2.5-0.5B-4bit at ~185 ms per turn for a ~1 550-token prompt, the
        same cost on turn 5 as on turn 2. ``mlx_lm`` ships the fix as
        ``LRUPromptCache``, the structure its own server uses.

        Bounded in bytes because on Apple Silicon the cache shares unified
        memory with the weights. ``HFL_MLX_PROMPT_CACHE_BYTES=0`` disables
        it, and disabled means the pre-cache code paths run unchanged — the
        off-switch is a real rollback, not an approximation of one.
        """
        from hfl.config import config

        budget = int(getattr(config, "mlx_prompt_cache_bytes", 0) or 0)
        if budget <= 0:
            return None
        try:
            from mlx_lm.models.cache import LRUPromptCache
        except ImportError:
            logger.info("mlx-lm has no LRUPromptCache; running without a prompt cache")
            return None
        return LRUPromptCache(max_size=4, max_bytes=budget)

    def _cached_responses(self, prompt: str, cfg: GenerationConfig) -> Generator[Any, None, None]:
        """Stream ``mlx_lm`` responses, reusing and then refreshing the cache.

        Mirrors ``mlx_lm.server``: fetch the cached KV nearest to this
        prompt, evaluate only the tokens it does not cover, then store the
        cache under prompt + generated token ids.

        The key is built from the ids the model actually produced, never by
        re-tokenising the text. Detokenise-then-encode is not guaranteed to
        round-trip, and a key that disagrees with the KV it names would let
        a later request reuse attention state computed for different tokens
        — silently wrong output. A response without a token id therefore
        disables the store for that request instead of guessing.

        A consumer that stops early (a stop string, a closed stream) still
        leaves a consistent cache: every token ``mlx_lm`` yields has already
        been fed to the model. A request that fails is not stored.
        """
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        store = self._prompt_store
        model_key = self._model_path or "mlx-engine"
        tokens = [int(t) for t in self._tokenizer.encode(prompt)]

        cache, rest = store.fetch_nearest_cache(model_key, tokens)
        if cache is None or not rest:
            # No reusable prefix — or an exact hit, which would leave nothing
            # to evaluate. The latter needs a prompt equal to an earlier
            # prompt PLUS its whole reply, so a fresh cache costs little.
            cache, rest = make_prompt_cache(self._model), tokens
        self.last_prompt_tokens_reused = len(tokens) - len(rest)

        self._maybe_seed(cfg)
        key: list[int] | None = list(tokens)
        store_it = False
        try:
            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt=rest,
                prompt_cache=cache,
                **self._build_sampling(cfg),
            ):
                token = getattr(response, "token", None)
                if token is None:
                    key = None
                elif key is not None:
                    key.append(int(token))
                yield response
            store_it = True
        except GeneratorExit:
            store_it = True
            raise
        finally:
            if store_it and key is not None:
                store.insert_cache(model_key, key, cache)

    @staticmethod
    def _measured_ns(last: Any) -> tuple[int, int] | None:
        """Prefill and generation time as mlx-lm measured them, in ns.

        The pre-cache path apportions total time by token count, which is
        the defect 0.18.2 removed from llama.cpp: it cannot show a cache hit,
        because it charges every prompt token the same. ``prompt_tps``
        covers only the tokens actually evaluated, so a reused prefix shows
        up as the saving it is.
        """
        p_tok = getattr(last, "prompt_tokens", 0) or 0
        p_tps = getattr(last, "prompt_tps", 0) or 0
        g_tok = getattr(last, "generation_tokens", 0) or 0
        g_tps = getattr(last, "generation_tps", 0) or 0
        if p_tps <= 0 and g_tps <= 0:
            return None
        prompt_ns = int(p_tok / p_tps * 1e9) if p_tps > 0 else 0
        eval_ns = int(g_tok / g_tps * 1e9) if g_tps > 0 else 0
        return prompt_ns, eval_ns

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

    def _messages_to_prompt(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        reasoning: str | None = None,
    ) -> str:
        """Render messages via the tokenizer's chat template.

        mlx-lm's tokenizers wrap HF's ``AutoTokenizer``, which has
        ``apply_chat_template``. Fall back to a role-tagged
        concatenation when the tokenizer lacks a template.

        Tools go to the template when it lists them, else they are written
        in as the GGUF engines do; past calls and tool results are kept.
        (They were all dropped: on MLX a model never saw its tools.) The
        reasoning switch (``think`` & co.) reaches the template too.
        """
        from hfl.engine.llama_cpp import (
            _history_for_template,
            _template_renders_tools,
            _tools_as_text,
        )

        if self._tokenizer is None:
            raise RuntimeError("tokenizer not loaded")
        dicts: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            if m.name:
                entry["name"] = m.name
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            dicts.append(entry)
        template = getattr(self._tokenizer, "chat_template", None)
        template = template if isinstance(template, str) else ""
        extras: dict[str, Any] = reasoning_template_vars(reasoning)
        if _template_renders_tools(template, None):
            dicts = _history_for_template(dicts, template)
            if tools:
                extras["tools"] = tools
        else:
            dicts = _tools_as_text(dicts, tools)
        apply = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return cast(str, apply(dicts, tokenize=False, add_generation_prompt=True, **extras))
            except TypeError:
                # An old tokenizer without extra template variables.
                try:
                    return cast(str, apply(dicts, tokenize=False, add_generation_prompt=True))
                except Exception:
                    logger.debug("apply_chat_template failed; manual", exc_info=True)
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
        from mlx_lm.sample_utils import (
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

    @staticmethod
    def _maybe_seed(cfg: GenerationConfig) -> None:
        """Seed mlx's global RNG for reproducible output when a concrete seed
        is requested (parity with the vLLM / diffusers backends — ENG-10/11).
        mlx-lm has no per-call seed argument, so we seed the global RNG."""
        if cfg.seed is not None and cfg.seed >= 0:
            try:
                import mlx.core as mx

                mx.random.seed(int(cfg.seed))
            except Exception:  # pragma: no cover - mlx optional / API drift
                logger.debug("mlx seed failed", exc_info=True)

    @staticmethod
    def _stop_strings(cfg: GenerationConfig) -> list[str]:
        """Normalise ``cfg.stop`` (str | list | None) to a list of non-empty
        stop strings. mlx-lm has no native stop-string support, so the engine
        enforces them by truncating its own output."""
        raw = cfg.stop
        if not raw:
            return []
        seq = [raw] if isinstance(raw, str) else list(raw)
        return [s for s in seq if isinstance(s, str) and s]

    @staticmethod
    def _earliest_stop(text: str, stops: list[str]) -> int | None:
        """Index of the earliest occurrence of any stop string, or None."""
        best: int | None = None
        for s in stops:
            i = text.find(s)
            if i != -1 and (best is None or i < best):
                best = i
        return best

    def _run_generate(self, prompt: str, cfg: GenerationConfig) -> tuple[str, int, int, int]:
        from mlx_lm import generate

        start_ns = time.monotonic_ns()
        self._maybe_seed(cfg)
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
        # Enforce the request's stop sequences (mlx-lm ignores them): truncate
        # at the first occurrence so output never runs past a caller stop string.
        stops = self._stop_strings(cfg)
        if stops:
            cut = self._earliest_stop(text, stops)
            if cut is not None:
                text = text[:cut]
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
        if self._prompt_store is not None:
            return self._generate_cached(prompt, cfg)
        text, n_prompt, n_gen, total_ns = self._run_generate(prompt, cfg)
        elapsed = max(total_ns, 1) / 1e9
        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=n_prompt,
            tokens_per_second=n_gen / elapsed if elapsed > 0 else 0,
            # mlx_lm.generate returns only text; n_gen re-tokenises it, so
            # reaching max_tokens is the best available sign it was cut.
            stop_reason="length" if cfg.max_tokens and n_gen >= cfg.max_tokens else "stop",
            total_duration=total_ns,
            load_duration=0,
            prompt_eval_duration=int(total_ns * n_prompt / max(1, n_prompt + n_gen)),
            eval_duration=int(total_ns * n_gen / max(1, n_prompt + n_gen)),
        )

    def _generate_cached(self, prompt: str, cfg: GenerationConfig) -> GenerationResult:
        """``generate`` through the prompt cache, with measured timings."""
        start_ns = time.monotonic_ns()
        stops = self._stop_strings(cfg)
        text = ""
        last: Any = None
        n_gen = 0
        responses = self._cached_responses(prompt, cfg)
        try:
            for response in responses:
                last = response
                n_gen = getattr(response, "generation_tokens", n_gen + 1) or n_gen + 1
                text += response.text if hasattr(response, "text") else str(response)
                if stops and self._earliest_stop(text, stops) is not None:
                    # Stop generating, not only stop showing: every token
                    # past the stop string was compute nobody reads.
                    break
        except Exception:
            logger.exception("MLX generate failed")
            raise
        finally:
            responses.close()
        if stops:
            cut = self._earliest_stop(text, stops)
            if cut is not None:
                text = text[:cut]
        total_ns = time.monotonic_ns() - start_ns
        n_prompt = self.last_prompt_tokens_reused + int(getattr(last, "prompt_tokens", 0) or 0)
        measured = self._measured_ns(last)
        if measured is None:
            prompt_ns, eval_ns = 0, total_ns
        else:
            prompt_ns, eval_ns = measured
        eval_s = eval_ns / 1e9
        finish = getattr(last, "finish_reason", None)
        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            tokens_prompt=n_prompt,
            tokens_per_second=n_gen / eval_s if eval_s > 0 else 0,
            stop_reason="length" if finish == "length" else "stop",
            total_duration=total_ns,
            load_duration=0,
            prompt_eval_duration=prompt_ns,
            eval_duration=eval_ns,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
        **_kwargs: Any,
    ) -> GenerationResult:
        cfg = self._chat_config(config, messages, tools)
        prompt = self._messages_to_prompt(messages, tools, cfg.reasoning)
        return self.generate(prompt, cfg)

    @staticmethod
    def _chat_config(
        config: GenerationConfig | None, messages: list[ChatMessage], tools: list[dict] | None
    ) -> GenerationConfig:
        """The config for a chat turn: no repetition penalty on a tool turn
        unless the client chose one (``repeat_penalty_for``)."""
        import dataclasses

        cfg = config or GenerationConfig()
        return dataclasses.replace(cfg, repeat_penalty=repeat_penalty_for(cfg, messages, tools))

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> CountedStream:
        """Token-by-token streaming via mlx-lm's ``stream_generate``.

        The stream knows its token counts once read: mlx-lm reports them on
        every response, so the streamed reply's ``prompt_eval_count`` /
        ``eval_count`` (and OpenAI's ``usage``) are measured, not missing.
        """
        cfg = config or GenerationConfig()
        if not self.is_loaded:
            raise RuntimeError("MLX engine is not loaded")
        stops = self._stop_strings(cfg)
        counted = CountedStream()
        last: list[Any] = [None]

        def _piece(token: Any) -> str:
            last[0] = token
            return token.text if hasattr(token, "text") else str(token)

        cached = self._prompt_store is not None
        gen: Iterator[Any]
        if cached:
            gen = self._cached_responses(prompt, cfg)
        else:
            from mlx_lm import stream_generate

            self._maybe_seed(cfg)
            kwargs = self._build_sampling(cfg)
            gen = stream_generate(self._model, self._tokenizer, prompt=prompt, **kwargs)

        def _stream() -> Iterator[str]:
            try:
                yield from self._stream_until_stop(gen, stops, _piece)
            finally:
                # Close now, on the thread holding the dispatcher slot: that
                # is when the cached path stores this turn's KV. Left to the
                # garbage collector it could land later, on another thread,
                # while the next request is already using the store.
                close = getattr(gen, "close", None)
                if close is not None:
                    close()
                response = last[0]
                if response is not None and hasattr(response, "generation_tokens"):
                    reused = self.last_prompt_tokens_reused if cached else 0
                    counted.prompt_tokens = reused + int(response.prompt_tokens or 0)
                    counted.completion_tokens = int(response.generation_tokens or 0)

        return counted.feed(_stream())

    def _stream_until_stop(
        self, gen: Iterator[Any], stops: list[str], _piece: Any
    ) -> Iterator[str]:
        if not stops:
            for token in gen:
                yield _piece(token)
            return

        # Enforce stop strings (mlx-lm has none): accumulate, hold back a tail
        # that could be the start of a stop string, and halt at the boundary so
        # output never includes text at or past a caller stop string.
        max_stop = max(len(s) for s in stops)
        acc = ""
        emitted = 0
        for token in gen:
            acc += _piece(token)
            cut = self._earliest_stop(acc, stops)
            if cut is not None:
                if cut > emitted:
                    yield acc[emitted:cut]
                return
            safe = len(acc) - max_stop + 1
            if safe > emitted:
                yield acc[emitted:safe]
                emitted = safe
        if len(acc) > emitted:
            yield acc[emitted:]

    def chat_stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[dict] | None = None,
        **_kwargs: Any,
    ) -> Iterator[str]:
        cfg = self._chat_config(config, messages, tools)
        prompt = self._messages_to_prompt(messages, tools, cfg.reasoning)
        return self.generate_stream(prompt, cfg)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Several requests at once on one MLX model: mlx-lm's ``BatchGenerator``.

The MLX engine served one request at a time; a second client waited for the
first to finish. mlx-lm batches them itself (continuous batching: requests
join and leave a running batch token by token), so this module only feeds
it. Measured on an M3 Max (02-10-2026, 128-token replies): Qwen3-14B in
BF16 went from 11.4 to 46 tokens/s with 8 requests (4.0x), the same with
one, for 0.5 GB more memory. Greedy output is the same as one at a time,
and sampling, a seed, a stop at EOS, a response format, logprobs and the
prompt cache all work per request.

One thread owns the ``BatchGenerator`` and makes every MLX call — adding,
stepping, removing — as mlx-lm's own server does; callers talk to it
through a command queue and read their tokens from a queue of their own.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["BatchScheduler", "Step", "batchable", "configured_slots"]


def configured_slots() -> int:
    """How many requests run at once: ``HFL_NUM_PARALLEL`` when the operator
    set it (or ``hfl serve --parallel``), else llama-server's default."""
    from hfl.config import config
    from hfl.engine.llama_server import DEFAULT_SLOTS

    configured = max(1, int(getattr(config, "queue_max_inflight", 1) or 1))
    return configured if getattr(config, "parallel_explicit", False) else DEFAULT_SLOTS


def batchable(model: Any) -> bool:
    """Whether every cache layer of ``model`` can join a batch (mlx-lm's own
    test: each must be mergeable)."""
    try:
        from mlx_lm.generate import BatchGenerator  # noqa: F401
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError:  # an mlx-lm without batching
        return False
    try:
        return all(hasattr(layer, "merge") for layer in make_prompt_cache(model))
    except Exception:
        logger.debug("could not build a prompt cache to test batching", exc_info=True)
        return False


def _float32(logprobs: Any) -> Any:
    """``logprobs`` as float32, evaluated on the batch thread (MLX arrays are
    lazy: evaluating on the request's thread would use another stream)."""
    astype = getattr(logprobs, "astype", None)
    if astype is None:
        return logprobs
    import mlx.core as mx

    converted = astype(mx.float32)
    mx.eval(converted)
    return converted


@dataclass
class Step:
    """One generated token, shaped like mlx-lm's ``GenerationResponse`` so the
    engine reads it as it reads a one-at-a-time stream."""

    text: str
    token: int
    logprobs: Any
    finish_reason: str | None
    prompt_tokens: int  # evaluated (not taken from the cache)
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    reused: int = 0  # prompt tokens the prompt cache served
    from_draft: bool = False


@dataclass
class _Entry:
    out: queue.Queue[Step | BaseException | None]
    detokenizer: Any
    submitted: float
    evaluated: int
    reused: int
    wants_logprobs: bool = False
    first_token_at: float = 0.0
    generated: int = 0
    finished: bool = False


@dataclass
class _Insert:
    tokens: list[int]
    max_tokens: int
    sampler: Callable[..., Any]
    processors: list[Callable[..., Any]]
    wants_logprobs: bool
    reply: queue.Queue[tuple[int, queue.Queue[Any]] | BaseException] = field(
        default_factory=lambda: queue.Queue(maxsize=1)
    )


class BatchScheduler:
    """Owns a ``BatchGenerator`` on a thread of its own."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        slots: int,
        store: Any = None,
        model_key: str = "mlx-engine",
    ) -> None:
        self._model, self._tokenizer, self._slots = model, tokenizer, slots
        self._store, self._model_key = store, model_key
        self._eos = {int(t) for t in (getattr(tokenizer, "eos_token_ids", None) or [])}
        self._commands: queue.Queue[Any] = queue.Queue()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hfl-mlx-batch", daemon=True)
        self._thread.start()

    # -- called from request threads -------------------------------------

    def stream(
        self,
        tokens: list[int],
        *,
        max_tokens: int,
        sampler: Callable[..., Any],
        processors: list[Callable[..., Any]],
        cancelled: Callable[[], bool] = lambda: False,
        logprobs: bool = False,
    ) -> Iterator[Step]:
        """The tokens of one request, as the batch produces them. Closing the
        iterator early (a stop string, a dropped client) or ``cancelled()``
        turning True removes the request from the batch."""
        if self._stopped.is_set():
            raise RuntimeError("the MLX batch scheduler is closed")
        command = _Insert(list(tokens), max_tokens, sampler, list(processors), logprobs)
        self._commands.put(command)
        reply = command.reply.get()
        if isinstance(reply, BaseException):
            raise reply
        uid, out = reply
        finished = False
        try:
            while True:
                try:
                    item = out.get(timeout=0.25)
                except queue.Empty:
                    if cancelled():
                        return
                    continue
                if item is None:
                    finished = True
                    return
                if isinstance(item, BaseException):
                    finished = True
                    raise item
                yield item
                if cancelled():
                    return
        finally:
            if not finished:
                self._commands.put(("remove", uid))

    def close(self) -> None:
        """Stop the thread; requests still running get an error."""
        if self._stopped.is_set():
            return
        self._commands.put(("stop",))
        self._thread.join(timeout=30)

    # -- the batch thread ---------------------------------------------------

    def _new_generator(self) -> Any:
        from mlx_lm.generate import BatchGenerator

        return BatchGenerator(
            self._model,
            stop_tokens=[[t] for t in sorted(self._eos)] or None,
            completion_batch_size=self._slots,
            prefill_batch_size=self._slots,
        )

    def _run(self) -> None:
        generator = self._new_generator()
        active: dict[int, _Entry] = {}
        try:
            while True:
                # Wait for work when idle; otherwise take what arrived and step.
                try:
                    command = self._commands.get(timeout=None if not active else 0)
                except queue.Empty:
                    command = None
                while command is not None:
                    if command == ("stop",):
                        return
                    self._apply(generator, command, active)
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        command = None
                if active:
                    try:
                        self._step(generator, active)
                    except Exception as exc:  # the batch is lost: say so to each
                        logger.exception("MLX batch step failed")
                        for entry in active.values():
                            entry.out.put(exc)
                        active.clear()
                        generator.close()
                        generator = self._new_generator()
        finally:
            self._stopped.set()
            for entry in active.values():
                entry.out.put(RuntimeError("the MLX model was unloaded"))
            try:
                generator.close()
            except Exception:
                logger.debug("closing the batch generator failed", exc_info=True)
            # Requests that arrived after the stop must not wait forever.
            while True:
                try:
                    late = self._commands.get_nowait()
                except queue.Empty:
                    break
                if isinstance(late, _Insert):
                    late.reply.put(RuntimeError("the MLX model was unloaded"))

    def _apply(self, generator: Any, command: Any, active: dict[int, _Entry]) -> None:
        if isinstance(command, _Insert):
            try:
                cache, rest = None, command.tokens
                if self._store is not None:
                    found, left = self._store.fetch_nearest_cache(self._model_key, command.tokens)
                    if found is not None and left:  # an exact hit leaves nothing to evaluate
                        cache, rest = found, left
                reused = len(command.tokens) - len(rest)
                (uid,) = generator.insert(
                    [rest],
                    max_tokens=[command.max_tokens],
                    caches=[cache] if cache is not None else None,
                    all_tokens=[command.tokens[:reused]] if cache is not None else None,
                    samplers=[command.sampler],
                    logits_processors=[command.processors],
                )
            except Exception as exc:
                command.reply.put(exc)
                return
            out: queue.Queue[Any] = queue.Queue()
            # ``detokenizer`` is a new one on each access: one per request.
            active[uid] = _Entry(
                out,
                self._tokenizer.detokenizer,
                time.perf_counter(),
                len(rest),
                reused,
                command.wants_logprobs,
            )
            command.reply.put((uid, out))
        elif isinstance(command, tuple) and command[0] == "remove":
            uid = command[1]
            if active.pop(uid, None) is not None:
                generator.remove([uid])

    def _step(self, generator: Any, active: dict[int, _Entry]) -> None:
        _prompts, responses = generator.next()
        now = time.perf_counter()
        for r in responses:
            entry = active.get(r.uid)
            if entry is None:  # removed meanwhile
                continue
            if not entry.first_token_at:
                entry.first_token_at = now
            token = int(r.token)
            if token not in self._eos:
                entry.detokenizer.add_token(token)
                entry.generated += 1
            if r.finish_reason is not None:
                entry.detokenizer.finalize()
            # The distribution only for the requests that read it, and in
            # float32: a BF16 model's could not reach numpy ("Item size 2 for
            # PEP 3118 buffer format string B"), measured on Qwen3-14B.
            logprobs = _float32(r.logprobs) if entry.wants_logprobs else None
            prefill = max(entry.first_token_at - entry.submitted, 1e-9)
            decoding = max(now - entry.first_token_at, 1e-9)
            entry.out.put(
                Step(
                    text=entry.detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    finish_reason=r.finish_reason,
                    prompt_tokens=entry.evaluated,
                    generation_tokens=entry.generated,
                    prompt_tps=entry.evaluated / prefill,
                    generation_tps=entry.generated / decoding if entry.generated > 1 else 0.0,
                    reused=entry.reused,
                )
            )
            if r.finish_reason is not None:
                if self._store is not None and getattr(r, "prompt_cache", None) is not None:
                    key = list(r.all_tokens)
                    try:
                        self._store.insert_cache(self._model_key, key, r.prompt_cache)
                    except Exception:
                        logger.debug("prompt cache insert failed", exc_info=True)
                entry.out.put(None)
                del active[r.uid]

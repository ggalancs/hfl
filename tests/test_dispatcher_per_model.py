# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A queue per llama.cpp model: two models no longer wait for each other.

Every in-process engine shared one global queue of one slot, so a request
to one model waited for another model's generation (measured: two models
at once, 1.0x). llama.cpp instances are independent (two generating at once
on Metal: same output as alone, 1.56x); each gets its own one-slot queue.
Engines not measured that way (MLX, Transformers) keep the global queue.
Health and metrics add every queue up."""

from __future__ import annotations

import asyncio

from hfl.core import dispatcher_for, dispatcher_totals, get_dispatcher


class _Independent:
    independent_instances = True
    supports_concurrent_inference = False


class _Shared:
    independent_instances = False
    supports_concurrent_inference = False


def test_each_llama_cpp_model_gets_its_own_one_slot_queue() -> None:
    a, b = _Independent(), _Independent()
    qa, qb = dispatcher_for(a), dispatcher_for(b)
    assert qa is not qb and qa is not get_dispatcher()
    assert qa.snapshot().max_inflight == 1  # still one call at a time per model
    assert dispatcher_for(a) is qa  # the same queue every time


def test_an_engine_not_known_to_be_independent_keeps_the_global_queue() -> None:
    assert dispatcher_for(_Shared()) is get_dispatcher()


def test_the_real_llama_cpp_engine_declares_it() -> None:
    from hfl.engine.llama_cpp import LlamaCppEngine
    from hfl.engine.mlx_engine import MLXEngine

    assert LlamaCppEngine.__new__(LlamaCppEngine).independent_instances is True
    assert MLXEngine.__new__(MLXEngine).independent_instances is False


def test_two_models_run_at_the_same_time() -> None:
    """Two requests to two models overlap; to one model they do not."""

    async def run(first, second) -> float:
        async def hold(engine):
            async with dispatcher_for(engine).slot():
                await asyncio.sleep(0.2)

        started = asyncio.get_running_loop().time()
        await asyncio.gather(hold(first), hold(second))
        return asyncio.get_running_loop().time() - started

    a, b = _Independent(), _Independent()
    assert asyncio.run(run(a, b)) < 0.35  # overlapped
    assert asyncio.run(run(a, a)) >= 0.39  # one after the other


def test_totals_add_every_queue() -> None:
    before = dispatcher_totals()
    engine = _Independent()  # alive, as a resident model is
    dispatcher_for(engine)
    assert dispatcher_totals().max_inflight == before.max_inflight + 1


def test_an_unloaded_model_s_queue_leaves_the_totals() -> None:
    """Weakly held: a model's queue goes with the model."""
    import gc

    before = dispatcher_totals()
    engine = _Independent()
    dispatcher_for(engine)
    del engine
    gc.collect()
    assert dispatcher_totals().max_inflight == before.max_inflight

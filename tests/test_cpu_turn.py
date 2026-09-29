# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Models that each generate on every CPU core take turns.

Two llama-server processes on a 4-core CPU, each with a thread per core,
went from 154 tok/s in turn to 30 tok/s at once: they fight over the cores.
On a CPU-only machine their requests take turns across models, while the
requests to one model still share its parallel slots. With a GPU nothing
changes."""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from hfl.core import dispatcher_for
from hfl.engine.dispatcher import CpuTurn, QueueTimeoutError


class _CpuServer:
    supports_concurrent_inference = True
    independent_instances = False
    generates_on_all_cpu_cores = True
    parallel_slots = 4


class _GpuServer(_CpuServer):
    generates_on_all_cpu_cores = False


async def _hold(engine, log: list, name: str, seconds: float = 0.1) -> None:
    async with dispatcher_for(engine).slot():
        log.append(("in", name))
        await asyncio.sleep(seconds)
        log.append(("out", name))


def _overlap(log: list, a: str, b: str) -> bool:
    """Whether a request named ``a`` and one named ``b`` were ever in at once."""
    inside: dict[str, int] = {}
    for step, name in log:
        inside[name] = inside.get(name, 0) + (1 if step == "in" else -1)
        if inside.get(a, 0) > 0 and inside.get(b, 0) > 0:
            return True
    return False


def test_two_cpu_bound_models_take_turns() -> None:
    a, b, log = _CpuServer(), _CpuServer(), []

    async def run() -> None:
        await asyncio.gather(_hold(a, log, "a"), _hold(b, log, "b"))

    asyncio.run(run())
    assert not _overlap(log, "a", "b")


def test_one_cpu_bound_model_still_uses_its_parallel_slots() -> None:
    a, log = _CpuServer(), []

    async def run() -> float:
        started = asyncio.get_running_loop().time()
        await asyncio.gather(*(_hold(a, log, f"a{i}", 0.2) for i in range(4)))
        return asyncio.get_running_loop().time() - started

    assert asyncio.run(run()) < 0.4  # 4 x 0.2 s at once, not 0.8 s


def test_models_with_a_gpu_do_not_take_turns() -> None:
    a, b, log = _GpuServer(), _GpuServer(), []

    async def run() -> None:
        await asyncio.gather(_hold(a, log, "a"), _hold(b, log, "b"))

    asyncio.run(run())
    assert _overlap(log, "a", "b")


def test_a_waiting_model_is_not_starved() -> None:
    """While b waits, new requests to a queue behind it instead of joining a."""

    async def run() -> list:
        turn, order = CpuTurn(), []
        a, b = object(), object()
        await turn.enter(a, 1)  # a holds the turn

        async def take(key, name):
            await turn.enter(key, 1)
            order.append(name)
            await asyncio.sleep(0.01)
            turn.leave()

        waiting_b = asyncio.create_task(take(b, "b"))
        await asyncio.sleep(0)
        later_a = asyncio.create_task(take(a, "a2"))
        await asyncio.sleep(0.05)
        assert order == []  # a2 did not slip in while a still held it
        turn.leave()
        await asyncio.gather(waiting_b, later_a)
        return order

    assert asyncio.run(run()) == ["b", "a2"]


def test_a_request_that_waits_too_long_gets_a_timeout_and_frees_everything() -> None:
    async def run() -> None:
        a, b = _CpuServer(), _CpuServer()
        qa, qb = dispatcher_for(a), dispatcher_for(b)
        qb._acquire_timeout = 0.05
        async with qa.slot():
            with pytest.raises(QueueTimeoutError):
                async with qb.slot():
                    pass
            snap = qb.snapshot()
            assert (snap.in_flight, snap.rejected_timeout_total) == (0, 1)
        # a left: b gets in, and its queue is whole again.
        qb._acquire_timeout = 1
        async with qb.slot():
            assert qb.snapshot().in_flight == 1

    asyncio.run(run())


def test_a_cancelled_waiter_leaves_the_turn_usable() -> None:
    async def run() -> None:
        turn, a, b, c = CpuTurn(), object(), object(), object()
        await turn.enter(a, 1)
        waiter = asyncio.create_task(turn.enter(b, 10))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        turn.leave()
        assert turn.owner is None
        await asyncio.wait_for(turn.enter(c, 1), 0.5)
        assert turn.owner is c

    asyncio.run(run())


@pytest.mark.parametrize(
    ("listing", "gpu"),
    [
        ("Available devices:\n  (none)\n", False),
        ("Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n", False),
        (
            "Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n"
            "  MTL0: Apple M3 Max (110100 MiB, 110100 MiB free)\n",
            True,
        ),
        ("Available devices:\n  CUDA0: NVIDIA L4 (22478 MiB, 22000 MiB free)\n", True),
        ("error: invalid argument: --list-devices\n", True),  # unknown: as before
    ],
)
def test_llama_server_reads_its_devices(monkeypatch, listing: str, gpu: bool) -> None:
    from hfl.engine import llama_server

    def fake_run(argv, **kwargs):
        assert argv[1:] == ["--list-devices"]
        return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr="")

    monkeypatch.setattr(llama_server.subprocess, "run", fake_run)
    llama_server.has_gpu.cache_clear()
    try:
        assert llama_server.has_gpu("/fake/llama-server") is gpu
    finally:
        llama_server.has_gpu.cache_clear()

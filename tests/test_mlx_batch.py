# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The MLX engine's batch scheduler, against a stand-in ``BatchGenerator``.

mlx-lm's ``BatchGenerator`` serves several requests at once on one model
(measured on an M3 Max: 4x the throughput with 8). The scheduler feeds it
from one thread; these pin its contract without MLX: requests interleave,
each ends on its own, a dropped or cancelled one leaves the batch, a failed
step reaches every request without killing the scheduler, and the prompt
cache is fetched and refreshed per request.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from dataclasses import dataclass, field

import pytest

from hfl.engine import mlx_batch

EOS = 0


@dataclass
class _Resp:
    uid: int
    token: int
    logprobs: object
    finish_reason: str | None
    prompt_cache: object = None
    all_tokens: list[int] = field(default_factory=list)


class _FakeGenerator:
    """One token per active request per step: 1, 2, 3… then EOS (or the
    length limit)."""

    instances: list[_FakeGenerator] = []

    def __init__(self, model, stop_tokens=None, completion_batch_size=4, prefill_batch_size=4):
        self.active: dict[int, dict] = {}
        self.removed: list[int] = []
        self.next_uid = 0
        self.fail_next = False
        self.inserted: list[dict] = []
        _FakeGenerator.instances.append(self)

    def insert(self, prompts, max_tokens, caches=None, all_tokens=None, samplers=None,
               logits_processors=None):  # fmt: skip
        uids = []
        for i, prompt in enumerate(prompts):
            uid, self.next_uid = self.next_uid, self.next_uid + 1
            self.active[uid] = {"n": 0, "max": max_tokens[i], "prompt": prompt, "eos_at": None}
            self.inserted.append({"uid": uid, "prompt": list(prompt), "caches": caches,
                                  "all_tokens": all_tokens})  # fmt: skip
            uids.append(uid)
        return uids

    def next(self):
        time.sleep(0.005)  # a step takes time, as on a GPU
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Metal went away")
        out = []
        for uid, s in list(self.active.items()):
            s["n"] += 1
            if s["eos_at"] is not None and s["n"] >= s["eos_at"]:
                out.append(_Resp(uid, EOS, None, "stop", "KV", [*s["prompt"], 1]))
                del self.active[uid]
            elif s["n"] >= s["max"]:
                out.append(_Resp(uid, s["n"], None, "length", "KV", [*s["prompt"], 1]))
                del self.active[uid]
            else:
                out.append(_Resp(uid, s["n"], None, None))
        return [], out

    def remove(self, uids):
        self.removed.extend(uids)
        for uid in uids:
            self.active.pop(uid, None)

    def close(self):
        pass


class _Detok:
    def __init__(self):
        self.tokens: list[int] = []
        self.last_segment = ""

    def add_token(self, token):
        self.tokens.append(token)
        self.last_segment = f"<{token}>"

    def finalize(self):
        self.last_segment = ""


class _Tokenizer:
    eos_token_ids = [EOS]

    @property
    def detokenizer(self):
        return _Detok()


class _Store:
    def __init__(self, hit: int = 0):
        self.hit = hit
        self.inserted: list[list[int]] = []

    def fetch_nearest_cache(self, key, tokens):
        if not self.hit:
            return None, tokens
        return "CACHE", tokens[self.hit :]

    def insert_cache(self, key, tokens, cache):
        self.inserted.append(tokens)


@pytest.fixture
def fake_mlx(monkeypatch):
    _FakeGenerator.instances.clear()
    module = types.ModuleType("mlx_lm.generate")
    module.BatchGenerator = _FakeGenerator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", module)


def _scheduler(store=None) -> mlx_batch.BatchScheduler:
    return mlx_batch.BatchScheduler(object(), _Tokenizer(), slots=4, store=store)


def _read(stream) -> list:
    return list(stream)


def test_requests_interleave_and_each_ends_on_its_own(fake_mlx) -> None:
    scheduler = _scheduler()
    results: dict[str, list] = {}

    def run(name, n):
        results[name] = _read(scheduler.stream([5, 6], max_tokens=n, sampler=None, processors=[]))

    threads = [threading.Thread(target=run, args=(f"r{n}", n)) for n in (3, 6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    scheduler.close()
    assert [s.token for s in results["r3"]] == [1, 2, 3]
    assert [s.token for s in results["r6"]] == [1, 2, 3, 4, 5, 6]
    assert results["r6"][-1].finish_reason == "length"
    assert results["r6"][-1].generation_tokens == 6 and results["r6"][0].prompt_tokens == 2


def test_eos_ends_a_request_without_its_text(fake_mlx) -> None:
    scheduler = _scheduler()
    stream = scheduler.stream([5], max_tokens=10_000, sampler=None, processors=[])
    first = next(stream)
    state = _FakeGenerator.instances[0].active[0]
    state["eos_at"] = state["n"] + 3
    rest = list(stream)
    scheduler.close()
    assert first.text == "<1>" and rest[-1].finish_reason == "stop" and rest[-1].text == ""
    assert rest[-1].generation_tokens == len(rest)  # the EOS token is not counted


def test_closing_a_stream_early_removes_the_request(fake_mlx) -> None:
    scheduler = _scheduler()
    stream = scheduler.stream([5], max_tokens=10_000, sampler=None, processors=[])
    next(stream)
    stream.close()  # a stop string, a dropped client
    other = _read(scheduler.stream([5], max_tokens=2, sampler=None, processors=[]))
    scheduler.close()
    assert _FakeGenerator.instances[0].removed == [0] and len(other) == 2


def test_its_own_cancellation_stops_one_request_only(fake_mlx) -> None:
    scheduler = _scheduler()
    signal = threading.Event()
    seen: list = []

    def cancelled_one():
        for step in scheduler.stream(
            [5], max_tokens=1000, sampler=None, processors=[], cancelled=signal.is_set
        ):
            seen.append(step)
            if len(seen) == 3:
                signal.set()

    t = threading.Thread(target=cancelled_one)
    t.start()
    t.join(timeout=10)
    survivor = _read(scheduler.stream([5], max_tokens=4, sampler=None, processors=[]))
    scheduler.close()
    assert len(seen) == 3 and 0 in _FakeGenerator.instances[0].removed and len(survivor) == 4


def test_a_failed_step_reaches_the_request_and_the_scheduler_lives_on(fake_mlx) -> None:
    scheduler = _scheduler()
    stream = scheduler.stream([5], max_tokens=10_000, sampler=None, processors=[])
    next(stream)
    _FakeGenerator.instances[0].fail_next = True
    with pytest.raises(RuntimeError, match="Metal went away"):
        list(stream)
    after = _read(scheduler.stream([5], max_tokens=2, sampler=None, processors=[]))
    scheduler.close()
    assert len(after) == 2 and len(_FakeGenerator.instances) == 2  # a fresh generator


def test_unloading_ends_requests_in_flight_and_refuses_new_ones(fake_mlx) -> None:
    scheduler = _scheduler()
    stream = scheduler.stream([5], max_tokens=10_000, sampler=None, processors=[])
    next(stream)
    scheduler.close()
    with pytest.raises(RuntimeError, match="unloaded"):
        list(stream)
    with pytest.raises(RuntimeError, match="closed"):
        next(scheduler.stream([5], max_tokens=2, sampler=None, processors=[]))


def test_the_prompt_cache_is_fetched_and_refreshed_per_request(fake_mlx) -> None:
    store = _Store(hit=3)
    scheduler = _scheduler(store)
    steps = _read(scheduler.stream([1, 2, 3, 4, 5], max_tokens=2, sampler=None, processors=[]))
    scheduler.close()
    insert = _FakeGenerator.instances[0].inserted[0]
    assert insert["prompt"] == [4, 5] and insert["caches"] == ["CACHE"]
    assert insert["all_tokens"] == [[1, 2, 3]]
    assert steps[-1].reused == 3 and steps[-1].prompt_tokens == 2
    assert store.inserted == [[4, 5, 1]]


def test_the_engine_batching_declares_its_slots(monkeypatch) -> None:
    """With a scheduler the engine is concurrent and sized, the dispatcher
    gives it its own queue, and ``cancel()`` no longer stops everyone."""
    from hfl.core.container import dispatcher_for
    from hfl.engine.mlx_engine import MLXEngine

    engine = MLXEngine()
    assert engine.supports_concurrent_inference is False
    engine._batch, engine._slots = object(), 4
    assert engine.supports_concurrent_inference is True and engine.parallel_slots == 4
    assert dispatcher_for(engine).snapshot().max_inflight == 4
    engine.cancel()
    assert not engine._cancel.is_set()


def test_bf16_logprobs_reach_numpy() -> None:
    """A BF16 model's logprobs failed every logprobs request ("Item size 2 for
    PEP 3118 buffer format string B"), measured on Qwen3-14B."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    from hfl.engine.mlx_engine import _as_float32

    logprobs = mx.array([-0.5, -1.5, -3.0]).astype(mx.bfloat16)
    with pytest.raises(RuntimeError):
        np.array(logprobs, dtype=np.float64)  # what the engine used to do
    assert np.array(_as_float32(logprobs), dtype=np.float64).tolist() == [-0.5, -1.5, -3.0]

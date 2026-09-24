# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Several models resident at once, admitted by memory.

Until this existed HFL kept one model: a request for another unloaded the
current one. Now it keeps as many as ``HFL_MEMORY_BUDGET`` allows. These
tests drive the real ``ServerState`` and ``load_llm`` with fake engines and
a scripted memory reading, and pin the properties that make that safe:

* each request talks to the model it asked for, even when another request
  loaded a different model in between (the routes read ``state.engine``);
* a model a request is using is never unloaded under it — room is made
  from idle models, least recently used first, or by waiting;
* a load that cannot fit is refused before anything is unloaded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hfl.engine.base import GenerationResult
from hfl.engine.residency import MemoryView

GB = 1024**3


class FakeEngine:
    """Answers with its own name, so a response shows which model served it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_loaded = False
        self.unloaded = False
        self.context_size = 0

    def load(self, path: str, **kwargs) -> None:
        self.is_loaded = True
        self.context_size = int(kwargs.get("n_ctx") or 0)

    def unload(self) -> None:
        self.is_loaded = False
        self.unloaded = True

    def chat(self, messages, config=None, tools=None, **kw) -> GenerationResult:
        from hfl.api.state import get_state

        # How many leases this engine had while it was serving.
        self.leases_while_serving = get_state()._engine_inuse.get(id(self), 0)
        return GenerationResult(text=f"from {self.name}", tokens_generated=1)

    def generate(self, prompt, config=None) -> GenerationResult:
        return GenerationResult(text=f"from {self.name}", tokens_generated=1)

    def chat_stream(self, messages, config=None, tools=None, **kw):
        yield f"from {self.name}"

    def generate_stream(self, prompt, config=None):
        yield f"from {self.name}"


class World:
    """Registry, engine factory and memory the tests control.

    Memory is computed from what is resident: other programs use
    ``others`` GB, and HFL's resident set is the sum of its models.
    """

    def __init__(self, tmp_path: Path, sizes: dict[str, int], others: int, total: int = 100):
        from hfl.models.manifest import ModelManifest

        self.sizes = sizes
        self.others = others * GB
        self.total = total * GB
        self.engines: dict[str, list[FakeEngine]] = {}
        self.events: list[str] = []
        self.manifests = {}
        for name in sizes:
            path = tmp_path / f"{name}.gguf"
            path.write_bytes(b"\0")
            self.manifests[name] = ModelManifest(
                name=name, repo_id=f"t/{name}", local_path=str(path), format="gguf"
            )

    def memory(self) -> MemoryView:
        from hfl.api.state import get_state

        rss = sum(r.footprint for r in get_state().resident_models())
        return MemoryView(total=self.total, in_use=self.others + rss, hfl_rss=rss)

    def engine_for(self, path) -> FakeEngine:
        name = Path(path).stem
        engine = FakeEngine(name)
        original = engine.unload

        def unload() -> None:
            self.events.append(f"unload {name}")
            original()

        engine.unload = unload  # type: ignore[method-assign]
        self.engines.setdefault(name, []).append(engine)
        self.events.append(f"new {name}")
        return engine


@pytest.fixture
def world(tmp_path, temp_config, monkeypatch):
    from hfl.api import model_loader
    from hfl.api.state import reset_state
    from hfl.converter.formats import ModelType
    from hfl.core import get_dispatcher

    reset_state()
    get_dispatcher().reset()
    made: dict[str, World] = {}

    def build(sizes: dict[str, int], others: int = 10, total: int = 100) -> World:
        w = World(tmp_path, sizes, others, total)
        registry = type("R", (), {"get": staticmethod(lambda n: w.manifests.get(n))})()
        monkeypatch.setattr(model_loader, "get_registry", lambda: registry)
        monkeypatch.setattr(model_loader, "detect_model_type", lambda p: ModelType.LLM)
        monkeypatch.setattr(model_loader, "select_engine", w.engine_for)
        monkeypatch.setattr(
            "hfl.engine.footprint.estimate_footprint",
            lambda path, n_ctx=0: _fp(w.sizes[Path(path).stem]),
        )
        monkeypatch.setattr(model_loader, "_measure_loaded", lambda e, m: w.sizes[m.name] * GB)
        monkeypatch.setattr("hfl.engine.residency.current_memory", w.memory)
        # Hermetic: no discrete GPU unless a test installs one.
        monkeypatch.setattr("hfl.engine.residency.current_gpu_memory", lambda: None)
        monkeypatch.setattr("hfl.engine.residency.discrete_gpu_unmeasured", lambda: False)
        made["w"] = w
        return w

    yield build
    reset_state()


def _fp(gb: int):
    from hfl.engine.footprint import Footprint

    return Footprint(gb * GB, 0, 0, False)


def _lease_scope():
    from hfl.api.state import close_lease_scope, open_lease_scope

    class Scope:
        def __enter__(self):
            self.token = open_lease_scope()
            return self

        def __exit__(self, *exc):
            close_lease_scope(self.token)

    return Scope()


async def _load(name: str, **kw):
    from hfl.api.model_loader import load_llm

    return await load_llm(name, **kw)


# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_models_stay_resident_when_they_fit(world):
    w = world({"a": 20, "b": 20})
    await _load("a")
    await _load("b")

    from hfl.api.state import get_state

    assert {r.name for r in get_state().resident_models()} == {"a", "b"}
    assert not w.engines["a"][0].unloaded


@pytest.mark.asyncio
async def test_each_request_keeps_its_own_model(world):
    """The regression this whole change could have introduced: request 1
    loads A, request 2 loads B, request 1 then reads ``state.engine`` —
    and must get A, not the most recently loaded B."""
    world({"a": 10, "b": 10})
    from hfl.api.state import get_state

    b_loaded = asyncio.Event()

    async def request_one():
        await _load("a")
        await b_loaded.wait()
        return get_state().engine.chat([]).text

    async def request_two():
        await _load("b")
        b_loaded.set()
        return get_state().engine.chat([]).text

    one, two = await asyncio.gather(
        asyncio.create_task(request_one()), asyncio.create_task(request_two())
    )
    assert one == "from a"
    assert two == "from b"


@pytest.mark.asyncio
async def test_room_is_made_from_the_least_recently_used_idle_model(world):
    # Budget 85 GB, others 10: a 40 + b 30 fit (80); c 30 needs one out.
    w = world({"a": 40, "b": 30, "c": 30})
    await _load("a")
    await _load("b")
    await _load("a")  # a is now more recently used than b

    await _load("c")

    from hfl.api.state import get_state

    assert {r.name for r in get_state().resident_models()} == {"a", "c"}
    assert w.engines["b"][0].unloaded and not w.engines["a"][0].unloaded
    # b was unloaded BEFORE c was allocated, or peak memory would double.
    assert w.events.index("unload b") < w.events.index("new c")


@pytest.mark.asyncio
async def test_a_model_in_use_is_waited_for_never_unloaded_under_its_request(world):
    w = world({"a": 50, "b": 40})

    release = asyncio.Event()
    a_leased = asyncio.Event()

    async def request_using_a():
        with _lease_scope():
            await _load("a")
            a_leased.set()
            await release.wait()
            assert not w.engines["a"][0].unloaded, "a was unloaded while in use"

    user = asyncio.create_task(request_using_a())
    await a_leased.wait()

    loader = asyncio.create_task(_load("b"))
    await asyncio.sleep(0.2)
    assert not loader.done(), "b loaded without room — or evicted a busy model"
    assert not w.engines["a"][0].unloaded

    release.set()
    await user
    await asyncio.wait_for(loader, 5)
    assert w.engines["a"][0].unloaded


@pytest.mark.asyncio
async def test_waiting_for_busy_models_is_bounded(world, monkeypatch):
    from hfl.config import config
    from hfl.exceptions import ModelsBusyError

    world({"a": 50, "b": 40})
    monkeypatch.setattr(config, "queue_acquire_timeout_seconds", 0.3)
    hold = asyncio.Event()
    leased = asyncio.Event()

    async def other_request():
        with _lease_scope():
            await _load("a")
            leased.set()
            await hold.wait()

    user = asyncio.create_task(other_request())
    await leased.wait()
    try:
        with pytest.raises(ModelsBusyError):
            await _load("b")
    finally:
        hold.set()
        await user


@pytest.mark.asyncio
async def test_a_model_the_loading_request_holds_is_not_waited_for(world):
    """Waiting for our own lease would wait forever: refuse instead."""
    from hfl.exceptions import MemoryBudgetExceededError

    world({"a": 50, "b": 40})
    with _lease_scope():
        await _load("a")
        with pytest.raises(MemoryBudgetExceededError, match="this same request"):
            await _load("b")


@pytest.mark.asyncio
async def test_too_big_is_refused_before_anything_is_unloaded(world):
    from hfl.api.state import get_state
    from hfl.exceptions import MemoryBudgetExceededError

    w = world({"a": 20, "huge": 90})
    await _load("a")
    with pytest.raises(MemoryBudgetExceededError) as info:
        await _load("huge")

    assert "HFL_MEMORY_BUDGET" in str(info.value)
    assert not w.engines["a"][0].unloaded
    assert [r.name for r in get_state().resident_models()] == ["a"]


@pytest.mark.asyncio
async def test_the_optional_count_ceiling(world, monkeypatch):
    from hfl.config import config

    w = world({"a": 1, "b": 1})
    monkeypatch.setattr(config, "max_loaded_models", 1)
    await _load("a")
    await _load("b")
    assert w.engines["a"][0].unloaded


@pytest.mark.asyncio
async def test_a_reload_unloads_the_stale_copy_first(world):
    """A bigger num_ctx reloads the model; the old copy goes before the new
    one is allocated (the llama.cpp preflight would otherwise measure free
    memory with both in it and reject a load that fits)."""
    w = world({"a": 30})
    await _load("a", num_ctx=4096)
    await _load("a", num_ctx=16384)
    assert w.events == ["new a", "unload a", "new a"]


@pytest.mark.asyncio
async def test_leases_end_with_their_scope(world):
    from hfl.api.state import get_state

    world({"a": 10})
    with _lease_scope():
        engine, _ = await _load("a")
        assert get_state()._engine_inuse.get(id(engine)) == 1
    assert id(engine) not in get_state()._engine_inuse


def test_http_requests_to_two_models(world):
    """End to end: two models through the API, both resident afterwards,
    each answer from its own model, and no lease outlives its request."""
    from fastapi.testclient import TestClient

    from hfl.api.server import app
    from hfl.api.state import get_state

    w = world({"a": 10, "b": 10})
    client = TestClient(app, client=("127.0.0.1", 5555))
    for name in ("a", "b", "a"):
        response = client.post(
            "/api/chat",
            json={"model": name, "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert response.status_code == 200, response.text
        assert response.json()["message"]["content"] == f"from {name}"

    streamed = client.post(
        "/api/chat",
        json={"model": "b", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert "from b" in streamed.text

    names = [m["name"] for m in client.get("/api/ps").json()["models"]]
    assert sorted(names) == ["a", "b"]
    served = w.engines["a"][0]
    assert served.leases_while_serving == 1, "the request did not hold its model"
    assert get_state()._engine_inuse == {}, "a lease outlived its request"


def test_a_websocket_turn_that_fails_before_streaming_releases_its_model(world):
    """The WS turn leases its model right after loading it. If the turn dies
    before the producer starts (here: an unparseable option), the lease must
    still be released, or the model could never be evicted again."""
    import json
    import time

    from fastapi.testclient import TestClient

    from hfl.api.server import app
    from hfl.api.state import get_state

    world({"a": 10})
    client = TestClient(app, client=("127.0.0.1", 5555))
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "chat",
                    "model": "a",
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {"temperature": "not-a-number"},
                }
            )
        )
        ws.send_text(json.dumps({"type": "ping"}))
        ws.receive_text()
    deadline = time.monotonic() + 2
    while get_state()._engine_inuse and time.monotonic() < deadline:
        time.sleep(0.02)
    assert get_state().resident("a") is not None, "the model never loaded — test proves nothing"
    assert get_state()._engine_inuse == {}, "the failed turn kept its model leased"


# ----------------------------------------------------------------------
# keep_alive is enforced
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_idle_models_are_unloaded_and_the_others_kept(world):
    from datetime import datetime, timedelta, timezone

    from hfl.api.state import get_state

    w = world({"old": 10, "fresh": 10, "busy": 10})
    state = get_state()
    await _load("old")
    await _load("fresh")
    hold = asyncio.Event()
    leased = asyncio.Event()

    async def user():
        with _lease_scope():
            await _load("busy")
            leased.set()
            await hold.wait()

    task = asyncio.create_task(user())
    await leased.wait()
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    state.set_keep_alive_deadline("old", past)
    state.set_keep_alive_deadline("busy", past)

    assert await state.reap_expired() == ["old"]
    assert w.engines["old"][0].unloaded
    assert not w.engines["fresh"][0].unloaded
    assert not w.engines["busy"][0].unloaded, "reaped a model in use"

    hold.set()
    await task
    # Released: its clock restarted from now, so it is not expired any more.
    assert state.keep_alive_deadline_for("busy") > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_every_use_renews_the_deadline_whatever_the_api(world, monkeypatch):
    """/v1 routes never call apply_keep_alive; the renewal happens when the
    model is bound to the request, so it covers every dialect."""
    from datetime import datetime, timedelta, timezone

    from hfl.api.state import get_state
    from hfl.config import config

    world({"a": 10})
    monkeypatch.setattr(config, "keep_alive_default", "10m")
    await _load("a")
    get_state().set_keep_alive_deadline("a", datetime.now(timezone.utc))
    await _load("a")
    deadline = get_state().keep_alive_deadline_for("a")
    assert deadline > datetime.now(timezone.utc) + timedelta(minutes=9)


@pytest.mark.asyncio
async def test_never_expire_is_respected(world):
    from hfl.api.state import get_state

    world({"a": 10})
    await _load("a")
    get_state().set_keep_alive("a", None)
    assert get_state().keep_alive_deadline_for("a") is None
    assert await get_state().reap_expired() == []


def test_the_server_reaper_unloads_an_expired_model(world, monkeypatch):
    """End to end with the real lifespan: a model idle past its keep_alive
    disappears from /api/ps without any request asking for it."""
    import time

    from fastapi.testclient import TestClient

    from hfl.api import server
    from hfl.config import config

    w = world({"a": 10})
    monkeypatch.setattr(server, "KEEP_ALIVE_REAP_INTERVAL", 0.1)
    monkeypatch.setattr(config, "keep_alive_default", "1s")
    with TestClient(server.app, client=("127.0.0.1", 5555)) as client:
        response = client.post(
            "/api/chat",
            json={"model": "a", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert response.status_code == 200
        assert [m["name"] for m in client.get("/api/ps").json()["models"]] == ["a"]
        deadline = time.monotonic() + 5
        while client.get("/api/ps").json()["models"] and time.monotonic() < deadline:
            time.sleep(0.1)
        assert client.get("/api/ps").json()["models"] == []
    assert w.engines["a"][0].unloaded


@pytest.mark.asyncio
async def test_a_count_ceiling_of_two_keeps_two(world, monkeypatch):
    """With room for two, the third load unloads exactly one. (The
    post-load reconcile once applied the ceiling minus one and evicted a
    second model right after every load.)"""
    from hfl.api.state import get_state
    from hfl.config import config

    w = world({"a": 1, "b": 1, "c": 1})
    monkeypatch.setattr(config, "max_loaded_models", 2)
    await _load("a")
    await _load("b")
    assert {r.name for r in get_state().resident_models()} == {"a", "b"}
    await _load("c")
    assert {r.name for r in get_state().resident_models()} == {"b", "c"}
    assert w.engines["a"][0].unloaded and not w.engines["b"][0].unloaded


@pytest.mark.asyncio
async def test_a_model_of_unknown_size_does_not_evict_everything(world, monkeypatch):
    """Without a size estimate only the (optional) count ceiling can apply;
    the others must stay loaded."""
    from hfl.api.state import get_state

    w = world({"a": 10, "b": 10, "mystery": 0})
    await _load("a")
    await _load("b")
    await _load("mystery")
    assert {r.name for r in get_state().resident_models()} == {"a", "b", "mystery"}
    assert not w.engines["a"][0].unloaded and not w.engines["b"][0].unloaded


def test_a_preloaded_model_gets_a_keep_alive_deadline(world):
    """`hfl serve --model` assigns the pointer directly; a model nobody then
    uses must still expire."""
    from hfl.api.state import get_state

    w = world({"a": 10})
    state = get_state()
    engine = w.engine_for(w.manifests["a"].local_path)
    engine.load("x")
    state.engine = engine
    state.current_model = w.manifests["a"]
    assert state.keep_alive_deadline_for("a") is not None


# ----------------------------------------------------------------------
# Discrete GPUs: the limit is VRAM, not RAM
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_discrete_gpu_limits_residency_even_with_ram_to_spare(world, monkeypatch):
    """128 GB of RAM and a 32 GB card (27.2 GB at 85%): two 10 GB models
    fit the card, a third must evict one — RAM alone would have kept all
    three and the third load would have failed in CUDA."""
    from hfl.api.state import get_state

    w = world({"a": 10, "b": 10, "c": 10}, others=10, total=128)

    def gpu():
        mine = sum(r.footprint for r in get_state().resident_models())
        return MemoryView(total=32 * GB, in_use=1 * GB + mine, hfl_rss=mine)

    monkeypatch.setattr("hfl.engine.residency.current_gpu_memory", gpu)
    await _load("a")
    await _load("b")
    await _load("c")
    assert {r.name for r in get_state().resident_models()} == {"b", "c"}
    assert w.engines["a"][0].unloaded
    # Before c was allocated: evicting afterwards is too late on a real
    # card, where c's load itself runs out of VRAM.
    assert w.events.index("unload a") < w.events.index("new c")


@pytest.mark.asyncio
async def test_a_model_larger_than_the_card_is_refused_and_says_gpu(world, monkeypatch):
    from hfl.exceptions import MemoryBudgetExceededError

    world({"big": 30}, others=10, total=128)
    monkeypatch.setattr(
        "hfl.engine.residency.current_gpu_memory",
        lambda: MemoryView(total=24 * GB, in_use=1 * GB, hfl_rss=0),
    )
    with pytest.raises(MemoryBudgetExceededError, match="GPU memory") as info:
        await _load("big")
    assert info.value.on_gpu


@pytest.mark.asyncio
async def test_an_unmeasurable_gpu_falls_back_to_one_model(world, monkeypatch):
    w = world({"a": 1, "b": 1})
    monkeypatch.setattr("hfl.engine.residency.discrete_gpu_unmeasured", lambda: True)
    await _load("a")
    await _load("b")
    assert w.engines["a"][0].unloaded


@pytest.mark.asyncio
async def test_an_explicit_ceiling_wins_over_the_unmeasured_gpu_fallback(world, monkeypatch):
    from hfl.config import config

    w = world({"a": 1, "b": 1})
    monkeypatch.setattr("hfl.engine.residency.discrete_gpu_unmeasured", lambda: True)
    monkeypatch.setattr(config, "max_loaded_models", 2)
    await _load("a")
    await _load("b")
    assert not w.engines["a"][0].unloaded

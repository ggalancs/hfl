# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Security: a client's ``num_ctx`` cannot unload the model others use.

``options.num_ctx`` larger than the resident window reloads the model. The
stale copy was retired first and the memory budget checked after, so any
client could send ``num_ctx: 10**7``: the resident model was unloaded, the
new load refused, and every other client lost the model. The load is now
refused before anything is unloaded, and a request's ``num_ctx`` is held to
the context the model was trained for (as Ollama does).
"""

from __future__ import annotations

import pytest

from hfl.engine.residency import MemoryView

GB = 1024**3
MB = 1024**2


class FakeEngine:
    def __init__(self) -> None:
        self.is_loaded = False
        self.unloaded = False
        self.context_size = 0

    def load(self, path: str, **kwargs) -> None:
        self.is_loaded = True
        self.context_size = int(kwargs.get("n_ctx") or 0)

    def unload(self) -> None:
        self.is_loaded = False
        self.unloaded = True


@pytest.fixture
def world(tmp_path, temp_config, monkeypatch):
    """One 20 GB model on a 100 GB machine; its KV cache costs 1 MB a token."""
    from hfl.api import model_loader
    from hfl.api.state import get_state, reset_state
    from hfl.converter.formats import ModelType
    from hfl.core import get_dispatcher
    from hfl.engine.footprint import Footprint
    from hfl.models.manifest import ModelManifest

    reset_state()
    get_dispatcher().reset()
    path = tmp_path / "m.gguf"
    path.write_bytes(b"\0")
    manifest = ModelManifest(name="m", repo_id="t/m", local_path=str(path), format="gguf")
    engines: list[FakeEngine] = []

    def make_engine(_path) -> FakeEngine:
        engines.append(FakeEngine())
        return engines[-1]

    def footprint(_path, n_ctx=0):
        return Footprint(20 * GB, n_ctx * MB, n_ctx, True)

    def memory() -> MemoryView:
        rss = sum(r.footprint for r in get_state().resident_models())
        return MemoryView(total=100 * GB, in_use=10 * GB + rss, hfl_rss=rss)

    registry = type("R", (), {"get": staticmethod(lambda n: manifest if n == "m" else None)})()
    monkeypatch.setattr(model_loader, "get_registry", lambda: registry)
    monkeypatch.setattr(model_loader, "detect_model_type", lambda p: ModelType.LLM)
    monkeypatch.setattr(model_loader, "select_engine", make_engine)
    monkeypatch.setattr("hfl.engine.footprint.estimate_footprint", footprint)
    monkeypatch.setattr(
        model_loader, "_measure_loaded", lambda e, m: footprint(None, e.context_size).total_bytes
    )
    monkeypatch.setattr("hfl.engine.residency.current_memory", memory)
    monkeypatch.setattr("hfl.engine.residency.current_gpu_memory", lambda: None)
    monkeypatch.setattr("hfl.engine.residency.discrete_gpu_unmeasured", lambda: False)
    monkeypatch.setattr("hfl.engine.footprint.advertised_context", lambda p: 0)
    yield engines
    reset_state()


@pytest.mark.asyncio
async def test_oversized_num_ctx_is_refused_and_the_resident_stays(world):
    from hfl.api.model_loader import load_llm
    from hfl.api.state import get_state
    from hfl.exceptions import MemoryBudgetExceededError

    engines = world
    await load_llm("m", num_ctx=4096)
    resident = engines[0]
    with pytest.raises(MemoryBudgetExceededError):
        await load_llm("m", num_ctx=10**7)
    assert not resident.unloaded and resident.is_loaded
    assert [r.name for r in get_state().resident_models()] == ["m"]
    assert len(engines) == 1  # no new load was even started
    # And it still serves.
    engine, _ = await load_llm("m", num_ctx=4096)
    assert engine is resident


@pytest.mark.asyncio
async def test_a_growth_that_fits_still_reloads(world):
    """The refusal is for loads that cannot fit even alone: growing the
    window within the budget reloads as before."""
    from hfl.api.model_loader import load_llm

    engines = world
    await load_llm("m", num_ctx=4096)
    engine, _ = await load_llm("m", num_ctx=32768)
    assert engines[0].unloaded and engine is engines[1]
    assert engine.context_size == 32768


@pytest.mark.asyncio
async def test_num_ctx_is_held_to_the_trained_context(world, monkeypatch):
    from hfl.api.model_loader import load_llm

    engines = world
    monkeypatch.setattr("hfl.engine.footprint.advertised_context", lambda p: 32768)
    engine, _ = await load_llm("m", num_ctx=10**7)
    assert engine.context_size == 32768
    # A larger ask than the model can use does not reload it.
    again, _ = await load_llm("m", num_ctx=10**6)
    assert again is engine and len(engines) == 1


def test_advertised_context_reads_config_json(tmp_path):
    from hfl.engine.footprint import advertised_context

    (tmp_path / "config.json").write_text('{"max_position_embeddings": 8192}')
    assert advertised_context(tmp_path) == 8192
    assert advertised_context(tmp_path / "missing") == 0

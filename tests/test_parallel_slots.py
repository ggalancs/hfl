# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``HFL_NUM_PARALLEL=1`` gives llama-server one slot. It used to mean "the
default" and gave 4: the default count and an explicit 1 were the same value
(plan 0.22 P1-12)."""

from __future__ import annotations

import pytest

from hfl.config import HFLConfig


@pytest.fixture
def cfg(monkeypatch):
    def make(**env: str) -> HFLConfig:
        for name in ("HFL_QUEUE_MAX_INFLIGHT", "HFL_NUM_PARALLEL", "OLLAMA_NUM_PARALLEL"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        made = HFLConfig()
        monkeypatch.setattr("hfl.config.config", made)
        return made

    return make


def test_unset_is_the_default_four(cfg) -> None:
    from hfl.engine.llama_server import DEFAULT_SLOTS, _slots

    cfg()
    assert _slots() == DEFAULT_SLOTS == 4


@pytest.mark.parametrize(
    "name", ["HFL_NUM_PARALLEL", "OLLAMA_NUM_PARALLEL", "HFL_QUEUE_MAX_INFLIGHT"]
)
def test_an_explicit_one_is_one(cfg, name) -> None:
    from hfl.engine.llama_server import _slots

    cfg(**{name: "1"})
    assert _slots() == 1


def test_an_explicit_count_is_that_count(cfg) -> None:
    from hfl.engine.llama_server import _slots

    cfg(HFL_NUM_PARALLEL="6")
    assert _slots() == 6


def test_serve_parallel_one_is_one(cfg, monkeypatch) -> None:
    from hfl.cli.main import _choose_backend
    from hfl.engine.llama_server import _slots

    made = cfg()
    monkeypatch.setattr("hfl.engine.llama_server.binary", lambda: "/opt/llama-server")
    # Recorded as unset now, so the value _choose_backend writes is undone.
    monkeypatch.setenv("HFL_LLM_LIBRARY", "")
    monkeypatch.delenv("HFL_LLM_LIBRARY")
    _choose_backend("llama-server", 1)
    assert made.parallel_explicit and _slots() == 1

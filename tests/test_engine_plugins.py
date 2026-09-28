# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Engines registered by other packages under the ``hfl.engines`` entry
point are selectable by name. The discovery existed but no backend choice
consulted it, so a plugin could never run. A plugin may not take a
built-in engine's name. (A real install of examples/hfl-echo-engine served
``hfl serve --backend echo``; here the entry points are simulated.)"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hfl.engine.base import InferenceEngine


class _Echo(InferenceEngine):
    def load(self, model_path, **kwargs): ...
    def unload(self): ...
    def generate(self, prompt, config=None): ...
    def generate_stream(self, prompt, config=None): ...
    def chat(self, messages, config=None, tools=None): ...
    def chat_stream(self, messages, config=None, tools=None): ...

    @property
    def model_name(self):
        return "m"

    @property
    def is_loaded(self):
        return True


@pytest.fixture
def plugins(monkeypatch):
    import importlib.metadata

    from hfl import plugins as registry

    def entry(name, target):
        return SimpleNamespace(name=name, load=lambda: target)

    found = [entry("echo", _Echo), entry("llama-cpp", _Echo), entry("broken", lambda: object())]
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: found if group == "hfl.engines" else [],
    )
    monkeypatch.setattr(registry, "_engine_cache", None)
    yield
    registry._engine_cache = None


def test_a_plugin_is_created_by_name(plugins) -> None:
    from hfl.engine.selector import _create_engine

    assert isinstance(_create_engine("echo"), _Echo)


def test_a_plugin_cannot_take_a_builtin_name(plugins) -> None:
    from hfl.engine.selector import plugin_engines

    assert "llama-cpp" not in plugin_engines()


def test_hfl_llm_library_accepts_a_plugin(plugins, monkeypatch) -> None:
    from hfl.engine.selector import _resolve_forced_backend

    monkeypatch.setenv("HFL_LLM_LIBRARY", "echo")
    assert _resolve_forced_backend() == "echo"


def test_a_plugin_that_is_not_an_engine_is_refused(plugins) -> None:
    from hfl.engine.selector import _create_engine

    with pytest.raises(ValueError, match="did not return an InferenceEngine"):
        _create_engine("broken")

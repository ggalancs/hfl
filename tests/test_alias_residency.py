# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A model asked for by alias stays resident under one name.

The alias ("chat") was the resident's key; loading another model re-filed
the first under its manifest name, and every later "chat" loaded another
copy — measured: 5 alternating requests, 5 loads, ~14 s and 0.6 GB each,
memory climbing. Resident models are kept under the registered name."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


class _Engine:
    loads: dict[str, int] = {}

    def __init__(self) -> None:
        self.is_loaded, self.path = False, ""

    def load(self, path, **kwargs):
        self.path = str(path)
        _Engine.loads[self.path] = _Engine.loads.get(self.path, 0) + 1
        self.is_loaded = True

    def unload(self):
        self.is_loaded = False

    def generate(self, prompt, config=None):
        return MagicMock(text="ok", tokens_generated=1, tokens_prompt=1)


@pytest.fixture
def server(temp_config, monkeypatch):
    from hfl.api.server import app
    from hfl.api.state import reset_state
    from hfl.core.container import get_registry, reset_container
    from hfl.models.manifest import ModelManifest

    reset_container()
    reset_state()
    _Engine.loads = {}
    for name, alias in (("qwen2.5-0.5b-q4", "chat"), ("qwen2.5-coder-1.5b-q4", "coder")):
        path = temp_config.models_dir / f"{name}.gguf"
        path.write_bytes(b"GGUF")
        get_registry().add(
            ModelManifest(name=name, repo_id=f"o/{name}", local_path=str(path), format="gguf",
                          alias=alias)
        )  # fmt: skip
    monkeypatch.setattr("hfl.api.model_loader.select_engine", lambda path: _Engine())
    yield TestClient(app, client=("127.0.0.1", 5555))
    reset_state()
    reset_container()


def _ask(client, model):
    body = {"model": model, "prompt": "hi", "stream": False, "options": {"num_predict": 1}}
    return client.post("/api/generate", json=body)


def test_alternating_aliases_load_each_model_once(server) -> None:
    for model in ("chat", "coder", "chat", "coder", "chat"):
        assert _ask(server, model).status_code == 200
    assert sorted(_Engine.loads.values()) == [1, 1], _Engine.loads


def test_the_resident_is_kept_under_its_registered_name(server) -> None:
    from hfl.api.state import get_state

    _ask(server, "chat")
    _ask(server, "coder")
    names = sorted(r.name for r in get_state().resident_models())
    assert names == ["qwen2.5-0.5b-q4", "qwen2.5-coder-1.5b-q4"]


def test_a_keep_alive_sent_with_the_alias_reaches_the_model(server) -> None:
    """Set under "chat", read under the model's name: an explicit keep_alive
    from a client using the alias was never applied."""
    from datetime import timedelta

    from hfl.api.state import get_state

    body = {"model": "chat", "prompt": "hi", "stream": False, "keep_alive": "42m",
            "options": {"num_predict": 1}}  # fmt: skip
    assert server.post("/api/generate", json=body).status_code == 200
    assert get_state().keep_alive_duration_for("qwen2.5-0.5b-q4") == timedelta(minutes=42)


def _resident_names() -> list[str]:
    from hfl.api.state import get_state

    return sorted(r.name for r in get_state().resident_models())


def test_keep_alive_zero_sent_with_the_alias_unloads_the_model(server) -> None:
    """``keep_alive: 0`` looked the resident up by the alias and found
    nothing: the model stayed loaded (measured on a real server)."""
    _ask(server, "coder")
    body = {"model": "chat", "prompt": "hi", "stream": False, "keep_alive": 0,
            "options": {"num_predict": 1}}  # fmt: skip
    assert server.post("/api/generate", json=body).status_code == 200
    assert _resident_names() == ["qwen2.5-coder-1.5b-q4"]  # only chat's model left


def test_stop_by_alias_unloads_the_model(server) -> None:
    _ask(server, "chat")
    _ask(server, "coder")
    out = server.post("/api/stop", json={"model": "chat"}).json()
    assert out["status"] == "stopped", out
    assert _resident_names() == ["qwen2.5-coder-1.5b-q4"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl start``: the first run — a model that fits, downloaded, a chat."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hfl.cli import main

runner = CliRunner()


@pytest.fixture
def world(monkeypatch):
    calls: dict[str, list] = {"pull": [], "run": [], "find": []}
    monkeypatch.setattr(
        "hfl.hub.hw_profile.get_hw_profile",
        lambda: SimpleNamespace(os="darwin", arch="arm64", system_ram_gb=16.0, gpu_kind="metal"),
    )

    def find(name):
        calls["find"].append(name)
        return SimpleNamespace(repo_id=f"org/{name}", size_bytes=1_000_000_000)

    monkeypatch.setattr("hfl.hub.shortname.find", find)
    monkeypatch.setattr(main, "pull", lambda **kw: calls["pull"].append(kw))
    monkeypatch.setattr(main, "run", lambda **kw: calls["run"].append(kw))
    monkeypatch.setattr(main, "_a_chat_model", lambda: None)
    return calls


def test_a_first_run_pulls_the_default_and_opens_a_chat(world):
    result = runner.invoke(main.app, ["start", "--yes"])
    assert result.exit_code == 0, result.output
    assert world["find"] == ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:0.5b"]  # fits 16 GB
    assert world["pull"][0]["model"] == "qwen2.5:1.5b" and world["pull"][0]["yes"] is True
    assert world["run"] and world["run"][0]["model"] == "qwen2.5-1.5b"


def test_little_ram_is_offered_only_small_models(world, monkeypatch):
    monkeypatch.setattr(
        "hfl.hub.hw_profile.get_hw_profile",
        lambda: SimpleNamespace(os="linux", arch="x86_64", system_ram_gb=8.0, gpu_kind="none"),
    )
    runner.invoke(main.app, ["start", "--yes", "--no-chat"])
    assert world["find"] == ["qwen2.5:1.5b", "qwen2.5:0.5b"] and world["run"] == []


def test_a_model_already_here_is_opened_instead(world, monkeypatch):
    monkeypatch.setattr(main, "_a_chat_model", lambda: "chat")
    result = runner.invoke(main.app, ["start"])
    assert result.exit_code == 0
    assert world["pull"] == [] and world["run"][0]["model"] == "chat"


def test_nothing_found_on_the_hub_says_so(world, monkeypatch):
    monkeypatch.setattr("hfl.hub.shortname.find", lambda name: None)
    result = runner.invoke(main.app, ["start", "--yes"])
    assert result.exit_code == 1 and world["pull"] == []


def test_a_choice_can_be_typed(world, monkeypatch):
    monkeypatch.setattr(main.console, "input", lambda prompt="": "2")
    runner.invoke(main.app, ["start", "--no-chat"])
    assert world["pull"][0]["model"] == "qwen2.5:7b"

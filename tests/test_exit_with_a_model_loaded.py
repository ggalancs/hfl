# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Leaving with a llama.cpp model loaded aborted the process: ggml frees its
Metal device at exit and asserts no buffer is in use
(``GGML_ASSERT([rsets->data count] == 0)``, exit 134). The tray's quit hit it
(crash of 2026-08-27; reproduced, and fixed, with a real AppKit terminate)."""

from __future__ import annotations

import sys
import types

import pytest


def test_unload_all_unloads_every_live_engine() -> None:
    from hfl.engine import llama_cpp

    unloaded: list[str] = []

    class Engine:
        def __init__(self, name: str) -> None:
            self.name = name

        def unload(self) -> None:
            unloaded.append(self.name)

    a, b = Engine("a"), Engine("b")
    llama_cpp._LIVE.add(a)
    llama_cpp._LIVE.add(b)
    llama_cpp.unload_all()
    assert sorted(unloaded) == ["a", "b"]


def test_one_engine_failing_does_not_stop_the_others() -> None:
    from hfl.engine import llama_cpp

    done: list[str] = []

    class Broken:
        def unload(self) -> None:
            raise RuntimeError("boom")

    class Fine:
        def unload(self) -> None:
            done.append("fine")

    keep = [Broken(), Fine()]
    for engine in keep:
        llama_cpp._LIVE.add(engine)
    llama_cpp.unload_all()
    assert done == ["fine"]


def test_the_tray_stops_the_server_when_macos_quits_it(monkeypatch) -> None:
    """AppKit's terminate: ends the process from C, skipping Python's exit
    handlers; the tray listens for NSApplicationWillTerminateNotification."""
    from hfl.tray import icon

    registered: dict = {}

    class Center:
        def addObserverForName_object_queue_usingBlock_(self, name, obj, queue, block):
            registered["name"], registered["block"] = name, block
            return "observer"

    foundation = types.SimpleNamespace(
        NSNotificationCenter=types.SimpleNamespace(defaultCenter=lambda: Center())
    )
    appkit = types.SimpleNamespace(NSApplicationWillTerminateNotification="will-terminate")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "Foundation", foundation)
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    stopped: list[bool] = []
    controller = types.SimpleNamespace(stop=lambda: stopped.append(True))

    assert icon._stop_on_macos_quit(controller) == "observer"
    assert registered["name"] == "will-terminate"
    registered["block"](None)
    assert stopped == [True]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_elsewhere_nothing_is_registered(monkeypatch, platform) -> None:
    from hfl.tray import icon

    monkeypatch.setattr(sys, "platform", platform)
    assert icon._stop_on_macos_quit(types.SimpleNamespace(stop=lambda: None)) is None

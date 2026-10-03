# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A load that outlives its request is unloaded, not leaked.

Measured in the soak: while the Mac slept, loads timed out; the request was
cancelled but the loading thread went on, and what it loaded (a
llama-server) belonged to nobody — 32 of them by morning. The engine is
now released when the thread finishes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest


class SlowEngine:
    def __init__(self) -> None:
        self.is_loaded = False
        self.unloaded = threading.Event()

    def load(self, path: str, **kwargs) -> None:
        time.sleep(0.5)  # longer than the request waits
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False
        self.unloaded.set()


@pytest.mark.asyncio
async def test_a_load_the_request_gave_up_on_is_unloaded(monkeypatch):
    from hfl.api import model_loader

    monkeypatch.setattr(model_loader, "load_kwargs_for", lambda manifest, n_ctx: {})
    engine = SlowEngine()
    manifest = SimpleNamespace(local_path="/x/m.gguf")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(model_loader._load_into(engine, manifest, 0), timeout=0.1)
    for _ in range(40):  # the thread finishes, then the release runs
        if engine.unloaded.is_set():
            break
        await asyncio.sleep(0.05)
    assert engine.unloaded.is_set() and not engine.is_loaded

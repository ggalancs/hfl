# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HFL loads Transformers weights on one thread: with safetensors 0.8.0 its
loader threads can deadlock (seen once, freezing the whole server; not
reproduced). Measured: with ``hfl.engine`` imported Transformers creates no
loader pool, without it one; load time unchanged for 0.5B and 1.1B."""

from __future__ import annotations

import importlib

import hfl.engine


def test_hfl_asks_for_single_thread_loading(monkeypatch) -> None:
    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    importlib.reload(hfl.engine)
    import os

    assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"


def test_the_users_own_setting_wins(monkeypatch) -> None:
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "0")
    importlib.reload(hfl.engine)
    import os

    assert os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "0"

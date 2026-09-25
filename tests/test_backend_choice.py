# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The backend is chosen per model; parallel requests are an option.

`hfl serve --backend auto` (the default) picks per model as always.
`--parallel N` asks for N requests at once per GGUF model, which needs
llama-server (vision models included, with their projector); anything that
is not GGUF keeps its usual backend. When a request has to wait on the in-process GGUF backend,
the log says once how to serve several at once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner


@pytest.fixture
def serve(monkeypatch, temp_config, tmp_path):
    from hfl.cli.main import app

    # `serve` sets HFL_LLM_LIBRARY in this process's environment; delenv on
    # an absent variable records nothing to undo, so restore it by hand —
    # a leaked value forced llama-server on every later selector test.
    before = os.environ.pop("HFL_LLM_LIBRARY", None)
    started = MagicMock()
    monkeypatch.setattr("hfl.api.server.start_server", started)
    fake = tmp_path / "llama-server"
    fake.write_text("")

    def run(*args, binary=True):
        monkeypatch.setenv("HFL_LLAMA_SERVER_BIN", str(fake if binary else tmp_path / "none"))
        return CliRunner().invoke(app, ["serve", "--host", "127.0.0.1", *args])

    yield run, started
    os.environ.pop("HFL_LLM_LIBRARY", None)
    if before is not None:
        os.environ["HFL_LLM_LIBRARY"] = before


def test_auto_changes_nothing(serve):
    run, started = serve
    assert run().exit_code == 0
    assert "HFL_LLM_LIBRARY" not in os.environ
    started.assert_called_once()


def test_parallel_implies_llama_server_for_gguf(serve, temp_config):
    import hfl.config

    run, _ = serve
    result = run("--parallel", "4")
    assert result.exit_code == 0, result.output
    assert os.environ["HFL_LLM_LIBRARY"] == "llama-server"
    assert hfl.config.config.queue_max_inflight == 4
    assert "4 at once" in result.output


def test_a_named_backend_is_used(serve):
    run, _ = serve
    assert run("--backend", "llama-cpp").exit_code == 0
    assert os.environ["HFL_LLM_LIBRARY"] == "llama-cpp"


def test_an_unknown_backend_is_refused(serve):
    run, started = serve
    result = run("--backend", "turbo")
    assert result.exit_code == 2 and "turbo" in result.output
    started.assert_not_called()


def test_parallel_without_llama_server_says_what_to_install(serve):
    run, started = serve
    result = run("--parallel", "4", binary=False)
    assert result.exit_code == 1 and "llama.cpp" in result.output
    started.assert_not_called()


def test_a_vision_model_goes_to_llama_server_too(monkeypatch, tmp_path):
    """llama-server serves it with its projector (``--mmproj``); it used to
    stay on the in-process engine, which then took no parallel requests."""
    from hfl.engine.llama_server import LlamaServerEngine
    from hfl.engine.selector import select_engine

    monkeypatch.setenv("HFL_LLM_LIBRARY", "llama-server")
    folder = tmp_path / "vlm"
    folder.mkdir()
    model = folder / "model-Q4_K_M.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    assert isinstance(select_engine(model), LlamaServerEngine)
    (folder / "mmproj-model-f16.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    assert isinstance(select_engine(model), LlamaServerEngine)


class _Engine:
    parallel_hint = True
    supports_concurrent_inference = False

    def work(self, seconds: float) -> str:
        time.sleep(seconds)
        return "ok"


@pytest.fixture
def fresh(temp_config, monkeypatch):
    import hfl.api.helpers as helpers
    from hfl.core import reset_container

    reset_container()
    monkeypatch.setattr(helpers, "_parallel_suggested", False)
    yield helpers
    reset_container()


def _two_at_once(engine, batches=1):
    from hfl.api.helpers import run_dispatched

    async def run():
        for _ in range(batches):
            await asyncio.gather(run_dispatched(engine.work, 0.2), run_dispatched(engine.work, 0.2))

    asyncio.run(run())


def test_a_waiting_gguf_request_suggests_parallel_once(fresh, caplog):
    engine = _Engine()
    with caplog.at_level(logging.WARNING, logger="hfl.api.helpers"):
        _two_at_once(engine, batches=2)
    hints = [r for r in caplog.records if "--parallel" in r.getMessage()]
    assert len(hints) == 1


def test_no_hint_without_waiting_or_for_other_engines(fresh, caplog):
    other = _Engine()
    other.parallel_hint = False
    with caplog.at_level(logging.WARNING, logger="hfl.api.helpers"):
        _two_at_once(other)  # waits, but not a GGUF on the in-process backend
    from hfl.core import reset_container

    reset_container()
    with caplog.at_level(logging.WARNING, logger="hfl.api.helpers"):
        from hfl.api.helpers import run_dispatched

        asyncio.run(run_dispatched(_Engine().work, 0.01))  # never waits
    assert not [r for r in caplog.records if "--parallel" in r.getMessage()]


def test_the_in_process_gguf_engine_carries_the_hint():
    from hfl.engine.llama_cpp import LlamaCppEngine
    from hfl.engine.llama_server import LlamaServerEngine

    assert LlamaCppEngine.parallel_hint is True
    assert getattr(LlamaServerEngine, "parallel_hint", False) is False

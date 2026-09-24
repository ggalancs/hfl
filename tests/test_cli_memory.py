# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""`hfl run` / `hfl serve --model` / the tray say what a load does to memory,
and refuse a model that cannot fit — before reading any weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer

from hfl.engine.residency import MemoryView

GB = 1024**3


@pytest.fixture
def memory(monkeypatch):
    def set_view(in_use=40, total=100, rss=0):
        monkeypatch.setattr(
            "hfl.engine.residency.current_memory",
            lambda: MemoryView(total=total * GB, in_use=in_use * GB, hfl_rss=rss * GB),
        )

    set_view()
    return set_view


@pytest.fixture
def model_of(monkeypatch):
    from hfl.engine.footprint import Footprint

    def size(gb):
        monkeypatch.setattr(
            "hfl.engine.footprint.estimate_footprint",
            lambda path, n_ctx=0: Footprint(int(gb * GB), 0, 0, False),
        )
        return SimpleNamespace(name="m", local_path="/x/m.gguf")

    return size


def _check(manifest, capsys):
    from hfl.cli.main import _memory_check_or_exit

    _memory_check_or_exit(manifest, 0)
    return capsys.readouterr().out


class TestMemoryCheck:
    def test_a_fitting_load_is_reported_with_the_numbers(self, memory, model_of, capsys):
        out = _check(model_of(10), capsys)
        assert "40.0" in out and "10.0" in out and "50.0" in out  # now, need, after
        assert "85" in out

    def test_a_load_that_cannot_fit_is_refused(self, memory, model_of, capsys):
        from hfl.cli.main import _memory_check_or_exit

        with pytest.raises(typer.Exit) as info:
            _memory_check_or_exit(model_of(60), 0)
        assert info.value.exit_code == 1
        out = capsys.readouterr().out
        assert "HFL_MEMORY_BUDGET" in out and "60.0" in out

    def test_an_unknown_size_says_so_and_loads(self, memory, model_of, capsys):
        out = _check(model_of(0), capsys)
        assert "m" in out and ("estimate" in out or "estimar" in out)

    def test_the_check_can_be_disabled(self, memory, model_of, capsys, monkeypatch):
        monkeypatch.setenv("HFL_DISABLE_MEMORY_PREFLIGHT", "1")
        assert _check(model_of(60), capsys) == ""


def test_a_gpu_refusal_names_the_gpu(memory, model_of, capsys, monkeypatch):
    from hfl.cli.main import _memory_check_or_exit

    monkeypatch.setattr(
        "hfl.engine.residency.current_gpu_memory",
        lambda: MemoryView(total=24 * GB, in_use=1 * GB, hfl_rss=0),
    )
    with pytest.raises(typer.Exit):
        _memory_check_or_exit(model_of(30), 0)
    out = capsys.readouterr().out
    assert "GPU" in out and "24.0" in out


def test_hfl_run_refuses_before_loading(memory, model_of, monkeypatch, temp_config):
    """The refusal must come before select_engine — nothing is read."""
    from typer.testing import CliRunner

    from hfl.cli import main
    from hfl.converter.formats import ModelType

    manifest = model_of(60)
    registry = SimpleNamespace(get=lambda name: manifest)
    monkeypatch.setattr("hfl.models.registry.ModelRegistry", lambda: registry)
    monkeypatch.setattr(main, "get_model_type", lambda m: ModelType.LLM)
    loaded = []
    monkeypatch.setattr(
        "hfl.engine.selector.select_engine", lambda *a, **k: loaded.append(a) or None
    )

    result = CliRunner().invoke(main.app, ["run", "m"])
    assert result.exit_code == 1
    assert "HFL_MEMORY_BUDGET" in result.stdout
    assert loaded == [], "an engine was created for a model that cannot fit"


def test_tray_does_not_preload_a_model_that_cannot_fit(memory, model_of, monkeypatch, caplog):
    from hfl.tray import controller as tray

    manifest = model_of(60)
    monkeypatch.setattr(
        "hfl.models.registry.ModelRegistry", lambda: SimpleNamespace(get=lambda n: manifest)
    )
    created = []
    monkeypatch.setattr("hfl.engine.selector.select_engine", lambda *a, **k: created.append(a))

    ctl = tray.TrayServerController.__new__(tray.TrayServerController)
    ctl.model = "m"
    ctl._preload_model()
    assert created == []
    assert "HFL_MEMORY_BUDGET" in caplog.text


def test_serve_preload_refuses_before_loading(memory, model_of, monkeypatch, temp_config):
    from typer.testing import CliRunner

    from hfl.cli import main

    manifest = model_of(60)
    monkeypatch.setattr(
        "hfl.models.registry.ModelRegistry", lambda: SimpleNamespace(get=lambda n: manifest)
    )
    created = []
    monkeypatch.setattr("hfl.engine.selector.select_engine", lambda *a, **k: created.append(a))
    started = []
    monkeypatch.setattr(main, "start_server", lambda **k: started.append(k), raising=False)
    monkeypatch.setattr("hfl.api.server.start_server", lambda **k: started.append(k))

    result = CliRunner().invoke(main.app, ["serve", "--model", "m", "--host", "127.0.0.1"])
    assert result.exit_code == 1, result.stdout
    assert "HFL_MEMORY_BUDGET" in result.stdout
    assert created == [] and started == []


class TestPsShowsMemory:
    PAYLOAD = {
        "models": [{"name": "a", "size": 9 * GB, "size_vram": 9 * GB, "digest": "sha256:x"}],
        "memory": {
            "total_bytes": 128 * GB,
            "in_use_bytes": 70 * GB,
            "in_use_percent": 54.7,
            "budget_percent": 85.0,
            "budget_bytes": int(108.8 * GB),
            "free_within_budget_bytes": int(38.8 * GB),
            "models_bytes": 9 * GB,
        },
    }

    def _run(self, monkeypatch, payload):
        import httpx
        from typer.testing import CliRunner

        from hfl.cli import main

        class Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
        return CliRunner().invoke(main.app, ["ps"], env={"COLUMNS": "200"})

    def test_the_summary_is_printed(self, monkeypatch):
        result = self._run(monkeypatch, self.PAYLOAD)
        assert result.exit_code == 0
        assert "70.0" in result.stdout and "128.0" in result.stdout
        assert "38.8" in result.stdout and "85" in result.stdout

    def test_also_with_no_model_loaded(self, monkeypatch):
        result = self._run(monkeypatch, {**self.PAYLOAD, "models": []})
        assert "128.0" in result.stdout

    def test_the_gpu_line_when_the_server_has_one(self, monkeypatch):
        payload = {**self.PAYLOAD}
        payload["memory"] = {
            **self.PAYLOAD["memory"],
            "gpu": {
                "total_bytes": 24 * GB,
                "in_use_bytes": 6 * GB,
                "in_use_percent": 25.0,
                "free_within_budget_bytes": int(14.4 * GB),
            },
        }
        result = self._run(monkeypatch, payload)
        assert "GPU" in result.stdout and "24.0" in result.stdout and "14.4" in result.stdout

    def test_an_older_server_without_the_summary(self, monkeypatch):
        result = self._run(monkeypatch, {"models": self.PAYLOAD["models"]})
        assert result.exit_code == 0

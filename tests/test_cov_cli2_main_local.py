# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""CLI commands that work on this machine or ask the Hub: ``search``,
``inspect``, ``alias``, ``login``/``logout``, ``train``, ``check``,
``debug``, ``config``, ``help``, ``discover``, ``start``, ``recommend``,
``compliance-dashboard``, ``draft-recommend`` and ``sessions``.

The Hub, the trainers and the hardware probes are fakes; the registry is
a real one in a temporary HFL home (``temp_config``).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hfl.cli import main
from hfl.cli.main import app

runner = CliRunner()


def _manifest(tmp: Path, name="m-q4", **extra):
    from hfl.models.manifest import ModelManifest

    path = tmp / f"{name}.gguf"
    path.write_bytes(b"GGUF")
    return ModelManifest(
        name=name,
        repo_id=extra.pop("repo_id", "org/m"),
        local_path=str(path),
        format="gguf",
        size_bytes=4,
        **extra,
    )


def _register(temp_config, manifest):
    from hfl.models.registry import ModelRegistry

    ModelRegistry().add(manifest)
    return manifest


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------


def _model(repo, gguf=True, downloads=100):
    m = MagicMock()
    m.id = repo
    m.downloads = downloads
    m.likes = 1
    m.pipeline_tag = "text-generation"
    m.siblings = [MagicMock(rfilename="m-Q4_K_M.gguf")] if gguf else []
    return m


def _search(args, results=None, error=None, keys=("q",)):
    pressed = iter(keys)
    picked: list[str] = []
    with patch("huggingface_hub.HfApi") as api_class:
        api = MagicMock()
        if error is not None:
            api.list_models.side_effect = error
        else:
            api.list_models.side_effect = lambda **kw: list(results)
        api_class.return_value = api
        with patch.object(main, "get_key", lambda: next(pressed)):
            with patch.object(main, "_pull_selected_model", lambda m: picked.append(m.id)):
                result = runner.invoke(app, ["search", *args], terminal_width=200)
    return result, api, picked


class TestSearch:
    def test_offline_says_offline(self, temp_config):
        result, _, _ = _search(["qwen3"], error=ConnectionRefusedError("refused"))
        assert result.exit_code == 1
        assert "you appear to be offline" in result.stdout

    def test_other_hub_errors_say_the_error(self, temp_config):
        result, _, _ = _search(["qwen3"], error=ValueError("bad sort"))
        assert result.exit_code == 1
        assert "Error searching: bad sort" in result.stdout

    def test_gguf_filter_leaves_nothing(self, temp_config):
        result, _, _ = _search(["qwen3", "--gguf"], [_model("org/Qwen3-4B", gguf=False)])
        assert result.exit_code == 0
        assert "No GGUF models found for: 'qwen3'" in result.stdout

    def test_size_filters_leave_nothing(self, temp_config):
        result, _, _ = _search(
            ["qwen3", "--min-params", "10", "--max-params", "20"],
            [_model("org/Qwen3-4B-GGUF"), _model("org/Qwen3-32B-GGUF"), _model("org/no-size")],
        )
        assert result.exit_code == 0
        assert "No models <20.0B and >10.0B found for: 'qwen3'" in result.stdout

    def test_min_params_alone(self, temp_config):
        result, _, _ = _search(
            ["qwen3", "--min-params", "10"],
            [_model("org/Qwen3-4B-GGUF"), _model("org/Qwen3-32B-GGUF")],
        )
        assert "Qwen3-32B-GGUF" in result.stdout
        assert "Qwen3-4B-GGUF" not in result.stdout

    def test_min_params_alone_leaving_nothing(self, temp_config):
        result, _, _ = _search(["qwen3", "--min-params", "100"], [_model("org/Qwen3-4B-GGUF")])
        assert "No models >100.0B found" in result.stdout

    def test_max_params_alone_leaving_nothing(self, temp_config):
        result, _, _ = _search(["qwen3", "--max-params", "1"], [_model("org/Qwen3-4B-GGUF")])
        assert "No models <1.0B found" in result.stdout
        assert " and " not in result.stdout

    def test_an_explicit_size_wins_over_the_one_read_in_the_query(self, temp_config):
        result, api, _ = _search(
            ["coding 7b", "--max-params", "40"],
            [_model("org/Coder-32B-GGUF", downloads=5), _model("org/Coder-7B-GGUF", downloads=9)],
        )
        assert result.exit_code == 0
        assert "Coder-32B-GGUF" in result.stdout  # 32B passes --max-params 40
        # Several searches (one per name of the task) merged into one list.
        assert api.list_models.call_count > 1
        assert result.stdout.index("Coder-7B-GGUF") < result.stdout.index("Coder-32B-GGUF")

    def test_a_number_not_on_the_page_shows_it_again(self, temp_config):
        result, _, picked = _search(
            ["qwen3"], [_model("org/A-GGUF"), _model("org/B-GGUF")], keys=("7", "1")
        )
        assert picked == ["org/B-GGUF"]

    def test_an_empty_list_shows_only_the_hint(self, capsys):
        main._page_through([], "qwen3", 10)
        out = capsys.readouterr().out
        assert "hfl pull" in out


# ----------------------------------------------------------------------
# inspect / alias
# ----------------------------------------------------------------------


class TestInspectAndAlias:
    def test_inspect_shows_the_license_terms(self, temp_config, temp_dir):
        _register(
            temp_config,
            _manifest(
                temp_dir,
                license="llama3",
                license_url="https://example.org/terms",
                gated=True,
                license_restrictions=["no-commercial", "attribution"],
                license_accepted_at="2026-09-01T10:00:00",
            ),
        )
        result = runner.invoke(app, ["inspect", "m-q4"], terminal_width=200)
        assert result.exit_code == 0
        out = result.stdout
        assert "https://example.org/terms" in out
        assert "Yes (required acceptance on HF)" in out
        assert "- no-commercial" in out and "- attribution" in out
        assert "2026-09-01" in out and "10:00" not in out

    def test_alias_the_registry_refuses(self, temp_config, temp_dir, monkeypatch):
        from hfl.models.registry import ModelRegistry

        _register(temp_config, _manifest(temp_dir))
        monkeypatch.setattr(ModelRegistry, "set_alias", lambda self, name, alias: False)
        result = runner.invoke(app, ["alias", "m-q4", "short"])
        assert result.exit_code == 1
        assert "Error assigning alias" in result.stdout


# ----------------------------------------------------------------------
# login / logout
# ----------------------------------------------------------------------


class TestLogin:
    def test_with_a_token(self, monkeypatch):
        import huggingface_hub

        calls = []
        monkeypatch.setattr(huggingface_hub, "login", lambda **kw: calls.append(kw))
        monkeypatch.setattr(huggingface_hub, "whoami", lambda: {"name": "gabriel"})
        result = runner.invoke(app, ["login", "--token", "hf_x"])
        assert result.exit_code == 0
        assert calls == [{"token": "hf_x", "add_to_git_credential": False}]
        assert "Authenticated as: gabriel" in result.stdout

    def test_interactive_says_where_to_get_a_token(self, monkeypatch):
        import huggingface_hub

        calls = []
        monkeypatch.setattr(huggingface_hub, "login", lambda **kw: calls.append(kw))
        monkeypatch.setattr(huggingface_hub, "whoami", lambda: {"name": "g"})
        result = runner.invoke(app, ["login"])
        assert result.exit_code == 0
        assert "huggingface.co/settings/tokens" in result.stdout
        assert calls == [{"add_to_git_credential": False}]

    def test_a_bad_token_is_exit_1(self, monkeypatch):
        import huggingface_hub

        def refuse(**kw):
            raise ValueError("Invalid token")

        monkeypatch.setattr(huggingface_hub, "login", refuse)
        result = runner.invoke(app, ["login", "-t", "nope"])
        assert result.exit_code == 1
        assert "Error authenticating: Invalid token" in result.stdout

    def test_logout(self, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "logout", lambda: None)
        result = runner.invoke(app, ["logout"])
        assert result.exit_code == 0 and "Token removed" in result.stdout

    def test_logout_failure_is_a_warning(self, monkeypatch):
        import huggingface_hub

        def fail():
            raise RuntimeError("not logged in")

        monkeypatch.setattr(huggingface_hub, "logout", fail)
        result = runner.invoke(app, ["logout"])
        assert result.exit_code == 0
        assert "Warning: not logged in" in result.stdout


# ----------------------------------------------------------------------
# train (fake trainers: mlx-lm and Transformers + PEFT are never run)
# ----------------------------------------------------------------------


class _TrainingError(Exception):
    pass


@dataclass
class _Trainer:
    """A stand-in for ``hfl.training.mlx_lora`` / ``hf_lora``."""

    why_not: str | None = None
    problem: str | None = None
    run_error: BaseException | None = None
    fuse_error: BaseException | None = None
    events: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    TrainingError = _TrainingError

    def available(self):
        return self.why_not

    def check_name(self, name):
        if "/" in name:
            raise _TrainingError(f"bad name {name}")

    def trainable(self, base):
        return self.problem

    def prepare_data(self, data, folder):
        self.calls.append(("prepare", data, folder))
        return types.SimpleNamespace(train=90, valid=10, format="chat")

    def Options(self, **kw):  # noqa: N802 - mirrors the dataclass it replaces
        return kw

    def command(self, path, prepared, adapter, options):
        return ["train", path, str(adapter), options]

    def run(self, cmd, log, show):
        self.calls.append(("run", cmd))
        for event in self.events:
            show(event)
        if self.run_error:
            raise self.run_error

    def register(self, base, target, adapter, log=None):
        self.calls.append(("register", target))
        return types.SimpleNamespace(local_path=str(adapter))

    def fuse(self, base, adapter, folder, log, dequantize):
        if self.fuse_error:
            raise self.fuse_error
        self.calls.append(("fuse", folder.name, dequantize))

    def register_fused(self, base, name, folder):
        self.calls.append(("register_fused", name))

    def to_gguf(self, base, name, folder, quant):
        self.calls.append(("to_gguf", name, quant))


@pytest.fixture
def trainers(monkeypatch, tmp_path):
    import hfl.config
    import hfl.core.container as container
    import hfl.training as training

    mlx, hf = _Trainer(), _Trainer()
    # Loaded first so the package attributes exist to be replaced.
    import hfl.training.hf_lora  # noqa: F401
    import hfl.training.mlx_lora  # noqa: F401

    monkeypatch.setattr(training, "mlx_lora", mlx)
    monkeypatch.setattr(training, "hf_lora", hf)
    models = {"base": types.SimpleNamespace(name="base", local_path=str(tmp_path / "base"))}
    registry = types.SimpleNamespace(get=lambda name: models.get(name))
    monkeypatch.setattr(container, "get_registry", lambda: registry)
    monkeypatch.setattr(hfl.config.config, "home_dir", tmp_path / "home")
    return types.SimpleNamespace(mlx=mlx, hf=hf, models=models, home=tmp_path / "home")


def _train(*args):
    return runner.invoke(app, ["train", *args], terminal_width=200)


class TestTrain:
    def test_unknown_backend_is_a_usage_error(self, trainers):
        result = _train("base", "--data", "d.jsonl", "--backend", "tpu")
        assert result.exit_code == 2
        assert "auto, mlx or transformers, not tpu" in result.stdout

    def test_a_backend_that_is_missing_says_why(self, trainers):
        trainers.hf.why_not = "pip install 'hfl[train]'"
        result = _train("base", "--data", "d.jsonl", "--backend", "transformers")
        assert result.exit_code == 1
        assert "pip install 'hfl[train]'" in result.stdout

    def test_unknown_model(self, trainers):
        result = _train("ghost", "--data", "d.jsonl", "--backend", "mlx")
        assert result.exit_code == 1 and "Model not found: ghost" in result.stdout

    @pytest.mark.parametrize(
        ("setup", "args", "said"),
        [
            ("problem", [], "quantized base"),
            ("exists", [], "base-lora already exists"),
            ("badname", ["--name", "a/b"], "bad name a/b"),
        ],
    )
    def test_refusals_before_training(self, trainers, setup, args, said):
        if setup == "problem":
            trainers.mlx.problem = "quantized base"
        if setup == "exists":
            trainers.models["base-lora"] = object()
        result = _train("base", "--data", "d.jsonl", "--backend", "mlx", *args)
        assert result.exit_code == 1
        assert said in result.stdout
        assert not any(c[0] == "run" for c in trainers.mlx.calls)

    def test_resume_accepts_an_existing_target(self, trainers):
        trainers.models["base-lora"] = object()
        result = _train("base", "--data", "d.jsonl", "--backend", "mlx", "--resume")
        assert result.exit_code == 0, result.stdout
        run = next(c for c in trainers.mlx.calls if c[0] == "run")
        assert run[1][3]["resume"] is True

    def test_auto_picks_mlx_when_available_and_reports_progress(self, trainers):
        trainers.mlx.events = [
            {"iteration": 10},
            {"iteration": 20, "train_loss": 1.23456, "tokens_per_sec": 512.4},
            {"iteration": 20, "val_loss": 1.5, "peak_memory_gb": 7.25},
        ]
        result = _train("base", "--data", "d.jsonl", "--iters", "20")
        assert result.exit_code == 0, result.stdout
        out = result.stdout
        assert "Data: 90 rows to train on, 10 to validate (chat)" in out
        assert "iteration 10/20 · loss –" in out
        assert "iteration 20/20 · loss 1.235 · 512 tok/s" in out
        assert "val 1.500 · 512 tok/s · 7.2 GB" in out
        assert "Trained: base-lora" in out
        assert ("register", "base-lora") in trainers.mlx.calls
        prepare = trainers.mlx.calls[0]
        assert prepare[2] == trainers.home / "training" / "base-lora" / "data"

    def test_ctrl_c_says_how_to_resume(self, trainers):
        trainers.mlx.run_error = KeyboardInterrupt()
        result = _train("base", "--data", "d.jsonl", "--backend", "mlx")
        assert result.exit_code == 130
        assert "--resume" in result.stdout

    def test_a_training_failure_points_at_the_log(self, trainers):
        trainers.mlx.run_error = _TrainingError("loss is nan")
        result = _train("base", "--data", "d.jsonl", "--backend", "mlx")
        assert result.exit_code == 1
        assert "loss is nan" in result.stdout and "train-base-lora.log" in result.stdout

    def test_fuse_and_gguf(self, trainers):
        result = _train("base", "--data", "d.jsonl", "--backend", "mlx", "--gguf", "q4_k_m")
        assert result.exit_code == 0, result.stdout
        assert ("fuse", "base-lora-fused", True) in trainers.mlx.calls
        assert ("register_fused", "base-lora-fused") in trainers.mlx.calls
        assert ("to_gguf", "base-lora-gguf", "Q4_K_M") in trainers.mlx.calls
        assert "Fused model: base-lora-fused" in result.stdout
        assert "GGUF model: base-lora-gguf" in result.stdout

    def test_fuse_only(self, trainers):
        result = _train("base", "--data", "d.jsonl", "--backend", "mlx", "--fuse")
        assert result.exit_code == 0
        assert ("fuse", "base-lora-fused", False) in trainers.mlx.calls
        assert not any(c[0] == "to_gguf" for c in trainers.mlx.calls)

    def test_a_failed_fuse_keeps_the_adapter_and_is_exit_1(self, trainers):
        trainers.mlx.fuse_error = RuntimeError("out of disk")
        result = _train("base", "--data", "d.jsonl", "--backend", "mlx", "--fuse")
        assert result.exit_code == 1
        assert "Trained: base-lora" in result.stdout and "out of disk" in result.stdout

    def test_transformers_merges_and_exports(self, trainers):
        trainers.mlx.why_not = "not Apple Silicon"
        result = _train("base", "--data", "d.jsonl", "--gguf", "q8_0")
        assert result.exit_code == 0, result.stdout
        assert ("register", "base-lora") in trainers.hf.calls
        assert ("to_gguf", "base-lora-gguf", "Q8_0") in trainers.hf.calls
        assert trainers.mlx.calls == []

    def test_transformers_without_gguf(self, trainers):
        result = _train("base", "--data", "d.jsonl", "--backend", "transformers")
        assert result.exit_code == 0
        assert not any(c[0] == "to_gguf" for c in trainers.hf.calls)

    def test_transformers_merge_failure(self, trainers):
        def broken(*a):
            raise RuntimeError("merge failed")

        trainers.hf.register = broken
        result = _train("base", "--data", "d.jsonl", "--backend", "transformers")
        assert result.exit_code == 1 and "merge failed" in result.stdout


# ----------------------------------------------------------------------
# check / debug / config
# ----------------------------------------------------------------------


@pytest.fixture
def fixed_report(monkeypatch):
    from hfl.cli.commands import doctor

    report = doctor.DoctorReport(
        llama_cpp_available=True,
        llama_cpp_build_features={"gpu_offload": True},
        llama_server="/opt/[bin]/llama-server",
        nvidia_devices=["NVIDIA [L4]"],
    )
    monkeypatch.setattr(doctor, "build_report", lambda: report)
    return report


def _availability(monkeypatch, **values):
    import hfl.engine.dependency_check as dc

    monkeypatch.setattr(dc, "check_engine_availability", lambda: values)


class TestCheck:
    def test_everything_present(self, temp_config, fixed_report, monkeypatch):
        _availability(monkeypatch, transformers=True, soundfile=True, torchaudio=True)
        result = runner.invoke(app, ["check"], terminal_width=200)
        assert result.exit_code == 0
        out = result.stdout
        assert "llama-server /opt/[bin]/llama-server" in out  # markup escaped
        assert "NVIDIA: NVIDIA [L4]" in out
        assert "Bark (via transformers)" in out
        assert "✓ soundfile" in out and "✓ torchaudio" in out
        assert "Registry: 0 models" in out
        assert "Models dir:" in out and "not created" not in out

    def test_a_broken_registry_and_no_models_dir(self, temp_config, fixed_report, monkeypatch):
        import shutil

        import hfl.models.registry as registry_module

        _availability(monkeypatch)

        def broken():
            raise RuntimeError("registry corrupt")

        monkeypatch.setattr(registry_module, "ModelRegistry", broken)
        shutil.rmtree(temp_config.models_dir)
        result = runner.invoke(app, ["check"], terminal_width=200)
        assert result.exit_code == 0
        assert "Registry: registry corrupt" in result.stdout
        assert "Models dir: not created" in result.stdout
        assert "Bark: requires transformers" in result.stdout


class TestDebug:
    def _versions(self, monkeypatch, installed):
        import importlib.metadata

        def version(name):
            if name in installed:
                return installed[name]
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", version)

    def test_installed_extras_cuda_and_server(self, fixed_report, monkeypatch):
        import hfl.engine.llama_server as ls

        self._versions(monkeypatch, {"typer": "0.12", "torch": "2.5.0"})
        monkeypatch.setattr(ls, "binary", lambda: "/usr/bin/llama-server")
        _availability(monkeypatch, torch_cuda=True)
        torch = types.ModuleType("torch")
        torch.version = types.SimpleNamespace(cuda="12.4")
        torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: 90100))
        monkeypatch.setitem(sys.modules, "torch", torch)
        result = runner.invoke(app, ["debug"], terminal_width=200)
        assert result.exit_code == 0
        out = result.stdout
        assert "torch: 2.5.0" in out
        assert "transformers: not installed" in out
        assert "llama-server: /usr/bin/llama-server" in out
        assert "CUDA Version: 12.4" in out and "cuDNN: 90100" in out
        assert "NVIDIA: NVIDIA [L4]" in out

    def test_cuda_details_that_cannot_be_read_and_no_psutil(self, fixed_report, monkeypatch):
        import hfl.engine.llama_server as ls

        self._versions(monkeypatch, {})
        monkeypatch.setattr(ls, "binary", lambda: None)
        _availability(monkeypatch, torch_cuda=True)
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))  # no .version
        monkeypatch.setitem(sys.modules, "psutil", None)
        result = runner.invoke(app, ["debug"], terminal_width=200)
        assert result.exit_code == 0
        assert "llama-server: not installed" in result.stdout
        assert "CUDA Version" not in result.stdout
        assert "Memory" not in result.stdout


def test_config_shows_a_configured_token_without_its_value(temp_config, monkeypatch):
    monkeypatch.setattr(temp_config, "hf_token", "hf_secret_value")
    result = runner.invoke(app, ["config"], terminal_width=200)
    assert result.exit_code == 0
    assert "Token: Configured" in result.stdout
    assert "hf_secret_value" not in result.stdout


# ----------------------------------------------------------------------
# help
# ----------------------------------------------------------------------


class TestHelp:
    def test_general_help_lists_the_common_commands(self):
        result = runner.invoke(app, ["help"], terminal_width=200)
        assert result.exit_code == 0
        for command in ("pull <model>", "run <model>", "serve [--tray]", "launch claude"):
            assert f"hfl {command}" in result.stdout

    def test_extras_with_an_installed_and_a_missing_module(self, monkeypatch):
        import importlib.util

        monkeypatch.setattr(
            main, "_declared_extras", lambda: (["mcp", "llama"], {"mcp": ["mcp>=1.0"]})
        )
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: (
                object() if name == "mcp" else (None if name == "llama_cpp" else real(name))
            ),
        )
        result = runner.invoke(app, ["help", "--extras"], terminal_width=200)
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        mcp_row = next(line for line in lines if line.startswith("│ mcp"))
        llama_row = next(line for line in lines if line.startswith("│ llama"))
        assert "installed" in mcp_row.lower() and "not installed" not in mcp_row.lower()
        assert "not installed" in llama_row.lower()
        assert "Packages: mcp>=1.0" in result.stdout
        assert "Packages: —" in result.stdout

    def test_the_all_extra_has_no_status(self, monkeypatch):
        monkeypatch.setattr(main, "_declared_extras", lambda: (["all"], {}))
        result = runner.invoke(app, ["help", "--extras"], terminal_width=200)
        row = next(line for line in result.stdout.splitlines() if line.startswith("│ all"))
        assert "not installed" not in row.lower() and "—" in row


# ----------------------------------------------------------------------
# discover / recommend / draft-recommend / compliance-dashboard
# ----------------------------------------------------------------------


def _entry(repo, **over):
    values = {
        "repo_id": repo,
        "family": "qwen",
        "parameter_estimate_b": 7.0,
        "quantization": "Q4_K_M",
        "likes": 1234,
        "downloads": 56789,
        "locally_available": False,
    }
    values.update(over)
    return types.SimpleNamespace(**values)


class TestDiscover:
    @pytest.fixture
    def hub(self, monkeypatch):
        import hfl.api.routes_discover as routes_discover
        import hfl.hub.discovery as discovery

        cache = MagicMock()
        cache.get.return_value = None
        state = types.SimpleNamespace(cache=cache, searched=[], answer=[], error=None)

        def search_hub(q):
            state.searched.append(q)
            if state.error:
                raise state.error
            return state.answer

        monkeypatch.setattr(routes_discover, "_build_cache", lambda: cache)
        monkeypatch.setattr(routes_discover, "_annotate_local_availability", lambda e: None)
        monkeypatch.setattr(discovery, "search_hub", search_hub)
        return state

    def test_lists_the_hub_and_caches_it(self, hub):
        hub.answer = [_entry("Qwen/Qwen2.5-7B", locally_available=True), _entry("x/y", family=None)]
        result = runner.invoke(
            app, ["discover", "qwen", "--family", "qwen", "--min-likes", "5"], terminal_width=200
        )
        assert result.exit_code == 0, result.stdout
        assert "Qwen/Qwen2.5-7B" in result.stdout and "1,234" in result.stdout
        assert "✓" in result.stdout
        assert "(cached)" not in result.stdout
        query = hub.searched[0]
        assert (query.q, query.family, query.min_likes) == ("qwen", "qwen", 5)
        hub.cache.put.assert_called_once_with(query, hub.answer)

    def test_a_cached_answer_skips_the_hub(self, hub):
        hub.cache.get.return_value = [_entry("a/b")]
        result = runner.invoke(app, ["discover"], terminal_width=200)
        assert "(cached)" in result.stdout and hub.searched == []

    def test_refresh_ignores_the_cache(self, hub):
        hub.cache.get.return_value = [_entry("a/b")]
        hub.answer = [_entry("c/d")]
        result = runner.invoke(app, ["discover", "--refresh"], terminal_width=200)
        assert "c/d" in result.stdout and len(hub.searched) == 1
        hub.cache.get.assert_not_called()

    def test_nothing_found(self, hub):
        result = runner.invoke(app, ["discover", "zzz"])
        assert result.exit_code == 0 and "No matching models found." in result.stdout

    def test_hub_failure_is_exit_1(self, hub):
        hub.error = RuntimeError("503")
        result = runner.invoke(app, ["discover", "zzz"])
        assert result.exit_code == 1 and "Hub unavailable: 503" in result.stdout
        hub.cache.put.assert_not_called()


def _profile(**over):
    from hfl.hub.hw_profile import HardwareProfile

    values = {
        "os": "linux",
        "arch": "x86_64",
        "system_ram_gb": 32.0,
        "gpu_kind": "cuda",
        "gpu_vram_gb": None,
        "has_mlx": False,
        "has_cuda": True,
        "has_rocm": False,
    }
    values.update(over)
    return HardwareProfile(**values)


class TestRecommend:
    @pytest.fixture
    def hub(self, monkeypatch):
        import hfl.hub.hw_profile as hw_profile
        import hfl.hub.recommend as recommend

        state = types.SimpleNamespace(recs=[], error=None, kwargs=None)

        def recommend_models(**kw):
            state.kwargs = kw
            if state.error:
                raise state.error
            return state.recs

        monkeypatch.setattr(hw_profile, "get_hw_profile", lambda: _profile())
        monkeypatch.setattr(recommend, "recommend_models", recommend_models)
        return state

    def test_an_unknown_task_is_refused(self, hub):
        result = runner.invoke(app, ["recommend", "--task", "poetry"])
        assert result.exit_code == 1 and "task must be one of" in result.stdout
        assert hub.kwargs is None

    def test_offline(self, hub):
        hub.error = ConnectionRefusedError("refused")
        result = runner.invoke(app, ["recommend"])
        assert result.exit_code == 1 and "offline" in result.stdout

    def test_other_errors(self, hub):
        hub.error = ValueError("bad family")
        result = runner.invoke(app, ["recommend", "--family", "x"])
        assert result.exit_code == 1 and "Error resolving model: bad family" in result.stdout

    def test_nothing_fits(self, hub):
        result = runner.invoke(app, ["recommend"], terminal_width=200)
        assert result.exit_code == 0
        assert "VRAM=n/aGB" in result.stdout
        assert "No models fit this hardware" in result.stdout

    def test_a_table_of_recommendations(self, hub, monkeypatch):
        monkeypatch.setattr(main.console, "width", 220)
        hub.recs = [
            types.SimpleNamespace(
                repo_id="Qwen/Qwen2.5-Coder-7B",
                family=None,
                quantization="Q4_K_M",
                estimated_vram_gb=5.04,
                score=0.876,
                reasoning=["fits VRAM", "popular", "third reason not shown"],
            )
        ]
        result = runner.invoke(
            app, ["recommend", "-t", "code", "-n", "1", "-q", "Q4_K_M"], terminal_width=220
        )
        assert result.exit_code == 0
        assert hub.kwargs["task"] == "code" and hub.kwargs["top_n"] == 1
        assert "Top 1 for code" in result.stdout
        assert "5.0 GB" in result.stdout and "0.88" in result.stdout
        assert "fits VRAM; popular" in result.stdout
        assert "third reason" not in result.stdout


class TestDraftRecommend:
    def _pick(self, monkeypatch, pick):
        import hfl.hub.draft_picker as draft_picker

        monkeypatch.setattr(draft_picker, "pick_draft_for", lambda model, max_ratio: pick)

    def test_none_found(self, monkeypatch):
        self._pick(monkeypatch, None)
        result = runner.invoke(app, ["draft-recommend", "org/m"])
        assert result.exit_code == 1 and "No draft candidate found" in result.stdout

    def test_full_pick(self, monkeypatch):
        self._pick(
            monkeypatch,
            types.SimpleNamespace(
                repo_id="meta-llama/Llama-3.2-1B",
                family="llama",
                parameter_estimate_b=1.2,
                quantization="Q8_0",
                rationale="canonical small sibling",
            ),
        )
        result = runner.invoke(app, ["draft-recommend", "org/m"], terminal_width=200)
        assert result.exit_code == 0
        assert "size:     ~1.2B" in result.stdout and "quant:    Q8_0" in result.stdout

    def test_pick_without_size_or_quant(self, monkeypatch):
        self._pick(
            monkeypatch,
            types.SimpleNamespace(
                repo_id="a/b",
                family=None,
                parameter_estimate_b=None,
                quantization=None,
                rationale="r",
            ),
        )
        result = runner.invoke(app, ["draft-recommend", "org/m"])
        assert result.exit_code == 0
        assert "family:   -" in result.stdout
        assert "size:" not in result.stdout and "quant:" not in result.stdout


class TestComplianceDashboard:
    def _snapshot(self, monkeypatch, **over):
        import hfl.api.routes_compliance as routes_compliance

        snapshot = {
            "total_models": 3,
            "has_hf_token": False,
            "by_risk": {"low": 2, "high": 1},
            "by_license": {},
            "gated_without_token": [],
            "missing_license": [],
            "eu_ai_act_warnings": [],
        }
        snapshot.update(over)
        monkeypatch.setattr(routes_compliance, "_build_compliance_dashboard", lambda: snapshot)

    def test_a_clean_registry(self, monkeypatch):
        self._snapshot(monkeypatch)
        result = runner.invoke(app, ["compliance-dashboard"], terminal_width=200)
        assert result.exit_code == 0
        assert "total=3" in result.stdout
        assert "By license id" not in result.stdout
        assert "EU AI Act" not in result.stdout

    def test_every_section(self, monkeypatch):
        self._snapshot(
            monkeypatch,
            by_license={"apache-2.0": 2, "llama3": 1},
            gated_without_token=["meta/llama"],
            missing_license=["anon/model"],
            eu_ai_act_warnings=[{"model": "x", "license": "y", "reason": "GPAI"}],
        )
        result = runner.invoke(app, ["compliance-dashboard"], terminal_width=200)
        out = result.stdout
        assert "apache-2.0" in out
        assert "Gated models without HF_TOKEN" in out and "- meta/llama" in out
        assert "without a declared license" in out and "- anon/model" in out
        assert "- x (y): GPAI" in out


# ----------------------------------------------------------------------
# start (only what tests/test_cli_start.py leaves out)
# ----------------------------------------------------------------------


class TestStart:
    def test_a_model_here_and_no_chat_opens_nothing(self, monkeypatch):
        opened = []
        monkeypatch.setattr(
            "hfl.hub.hw_profile.get_hw_profile",
            lambda: types.SimpleNamespace(
                os="linux", arch="x86_64", system_ram_gb=8.0, gpu_kind="none"
            ),
        )
        monkeypatch.setattr(main, "_a_chat_model", lambda: "chat")
        monkeypatch.setattr(main, "run", lambda **kw: opened.append(kw))
        result = runner.invoke(app, ["start", "--no-chat"])
        assert result.exit_code == 0 and opened == []

    def test_a_chat_model_is_a_text_model_by_alias_or_name(self, temp_config, temp_dir):
        assert main._a_chat_model() is None
        _register(temp_config, _manifest(temp_dir, name="voice", model_type="tts"))
        assert main._a_chat_model() is None
        _register(temp_config, _manifest(temp_dir, name="chatty", alias="c"))
        assert main._a_chat_model() == "c"

    def test_a_model_without_a_type_counts_as_text(self, temp_config, temp_dir):
        _register(temp_config, _manifest(temp_dir, name="legacy"))
        assert main._a_chat_model() == "legacy"

    def test_offers_skip_names_the_hub_cannot_answer(self, monkeypatch):
        def find(name):
            if name == "qwen2.5:1.5b":
                raise ConnectionRefusedError("offline")
            return types.SimpleNamespace(repo_id=f"org/{name}", size_bytes=1)

        monkeypatch.setattr("hfl.hub.shortname.find", find)
        offers = main._first_model_offers(8.0)
        assert [name for name, _ in offers] == ["qwen2.5:0.5b"]

    def test_pick_asks_again_until_a_valid_number(self, monkeypatch):
        answers = iter(["9", "x", "2"])
        monkeypatch.setattr(main.console, "input", lambda prompt="": next(answers))
        assert main._pick(3) == 2
        monkeypatch.setattr(main.console, "input", lambda prompt="": "")
        assert main._pick(3) == 1


# ----------------------------------------------------------------------
# sessions
# ----------------------------------------------------------------------


class TestSessions:
    @pytest.fixture
    def saved(self, temp_config, monkeypatch):
        from hfl.core import sessions

        monkeypatch.setattr(sessions, "sessions_dir", lambda: _dir(temp_config.home_dir))
        return sessions

    def test_empty(self, saved):
        result = runner.invoke(app, ["sessions", "list"])
        assert result.exit_code == 0 and "No saved sessions yet" in result.stdout

    def test_list_show_rm(self, saved):
        saved.save_session(
            saved.ChatSession(
                name="work",
                model="qwen",
                messages=[
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "tool", "content": "42"},
                ],
            )
        )
        listed = runner.invoke(app, ["sessions", "list"], terminal_width=200)
        assert "work" in listed.stdout and "qwen" in listed.stdout and " 4 " in listed.stdout
        shown = runner.invoke(app, ["sessions", "show", "work"], terminal_width=200)
        assert shown.exit_code == 0
        for line in ("system: be brief", "user: hi", "assistant: hello", "tool: 42"):
            assert line in shown.stdout
        removed = runner.invoke(app, ["sessions", "rm", "work"])
        assert removed.exit_code == 0 and "Deleted session 'work'" in removed.stdout
        assert runner.invoke(app, ["sessions", "rm", "work"]).exit_code == 1

    def test_show_missing(self, saved):
        result = runner.invoke(app, ["sessions", "show", "ghost"])
        assert result.exit_code == 1 and "No session named 'ghost'" in result.stdout


def _dir(home: Path) -> Path:
    path = home / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_cli_main_hands_the_child_guard_flag_to_the_guard(monkeypatch):
    from hfl.engine import _child_guard
    from hfl.utils.self_exec import CHILD_GUARD_FLAG

    seen = []
    monkeypatch.setattr(sys, "argv", ["hfl", CHILD_GUARD_FLAG, "a", "b"])
    monkeypatch.setattr(_child_guard, "main", lambda argv: seen.append(argv) or 7)
    with pytest.raises(SystemExit) as caught:
        main.cli_main()
    assert caught.value.code == 7
    assert seen == [["hfl", "a", "b"]]

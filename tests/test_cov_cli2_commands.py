# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl.cli.commands``: the shared helpers, ``hfl doctor``'s probes,
``hfl install llama-server`` and ``hfl launch``'s server handling.

Every probe runs against fakes (``sys.modules`` entries, a fake
``/sys/class/drm``, stub servers): no GPU, network or real process.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from hfl.cli.commands import _utils, doctor, install, launch

# ----------------------------------------------------------------------
# _utils
# ----------------------------------------------------------------------


class TestUtils:
    def test_show_progress_returns_the_result_and_says_done(self, capsys):
        @_utils.show_progress("working", finished_message="all done")
        def add(a, b):
            return a + b

        assert add(2, b=3) == 5
        assert "all done" in capsys.readouterr().out

    def test_show_progress_without_a_finished_message_is_silent(self, capsys):
        @_utils.show_progress("working")
        def one():
            return 1

        assert one() == 1
        assert "all done" not in capsys.readouterr().out

    def test_get_key_reads_one_char_and_restores_the_terminal(self, monkeypatch):
        import termios
        import tty

        calls: list[tuple] = []
        monkeypatch.setattr(termios, "tcgetattr", lambda fd: ["saved"])
        monkeypatch.setattr(tty, "setraw", lambda fd: calls.append(("raw", fd)))
        monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, old: calls.append(("set", old)))
        fake_stdin = MagicMock()
        fake_stdin.fileno.return_value = 7
        fake_stdin.read.return_value = "q"
        monkeypatch.setattr(sys, "stdin", fake_stdin)

        assert _utils.get_key() == "q"
        assert calls == [("raw", 7), ("set", ["saved"])]

    def test_get_key_restores_the_terminal_even_when_reading_fails(self, monkeypatch):
        import termios
        import tty

        restored: list = []
        monkeypatch.setattr(termios, "tcgetattr", lambda fd: "old")
        monkeypatch.setattr(tty, "setraw", lambda fd: None)
        monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, old: restored.append(old))
        fake_stdin = MagicMock()
        fake_stdin.fileno.return_value = 0
        fake_stdin.read.side_effect = OSError("gone")
        monkeypatch.setattr(sys, "stdin", fake_stdin)

        with pytest.raises(OSError):
            _utils.get_key()
        assert restored == ["old"]

    def test_estimate_model_size_of_an_unreadable_count_is_unknown(self):
        assert _utils.estimate_model_size("sevenB") == "?"
        assert _utils.estimate_model_size(None) == "?"
        assert _utils.estimate_model_size("70B", "Q4_K_M") == "37GB"
        assert _utils.estimate_model_size("1B", "Q8_0") == "0.9GB"

    def test_get_params_value_of_an_unparsable_name_is_none(self, monkeypatch):
        monkeypatch.setattr(_utils, "extract_params_from_name", lambda _id: "manyB")
        assert _utils.get_params_value("org/model-manyB") is None

    def test_get_params_value_reads_the_count(self):
        assert _utils.get_params_value("org/Llama-3-8B") == 8.0
        assert _utils.get_params_value("org/no-size-here") is None


# ----------------------------------------------------------------------
# doctor
# ----------------------------------------------------------------------


class TestDoctorProbes:
    def test_to_dict_is_every_field(self):
        report = doctor.DoctorReport(python_version="3.12.1", nvidia_devices=["A100"])
        data = report.to_dict()
        assert data["python_version"] == "3.12.1"
        assert data["nvidia_devices"] == ["A100"]
        assert set(data) >= {"llama_server", "power_source", "recommendations"}

    def test_llama_cpp_with_offload_reports_it(self, monkeypatch):
        fake = types.ModuleType("llama_cpp")
        fake.llama_supports_gpu_offload = lambda: 1
        monkeypatch.setitem(sys.modules, "llama_cpp", fake)
        assert doctor._probe_llama_cpp() == (True, {"gpu_offload": True})

    def test_llama_cpp_without_the_helper_reports_no_features(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
        assert doctor._probe_llama_cpp() == (True, {})

    def _pynvml(self, names, *, init_error=None, count_error=None, shutdown_error=None):
        fake = types.ModuleType("pynvml")
        fake.shutdowns = 0

        def init():
            if init_error:
                raise init_error

        def count():
            if count_error:
                raise count_error
            return len(names)

        def shutdown():
            fake.shutdowns += 1
            if shutdown_error:
                raise shutdown_error

        fake.nvmlInit = init
        fake.nvmlDeviceGetCount = count
        fake.nvmlDeviceGetHandleByIndex = lambda i: i
        fake.nvmlDeviceGetName = lambda handle: names[handle]
        fake.nvmlShutdown = shutdown
        return fake

    def test_nvidia_names_bytes_and_str(self, monkeypatch):
        fake = self._pynvml([b"NVIDIA L4", "RTX 4090"])
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert doctor._probe_nvidia() == ["NVIDIA L4", "RTX 4090"]
        assert fake.shutdowns == 1

    def test_nvidia_init_failure_is_no_devices(self, monkeypatch):
        fake = self._pynvml(["x"], init_error=RuntimeError("no driver"))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert doctor._probe_nvidia() == []
        assert fake.shutdowns == 0

    def test_nvidia_count_and_shutdown_failures_are_swallowed(self, monkeypatch):
        fake = self._pynvml(
            ["x"], count_error=RuntimeError("lost"), shutdown_error=RuntimeError("also")
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert doctor._probe_nvidia() == []
        assert fake.shutdowns == 1

    def test_nvidia_without_pynvml_asks_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda i: f"GPU{i}",
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
        assert doctor._probe_nvidia() == ["GPU0", "GPU1"]

    def test_torch_without_cuda_is_no_devices(self, monkeypatch):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch)
        assert doctor._probe_nvidia_torch() == []

    def test_torch_that_breaks_is_no_devices(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert doctor._probe_nvidia_torch() == []

    def test_metal_off_darwin_and_off_arm(self, monkeypatch):
        monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
        assert doctor._probe_metal() is False
        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(doctor.platform, "machine", lambda: "x86_64")
        assert doctor._probe_metal() is False

    @pytest.mark.parametrize(
        ("torch_module", "expected"),
        [
            (None, True),  # torch not installed
            ("no-mps", True),
            ("mps-off", False),
            ("mps-on", True),
        ],
    )
    def test_metal_on_apple_silicon(self, monkeypatch, torch_module, expected):
        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
        if torch_module is None:
            fake = None
        else:
            fake = types.ModuleType("torch")
            if torch_module == "no-mps":
                fake.backends = types.SimpleNamespace(mps=None)
            else:
                on = torch_module == "mps-on"
                fake.backends = types.SimpleNamespace(
                    mps=types.SimpleNamespace(is_available=lambda: on)
                )
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert doctor._probe_metal() is expected

    def _drm(self, monkeypatch, tmp_path: Path) -> Path:
        import pathlib

        real = pathlib.Path
        root = tmp_path / "drm"

        def fake_path(*parts):
            if parts == ("/sys/class/drm",):
                return real(root)
            return real(*parts)

        monkeypatch.setattr(pathlib, "Path", fake_path)
        return root

    def test_rocm_without_drm_is_no_cards(self, monkeypatch, tmp_path):
        self._drm(monkeypatch, tmp_path)
        assert doctor._probe_rocm() == []

    def test_rocm_lists_cards_with_their_pci_id(self, monkeypatch, tmp_path):
        root = self._drm(monkeypatch, tmp_path)
        (root / "card0" / "device").mkdir(parents=True)
        (root / "card0" / "device" / "device").write_text("0x744c\n")
        (root / "card1").mkdir()  # no PCI id file
        (root / "card2" / "device" / "device").mkdir(parents=True)  # unreadable: a dir
        (root / "card0-DP-1").mkdir()  # a connector, not a card
        (root / "renderD128").mkdir()
        assert doctor._probe_rocm() == ["card0 (0x744c)", "card1", "card2"]

    def test_probe_optional(self):
        assert doctor._probe_optional("json") is True
        assert doctor._probe_optional("hfl_no_such_module_xyz") is False


class TestDoctorReport:
    def _probes(self, monkeypatch, **over):
        values = {
            "_probe_llama_cpp": (True, {"gpu_offload": True}),
            "_probe_nvidia": [],
            "_probe_metal": False,
            "_probe_rocm": [],
            "_probe_optional": False,
        }
        values.update({k: v for k, v in over.items() if k.startswith("_")})
        for name, value in values.items():
            monkeypatch.setattr(doctor, name, lambda *a, _v=value: _v)
        import hfl.engine.llama_server as ls
        import hfl.engine.power as power

        monkeypatch.setattr(ls, "binary", lambda: over.get("server", "/opt/llama-server"))
        monkeypatch.setattr(power, "power_source_label", lambda: over.get("power", "AC power"))

    def test_battery_and_missing_pieces_are_recommended(self, monkeypatch):
        self._probes(monkeypatch, _probe_llama_cpp=(False, {}), server=None, power="battery")
        report = doctor.build_report()
        text = " ".join(report.recommendations)
        assert "battery" in text
        assert "llama-cpp-python missing" in text
        assert "hfl install llama-server" in text
        assert "No GPU accelerator" in text

    def test_a_complete_install_has_no_recommendations(self, monkeypatch):
        self._probes(monkeypatch, _probe_nvidia=["L4"])
        monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
        report = doctor.build_report()
        assert report.recommendations == []
        assert report.llama_server == "/opt/llama-server"
        assert report.power_source == "AC power"

    def test_apple_silicon_without_mlx_is_told_about_it(self, monkeypatch):
        self._probes(monkeypatch, _probe_metal=True)
        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
        report = doctor.build_report()
        assert any("hfl[mlx]" in r for r in report.recommendations)

    def test_format_report_shows_power_and_offload(self):
        report = doctor.DoctorReport(
            python_version="3.12.0",
            platform_system="Darwin",
            platform_machine="arm64",
            llama_cpp_available=True,
            llama_cpp_build_features={"gpu_offload": False},
            metal_available=True,
            rocm_devices=["card0"],
            vram_gib=24.0,
            recommended_ctx=8192,
            power_source="battery",
        )
        text = doctor.format_report(report)
        assert "GPU offload ✗" in text
        assert "Power source:     battery" in text
        assert "Apple Metal" in text and "AMD ROCm" in text
        assert "24.0 GiB" in text and "8192" in text
        assert "Recommendations" not in text


# ----------------------------------------------------------------------
# install llama-server
# ----------------------------------------------------------------------


@pytest.fixture
def dist(monkeypatch, tmp_path):
    from hfl.engine import llama_server_dist as d

    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    monkeypatch.setattr(d, "bundled_binary", lambda: None)
    monkeypatch.setattr(d, "managed_binary", lambda: None)
    monkeypatch.setattr(d, "variants", lambda: ["cpu", "vulkan"])
    monkeypatch.setattr(d, "default_variant", lambda: "cpu")
    monkeypatch.setattr(d, "size_of", lambda variant: 30_000_000)
    monkeypatch.setattr(d, "install_dir", lambda: tmp_path / "bin")
    return d


class TestInstall:
    def test_one_installed_before_is_kept(self, dist, monkeypatch, capsys):
        monkeypatch.setattr(dist, "managed_binary", lambda: "/hfl/bin/llama-server")
        assert install.install_llama_server(variant=None, assume_yes=True, force=False) == 0
        assert "Already installed: /hfl/bin/llama-server" in capsys.readouterr().out

    def test_no_terminal_and_no_yes_installs_nothing(self, dist, monkeypatch, capsys):
        monkeypatch.setattr(install, "stdin_is_terminal", lambda: False)
        monkeypatch.setattr(dist, "install", MagicMock())
        assert install.install_llama_server(variant=None, assume_yes=False, force=False) == 1
        assert "--yes" in capsys.readouterr().out
        dist.install.assert_not_called()

    def test_declining_installs_nothing(self, dist, monkeypatch):
        import typer

        monkeypatch.setattr(install, "stdin_is_terminal", lambda: True)
        monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
        monkeypatch.setattr(dist, "install", MagicMock())
        assert install.install_llama_server(variant=None, assume_yes=False, force=False) == 1
        dist.install.assert_not_called()

    def test_accepting_installs_and_reports_the_path(self, dist, monkeypatch, capsys, tmp_path):
        import typer

        monkeypatch.setattr(install, "stdin_is_terminal", lambda: True)
        monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
        seen = {}

        def fake_install(variant, progress):
            seen["variant"] = variant
            progress(10)
            return tmp_path / "bin" / "llama-server"

        monkeypatch.setattr(dist, "install", fake_install)
        assert install.install_llama_server(variant="vulkan", assume_yes=False, force=False) == 0
        assert seen["variant"] == "vulkan"
        assert "Installed:" in capsys.readouterr().out

    def test_missing_system_library_names_the_package(self, dist, monkeypatch, capsys):
        def broken(variant, progress):
            raise dist.InstallError(
                "llama-server: error while loading shared libraries: "
                "libgomp.so.1: cannot open shared object file"
            )

        monkeypatch.setattr(dist, "install", broken)
        assert install.install_llama_server(variant=None, assume_yes=True, force=True) == 1
        out = capsys.readouterr().out
        assert "libgomp.so.1" in out
        assert "sudo apt install libgomp1" in out

    def test_other_failures_say_only_the_error(self, dist, monkeypatch, capsys):
        def broken(variant, progress):
            raise dist.InstallError("sha256 mismatch")

        monkeypatch.setattr(dist, "install", broken)
        assert install.install_llama_server(variant=None, assume_yes=True, force=False) == 1
        out = capsys.readouterr().out
        assert "sha256 mismatch" in out
        assert "apt install" not in out

    def test_missing_library_without_a_known_package(self):
        assert install._missing_library("libfoo.so.3: cannot open shared object file") is None
        assert install._missing_library("timeout") is None


# ----------------------------------------------------------------------
# launch
# ----------------------------------------------------------------------


class _Response:
    def __init__(self, status_code=200, payload=None, error=None):
        self.status_code = status_code
        self._payload = payload
        self._error = error

    def json(self):
        if self._error:
            raise self._error
        return self._payload


class TestLaunchPieces:
    def test_shell_lines_quote_values(self):
        plan = launch.Launch(argv=["claude", "-p", "hi there"], env={"A": "x y", "B": "z"})
        assert launch.shell_lines(plan) == [
            "export A='x y'",
            "export B=z",
            "claude -p 'hi there'",
        ]

    def test_check_tool_refuses_an_unknown_tool(self):
        with pytest.raises(launch.LaunchError, match="Unknown tool vim"):
            launch.check_tool("vim")

    def test_build_launch_refuses_an_unknown_tool(self):
        with pytest.raises(launch.LaunchError, match="Unknown tool"):
            launch.build_launch("vim", "http://x", "m")

    def test_server_up(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda url, timeout: _Response(200))
        assert launch.server_up("http://h:1") is True
        monkeypatch.setattr(httpx, "get", lambda url, timeout: _Response(503))
        assert launch.server_up("http://h:1") is False

        def refuse(url, timeout):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", refuse)
        assert launch.server_up("http://h:1") is False

    def test_start_server_passes_parallel_and_key_in_env(self, monkeypatch, tmp_path):
        seen = {}

        def fake_popen(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["env"] = kwargs["env"]
            return "proc"

        monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
        log = tmp_path / "logs" / "serve.log"
        assert launch.start_server(18000, "k3y", log, parallel=4) == "proc"
        assert seen["cmd"][-2:] == ["--parallel", "4"]
        assert "k3y" not in seen["cmd"]
        assert seen["env"]["HFL_API_KEY"] == "k3y"
        assert log.parent.is_dir()

    def test_start_server_without_key_or_parallel(self, monkeypatch, tmp_path):
        seen = {}

        def fake_popen(cmd, **kwargs):
            seen.update(cmd=cmd, env=kwargs["env"])
            return "proc"

        monkeypatch.delenv("HFL_API_KEY", raising=False)
        monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
        launch.start_server(18001, None, tmp_path / "serve.log")
        assert "--parallel" not in seen["cmd"]
        assert seen["cmd"][-4:] == ["--host", "127.0.0.1", "--port", "18001"]
        assert "HFL_API_KEY" not in seen["env"]
        assert seen["env"]["PYTHONUNBUFFERED"] == "1"

    def test_wait_until_up(self, monkeypatch):
        proc = MagicMock()
        proc.poll.return_value = None
        answers = iter([False, True])
        monkeypatch.setattr(launch, "server_up", lambda url: next(answers))
        monkeypatch.setattr(launch.time, "sleep", lambda s: None)
        assert launch.wait_until_up("http://h", proc, timeout=60) is True

    def test_wait_until_up_stops_when_the_process_exits(self, monkeypatch):
        proc = MagicMock()
        proc.poll.return_value = 1
        monkeypatch.setattr(launch, "server_up", MagicMock(return_value=True))
        assert launch.wait_until_up("http://h", proc) is False
        launch.server_up.assert_not_called()

    def test_wait_until_up_times_out(self, monkeypatch):
        proc = MagicMock()
        proc.poll.return_value = None
        clock = iter([0.0, 0.0, 100.0])
        monkeypatch.setattr(launch.time, "monotonic", lambda: next(clock))
        monkeypatch.setattr(launch.time, "sleep", lambda s: None)
        monkeypatch.setattr(launch, "server_up", lambda url: False)
        assert launch.wait_until_up("http://h", proc, timeout=5) is False

    def test_stop_server(self):
        done = MagicMock()
        done.poll.return_value = 0
        launch.stop_server(done)
        done.send_signal.assert_not_called()

        polite = MagicMock()
        polite.poll.return_value = None
        launch.stop_server(polite)
        polite.send_signal.assert_called_once()
        polite.kill.assert_not_called()

        stubborn = MagicMock()
        stubborn.poll.return_value = None
        stubborn.wait.side_effect = [subprocess.TimeoutExpired("hfl", 30), 0]
        launch.stop_server(stubborn)
        stubborn.kill.assert_called_once()

    def test_preload_unreachable(self, monkeypatch):
        def refuse(*a, **k):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "post", refuse)
        with pytest.raises(launch.LaunchError, match="Could not reach http://h"):
            launch.preload("http://h", "m", None)

    def test_preload_rejected_says_the_servers_words(self, monkeypatch):
        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: _Response(404, {"error": {"message": "model 'm' not found"}}),
        )
        with pytest.raises(launch.LaunchError, match="HTTP 404: model 'm' not found"):
            launch.preload("http://h", "m", "key")

    def test_preload_ps_unreadable_is_no_context(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response(200, {}))
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(200, error=ValueError("x")))
        assert launch.preload("http://h", "m", None) is None

    def test_preload_model_not_listed_is_no_context(self, monkeypatch):
        monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response(200, {}))
        monkeypatch.setattr(
            httpx, "get", lambda *a, **k: _Response(200, {"models": [{"name": "other"}]})
        )
        assert launch.preload("http://h", "m", None) is None

    def test_preload_sends_the_key(self, monkeypatch):
        seen = {}

        def post(url, json, headers, timeout):
            seen["headers"] = headers
            return _Response(200, {})

        monkeypatch.setattr(httpx, "post", post)
        monkeypatch.setattr(
            httpx,
            "get",
            lambda *a, **k: _Response(
                200, {"models": [{"model": "m", "details": {"context_size": 8192}}]}
            ),
        )
        assert launch.preload("http://h", "m", "s3") == 8192
        assert seen["headers"] == {"Authorization": "Bearer s3"}

    def test_server_error_shapes(self):
        assert launch._server_error(_Response(500, error=ValueError())) == "HTTP 500"
        assert launch._server_error(_Response(400, {"detail": "bad"})) == "HTTP 400: bad"
        assert launch._server_error(_Response(400, {"x": 1})) == "HTTP 400: {'x': 1}"
        assert (
            launch._server_error(_Response(507, {"error": {"error": "no room"}}))
            == "HTTP 507: no room"
        )


class TestLaunchRun:
    def _common(self, monkeypatch):
        monkeypatch.setattr(launch, "check_tool", lambda tool: None)
        monkeypatch.setattr(launch, "preload", lambda *a: 4096)
        ran = {}

        def fake_run(argv, env):
            ran["argv"] = argv
            ran["env"] = env
            return types.SimpleNamespace(returncode=3)

        monkeypatch.setattr(launch.subprocess, "run", fake_run)
        return ran

    def test_a_running_server_keeps_its_settings(self, monkeypatch, tmp_path):
        ran = self._common(monkeypatch)
        monkeypatch.setattr(launch, "server_up", lambda url: True)
        said: list[str] = []
        code = launch.run(
            "claude",
            "m",
            host="127.0.0.1",
            port=11434,
            api_key=None,
            extra=[],
            log_path=tmp_path / "l.log",
            say=said.append,
            parallel=4,
        )
        assert code == 3
        assert any("keeps its own settings" in s for s in said)
        assert ran["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "4096"

    def test_a_server_that_does_not_start_is_reported_and_stopped(self, monkeypatch, tmp_path):
        self._common(monkeypatch)
        monkeypatch.setattr(launch, "server_up", lambda url: False)
        proc = MagicMock()
        monkeypatch.setattr(launch, "start_server", lambda *a: proc)
        monkeypatch.setattr(launch, "wait_until_up", lambda url, p: False)
        stopped = []
        monkeypatch.setattr(launch, "stop_server", stopped.append)
        with pytest.raises(launch.LaunchError, match="did not start"):
            launch.run(
                "codex",
                "m",
                host="127.0.0.1",
                port=1,
                api_key=None,
                extra=[],
                log_path=tmp_path / "l.log",
                say=lambda s: None,
            )
        assert stopped == [proc]

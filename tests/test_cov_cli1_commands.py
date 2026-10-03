# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl serve`` (backend, exposure, tray, preload), ``list``, ``cp``,
``import``, ``outdated``, ``stop`` and ``show``: the paths the other CLI
tests leave out. Registry under a temporary HFL home; Hub, engines and the
server are faked."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import typer
from typer.testing import CliRunner

from hfl.cli import main
from hfl.models.manifest import ModelManifest

runner = CliRunner()


@pytest.fixture
def out(capsys):
    def read() -> str:
        return capsys.readouterr().out

    return read


def _manifest(name="m1", model_type="llm", **kw) -> ModelManifest:
    return ModelManifest(
        name=name,
        repo_id=kw.pop("repo_id", f"org/{name}"),
        local_path=kw.pop("local_path", f"/nonexistent/{name}.gguf"),
        format=kw.pop("format", "gguf"),
        model_type=model_type,
        size_bytes=kw.pop("size_bytes", 1024**3),
        **kw,
    )


@pytest.fixture
def registry(temp_config):
    from hfl.models.registry import ModelRegistry

    return ModelRegistry()


# --- serve: backend, container network, exposure, tray, preload -----------------


def test_a_negative_parallel_is_refused(monkeypatch, out):
    with pytest.raises(typer.Exit) as exc:
        main._choose_backend("auto", -1)
    assert exc.value.exit_code == 2
    assert "--parallel must be 0 or more" in out()


class TestContainerNetwork:
    def _linux(self, monkeypatch, *, ino=None, release="7.0.1", stat_error=False):
        def stat(path):
            assert path == "/proc/self/ns/net"
            if stat_error:
                raise OSError("no /proc")
            return SimpleNamespace(st_ino=ino)

        fake_os = SimpleNamespace(stat=stat, uname=lambda: SimpleNamespace(release=release))
        monkeypatch.setattr(main, "sys", SimpleNamespace(platform="linux"))
        monkeypatch.setattr(main, "os", fake_os)

    def test_not_linux_is_unknown(self, monkeypatch):
        monkeypatch.setattr(main, "sys", SimpleNamespace(platform="darwin"))
        assert main._container_network() == "unknown"

    def test_no_proc_is_unknown(self, monkeypatch):
        self._linux(monkeypatch, stat_error=True)
        assert main._container_network() == "unknown"

    def test_the_hosts_namespace_is_host(self, monkeypatch):
        self._linux(monkeypatch, ino=main._HOST_NETNS_INO)
        assert main._container_network() == "host"

    def test_another_namespace_on_a_new_kernel_is_its_own(self, monkeypatch):
        self._linux(monkeypatch, ino=4026531999, release="6.18.0-generic")
        assert main._container_network() == "own"

    def test_an_old_kernel_cannot_tell(self, monkeypatch):
        self._linux(monkeypatch, ino=4026531999, release="5.15.0")
        assert main._container_network() == "unknown"

    def test_an_unparseable_release_cannot_tell(self, monkeypatch):
        self._linux(monkeypatch, ino=4026531999, release="custom-kernel")
        assert main._container_network() == "unknown"


def test_declining_the_exposure_question_exits_0(monkeypatch, out):
    monkeypatch.delenv("HFL_ACCEPT_NETWORK_EXPOSURE", raising=False)
    monkeypatch.setattr(main, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    with pytest.raises(typer.Exit) as exc:
        main._confirm_exposure("0.0.0.0", None)  # noqa: S104 - not a bind
    assert exc.value.exit_code == 0
    assert "Warning:" in out()


def test_accepting_the_exposure_question_goes_on(monkeypatch, out):
    monkeypatch.delenv("HFL_ACCEPT_NETWORK_EXPOSURE", raising=False)
    monkeypatch.setattr(main, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    main._confirm_exposure("192.168.1.10", "secret")
    assert "API key authentication enabled." in out()


class TestTray:
    def _args(self):
        return ("127.0.0.1", 11434, None, None, "INFO", False)

    def test_linux_without_a_display_says_so(self, monkeypatch, out):
        monkeypatch.setattr(main.sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        with pytest.raises(typer.Exit) as exc:
            main._run_tray(*self._args())
        assert exc.value.exit_code == 1
        assert "needs a desktop session" in out()

    def test_runs_the_tray_with_the_servers_settings(self, monkeypatch):
        calls: list[dict] = []
        icon = SimpleNamespace(run_tray=lambda **kw: calls.append(kw))
        monkeypatch.setitem(sys.modules, "hfl.tray.icon", icon)
        monkeypatch.setattr(main.sys, "platform", "darwin")
        main._run_tray("0.0.0.0", 9000, "k", "m", "DEBUG", True)  # noqa: S104
        assert calls == [
            {
                "host": "0.0.0.0",  # noqa: S104
                "port": 9000,
                "api_key": "k",
                "model": "m",
                "log_level": "DEBUG",
                "json_logs": True,
                "auto_start": True,
            }
        ]

    def test_without_pystray_exits_1(self, monkeypatch, out):
        monkeypatch.setitem(sys.modules, "hfl.tray.icon", None)  # import fails
        monkeypatch.setattr(main.sys, "platform", "darwin")
        with pytest.raises(typer.Exit) as exc:
            main._run_tray(*self._args())
        assert exc.value.exit_code == 1
        assert "Tray mode requires pystray and Pillow" in out()

    def test_without_pystray_names_the_tray_extra(self, monkeypatch, out):
        monkeypatch.setitem(sys.modules, "hfl.tray.icon", None)
        monkeypatch.setattr(main.sys, "platform", "darwin")
        with pytest.raises(typer.Exit):
            main._run_tray(*self._args())
        assert "pip install hfl[tray]" in out()

    def _raising(self, monkeypatch, exc: Exception):
        def run_tray(**kw):
            raise exc

        monkeypatch.setitem(sys.modules, "hfl.tray.icon", SimpleNamespace(run_tray=run_tray))
        monkeypatch.setattr(main.sys, "platform", "darwin")

    def test_an_x_server_that_does_not_answer_says_so(self, monkeypatch, out):
        xerror = type("DisplayConnectionError", (Exception,), {"__module__": "Xlib.error"})
        self._raising(monkeypatch, xerror("cannot connect to :0"))
        with pytest.raises(typer.Exit) as exc:
            main._run_tray(*self._args())
        assert exc.value.exit_code == 1
        text = out()
        assert "needs a desktop session" in text and "cannot connect to :0" in text

    def test_any_other_tray_failure_propagates(self, monkeypatch):
        self._raising(monkeypatch, ValueError("a real bug"))
        with pytest.raises(ValueError, match="a real bug"):
            main._run_tray(*self._args())


class TestPreload:
    def test_a_missing_model_exits_1(self, monkeypatch, out):
        monkeypatch.setattr(main, "_local_or_pulled", lambda *a, **k: None)
        with pytest.raises(typer.Exit) as exc:
            main._preload("ghost", 0, SimpleNamespace())
        assert exc.value.exit_code == 1
        assert "ghost" in out()

    def test_loads_the_model_into_the_state(self, monkeypatch, out):
        manifest = _manifest()
        engine = MagicMock()
        checked: list = []
        monkeypatch.setattr(main, "_local_or_pulled", lambda *a, **k: manifest)
        monkeypatch.setattr(main, "_memory_check_or_exit", lambda m, c: checked.append(c))
        monkeypatch.setattr("hfl.engine.selector.select_engine", lambda path: engine)
        monkeypatch.setattr("hfl.api.model_loader.load_kwargs_for", lambda m, n: {"n_ctx": n})
        state = SimpleNamespace(engine=None, current_model=None)
        main._preload("m1", 8192, state)
        assert state.engine is engine and state.current_model is manifest
        engine.load.assert_called_once_with(manifest.local_path, n_ctx=8192)
        assert checked == [8192]
        assert "Pre-loading" in out()

    def _missing_backend(self, monkeypatch):
        from hfl.engine.selector import MissingDependencyError

        def select(path):
            # The selector's own wording (see its MLX / transformers errors).
            raise MissingDependencyError("Install it with:\n  pip install 'hfl[mlx]'")

        monkeypatch.setattr(main, "_local_or_pulled", lambda *a, **k: _manifest())
        monkeypatch.setattr(main, "_memory_check_or_exit", lambda m, c: None)
        monkeypatch.setattr("hfl.engine.selector.select_engine", select)

    def test_a_missing_backend_exits_1(self, monkeypatch, out):
        self._missing_backend(monkeypatch)
        with pytest.raises(typer.Exit) as exc:
            main._preload("m1", 0, SimpleNamespace(engine=None))
        assert exc.value.exit_code == 1
        text = out()
        assert "Missing dependency" in text and "pip install" in text

    def test_a_missing_backend_keeps_the_extra_in_the_hint(self, monkeypatch, out):
        self._missing_backend(monkeypatch)
        with pytest.raises(typer.Exit):
            main._preload("m1", 0, SimpleNamespace(engine=None))
        assert "pip install 'hfl[mlx]'" in out()


def test_serve_ctx_reaches_the_state(monkeypatch):
    state = SimpleNamespace(context_size_override=None)
    started: list[dict] = []
    monkeypatch.delenv("HFL_API_KEY", raising=False)
    monkeypatch.setattr("hfl.api.state.get_state", lambda: state)
    monkeypatch.setattr("hfl.api.server.start_server", lambda **kw: started.append(kw))
    monkeypatch.setattr(main, "_choose_backend", lambda backend, parallel: None)
    monkeypatch.setattr(main, "_apply_sandbox", lambda s: None)
    result = runner.invoke(main.app, ["serve", "--host", "127.0.0.1", "--ctx", "8192"])
    assert result.exit_code == 0, result.stdout
    assert state.context_size_override == 8192
    assert started and started[0]["host"] == "127.0.0.1"


# --- list ---------------------------------------------------------------------


class TestList:
    def test_supported_only_hides_unsupported_models(self, registry):
        registry.add(_manifest("chat", "llm", license="mit"))
        registry.add(_manifest("whisper", "stt"))
        result = runner.invoke(main.app, ["list", "--supported-only"])
        assert result.exit_code == 0, result.stdout
        assert "chat" in result.stdout and "whisper" not in result.stdout
        assert "--supported-only" not in result.stdout  # no tip when filtered

    def test_supported_only_with_nothing_supported(self, registry):
        registry.add(_manifest("whisper", "stt"))
        result = runner.invoke(main.app, ["list", "-s"])
        assert result.exit_code == 0
        assert "No supported models" in result.stdout

    def test_all_supported_shows_no_tip(self, registry):
        registry.add(_manifest("chat", "llm", license="cc-by-nc-4.0"))
        registry.add(_manifest("other", "llm", license="llama3"))
        result = runner.invoke(main.app, ["list"])
        assert result.exit_code == 0, result.stdout
        assert "chat" in result.stdout and "other" in result.stdout
        assert "unsupported types" not in result.stdout


# --- pulling from search results --------------------------------------------------


class TestPullSelected:
    def test_declined_downloads_nothing(self, monkeypatch, out):
        monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
        monkeypatch.setattr(main, "pull", lambda **kw: pytest.fail("pulled after a no"))
        main._pull_selected_model(SimpleNamespace(id="org/x", siblings=None))
        assert "Download cancelled" in out()

    def test_a_gguf_repo_is_pulled_with_the_defaults(self, monkeypatch, out):
        calls: list[dict] = []
        monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
        monkeypatch.setattr(main, "pull", lambda **kw: calls.append(kw))
        siblings = [SimpleNamespace(rfilename="README.md"), SimpleNamespace(rfilename="a.gguf")]
        main._pull_selected_model(SimpleNamespace(id="org/x-GGUF", siblings=siblings))
        assert calls == [
            {
                "model": "org/x-GGUF",
                "quantize": "Q4_K_M",
                "format": "auto",
                "revision": None,
                "alias": None,
                "skip_license": False,
            }
        ]
        assert "org/x-GGUF" in out()

    def test_an_exit_from_pull_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)

        def exits(**kw):
            raise SystemExit(0)

        monkeypatch.setattr(main, "pull", exits)
        main._pull_selected_model(SimpleNamespace(id="org/x", siblings=[]))


# --- cp -----------------------------------------------------------------------


class TestCp:
    def test_copies_under_a_new_name(self, registry):
        registry.add(_manifest("src"))
        result = runner.invoke(main.app, ["cp", "src", "dst"])
        assert result.exit_code == 0, result.stdout
        assert "Copied" in result.stdout
        from hfl.models.registry import ModelRegistry

        copy = ModelRegistry().get("dst")
        assert copy is not None and copy.local_path == "/nonexistent/src.gguf"

    def test_a_missing_source_exits_1(self, registry):
        result = runner.invoke(main.app, ["cp", "nope", "dst"])
        assert result.exit_code == 1
        assert "Source model not found" in result.stdout

    def test_an_existing_destination_exits_1(self, registry):
        registry.add(_manifest("src"))
        registry.add(_manifest("dst"))
        result = runner.invoke(main.app, ["cp", "src", "dst"])
        assert result.exit_code == 1
        assert "Destination already exists" in result.stdout

    def test_a_copy_that_raises_exits_1(self, registry, monkeypatch):
        registry.add(_manifest("src"))

        def boom(self, s, d):
            raise OSError("disk gone")

        monkeypatch.setattr("hfl.models.registry.ModelRegistry.copy", boom)
        result = runner.invoke(main.app, ["cp", "src", "dst"])
        assert result.exit_code == 1
        assert "Copy failed" in result.stdout and "disk gone" in result.stdout

    def test_a_copy_that_loses_a_race_exits_1(self, registry, monkeypatch):
        registry.add(_manifest("src"))
        monkeypatch.setattr("hfl.models.registry.ModelRegistry.copy", lambda self, s, d: False)
        result = runner.invoke(main.app, ["cp", "src", "dst"])
        assert result.exit_code == 1
        assert "concurrent write" in result.stdout


# --- import -------------------------------------------------------------------


class TestImport:
    def _fake(self, monkeypatch, tmp_path, model_type, kind="gguf"):
        model = tmp_path / "weights.gguf"
        model.write_bytes(b"GGUF")
        built = _manifest("imported", model_type, local_path=str(model))
        monkeypatch.setattr("hfl.models.importer.choose_model", lambda p: (model, kind))
        monkeypatch.setattr("hfl.models.importer.default_name", lambda m: "imported")
        monkeypatch.setattr("hfl.models.importer.manifest_for", lambda m, n, a: built)
        monkeypatch.setattr("hfl.models.importer.manifest_for_folder", lambda m, n, a: built)
        return model

    def test_an_embedding_model_points_at_the_api(self, registry, monkeypatch, tmp_path):
        model = self._fake(monkeypatch, tmp_path, "embedding")
        result = runner.invoke(main.app, ["import", str(model), "--alias", "emb"])
        assert result.exit_code == 0, result.stdout
        assert '"model": "emb"' in result.stdout and "/api/embed" in result.stdout
        from hfl.models.registry import ModelRegistry

        assert ModelRegistry().get("imported") is not None

    def test_a_tts_folder_points_at_hfl_tts(self, registry, monkeypatch, tmp_path):
        self._fake(monkeypatch, tmp_path, "tts", kind="folder")
        result = runner.invoke(main.app, ["import", str(tmp_path)])
        assert result.exit_code == 0, result.stdout
        assert 'hfl tts imported "..."' in result.stdout


# --- outdated -------------------------------------------------------------------


class TestOutdated:
    def test_an_unknown_model_exits_1(self, registry):
        result = runner.invoke(main.app, ["outdated", "ghost"])
        assert result.exit_code == 1
        assert "ghost" in result.stdout

    def test_no_models(self, registry):
        result = runner.invoke(main.app, ["outdated"])
        assert result.exit_code == 0
        assert "No local models." in result.stdout

    def test_one_model_is_checked_alone(self, registry, monkeypatch):
        from hfl.hub.outdated import Check

        registry.add(_manifest("a"))
        registry.add(_manifest("b"))
        checked: list[list[str]] = []

        def check_all(manifests, api, models_dir):
            checked.append([m.name for m in manifests])
            return [Check(name="b", status="current")]

        monkeypatch.setattr("hfl.hub.outdated.check_all", check_all)
        result = runner.invoke(main.app, ["outdated", "b"])
        assert result.exit_code == 0, result.stdout
        assert checked == [["b"]]


# --- stop -----------------------------------------------------------------------


class TestStop:
    def _post(self, monkeypatch, *, json=None, exc=None):
        sent: list[tuple] = []

        def post(url, json=None, timeout=None):  # noqa: A002 - httpx's name
            sent.append((url, json))
            if exc is not None:
                raise exc
            response = MagicMock()
            response.json.return_value = reply
            return response

        reply = json
        monkeypatch.setattr(httpx, "post", post)
        return sent

    def test_stop_all_sends_no_model(self, monkeypatch):
        sent = self._post(monkeypatch, json={"status": "stopped"})
        result = runner.invoke(main.app, ["stop", "--port", "18000"])
        assert result.exit_code == 0, result.stdout
        assert sent == [("http://127.0.0.1:18000/api/stop", {})]
        assert "Stopped" in result.stdout and "(all)" in result.stdout

    @pytest.mark.parametrize(
        ("reply", "said"),
        [
            ({"status": "not_loaded", "model": "m1"}, "is not currently loaded"),
            ({"status": "nothing_loaded"}, "nothing to stop"),
            ({"status": "weird"}, "Unexpected response"),
        ],
    )
    def test_each_reply_is_said(self, monkeypatch, reply, said):
        sent = self._post(monkeypatch, json=reply)
        result = runner.invoke(main.app, ["stop", "m1"])
        assert result.exit_code == 0, result.stdout
        assert sent[0][1] == {"model": "m1"}
        assert said in result.stdout

    def test_no_server_says_how_to_start_it(self, monkeypatch):
        self._post(monkeypatch, exc=httpx.ConnectError("refused"))
        result = runner.invoke(main.app, ["stop", "m1"])
        assert result.exit_code == 1
        assert "Cannot reach HFL server" in result.stdout and "hfl serve" in result.stdout

    def test_a_server_error_exits_1(self, monkeypatch):
        request = httpx.Request("POST", "http://127.0.0.1:11434/api/stop")
        error = httpx.HTTPStatusError(
            "500 boom", request=request, response=httpx.Response(500, request=request)
        )
        self._post(monkeypatch, exc=error)
        result = runner.invoke(main.app, ["stop", "m1"])
        assert result.exit_code == 1
        assert "Server error" in result.stdout and "500 boom" in result.stdout


# --- show -----------------------------------------------------------------------


class TestShow:
    @pytest.fixture
    def shown(self, registry):
        registry.add(
            _manifest(
                "m1",
                license="mit",
                license_name="MIT \x1b]0;pwned\x07License",
                context_length=8192,
                architecture="llama",
                chat_template="{{ [x] }}",
                default_parameters={"temperature": 0.2},
            )
        )

    def test_unknown_model_exits_1(self, registry):
        result = runner.invoke(main.app, ["show", "ghost[/x]"])
        assert result.exit_code == 1
        assert "Model not found" in result.stdout and "ghost[/x]" in result.stdout

    def test_modelfile(self, shown):
        result = runner.invoke(main.app, ["show", "m1", "--modelfile"])
        assert result.exit_code == 0, result.stdout
        assert "FROM" in result.stdout

    def test_parameters(self, shown):
        result = runner.invoke(main.app, ["show", "m1", "--parameters"])
        assert result.exit_code == 0, result.stdout
        assert "temperature" in result.stdout and "0.2" in result.stdout

    def test_template_keeps_its_brackets(self, shown):
        result = runner.invoke(main.app, ["show", "m1", "--template"])
        assert result.exit_code == 0, result.stdout
        assert "{{ [x] }}" in result.stdout

    def test_license_drops_terminal_controls(self, shown):
        result = runner.invoke(main.app, ["show", "m1", "--license"])
        assert result.exit_code == 0, result.stdout
        assert "\x1b" not in result.stdout and "\x07" not in result.stdout
        assert "MIT ]0;pwnedLicense" in result.stdout

    def test_summary_shows_the_context(self, shown):
        result = runner.invoke(main.app, ["show", "m1"])
        assert result.exit_code == 0, result.stdout
        assert "8192 tokens" in result.stdout and "llama" in result.stdout

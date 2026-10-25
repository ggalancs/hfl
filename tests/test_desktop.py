# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""HFL opened from a desktop: HFL.app from Finder, the MSI's Start-menu
shortcut (hfl.utils.desktop and its wiring into the CLI)."""

from __future__ import annotations

import io
import os
import sys
import types

import httpx
import pytest

from hfl.utils import desktop

APP_EXE = "/Applications/HFL.app/Contents/MacOS/hfl"


class _Stdin(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def mac_app(monkeypatch):
    """A frozen HFL inside HFL.app on macOS, stdin not a terminal."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", APP_EXE)
    monkeypatch.setattr(sys, "stdin", _Stdin(tty=False))


class TestOpenedAsMacApp:
    def test_finder_opening_the_app(self, mac_app) -> None:
        assert desktop.opened_as_mac_app([APP_EXE]) is True

    def test_old_macos_passes_a_process_serial_number(self, mac_app) -> None:
        assert desktop.opened_as_mac_app([APP_EXE, "-psn_0_12345"]) is True

    def test_any_real_argument_is_the_cli(self, mac_app) -> None:
        assert desktop.opened_as_mac_app([APP_EXE, "list"]) is False

    def test_a_terminal_gets_the_help(self, mac_app, monkeypatch) -> None:
        monkeypatch.setattr(sys, "stdin", _Stdin(tty=True))
        assert desktop.opened_as_mac_app([APP_EXE]) is False

    def test_outside_an_app_bundle(self, mac_app, monkeypatch) -> None:
        monkeypatch.setattr(sys, "executable", "/usr/local/bin/hfl")
        assert desktop.opened_as_mac_app(["/usr/local/bin/hfl"]) is False

    def test_not_frozen(self, mac_app, monkeypatch) -> None:
        monkeypatch.setattr(sys, "frozen", False)
        assert desktop.opened_as_mac_app([APP_EXE]) is False

    def test_not_macos(self, mac_app, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        assert desktop.opened_as_mac_app([APP_EXE]) is False

    def test_no_stdin_at_all(self, mac_app, monkeypatch) -> None:
        monkeypatch.setattr(sys, "stdin", None)
        assert desktop.opened_as_mac_app([APP_EXE]) is True


class TestChatUrl:
    @pytest.mark.parametrize(
        ("host", "url"),
        [
            ("127.0.0.1", "http://127.0.0.1:11434/ui"),
            ("0.0.0.0", "http://127.0.0.1:11434/ui"),  # noqa: S104
            ("::", "http://127.0.0.1:11434/ui"),
            ("", "http://127.0.0.1:11434/ui"),
            ("::1", "http://[::1]:11434/ui"),
            ("hfl.local", "http://hfl.local:11434/ui"),
        ],
    )
    def test_the_address_a_local_browser_reaches(self, host, url) -> None:
        assert desktop.chat_url(host, 11434) == url


class TestOpenChatWhenReady:
    def test_opens_once_the_server_answers(self, monkeypatch) -> None:
        answers = iter([httpx.ConnectError("not yet"), types.SimpleNamespace(status_code=503)])
        asked: list[str] = []

        def get(url, timeout):
            asked.append(url)
            answer = next(answers, types.SimpleNamespace(status_code=200))
            if isinstance(answer, Exception):
                raise answer
            return answer

        opened: list[str] = []
        monkeypatch.setattr(httpx, "get", get)
        monkeypatch.setattr(desktop.webbrowser, "open", opened.append)
        monkeypatch.setattr(desktop.time, "sleep", lambda _s: None)
        desktop.open_chat_when_ready("0.0.0.0", 8080).join(5)  # noqa: S104
        assert opened == ["http://127.0.0.1:8080/ui"]
        assert asked == ["http://127.0.0.1:8080/healthz"] * 3

    def test_gives_up_without_opening(self, monkeypatch) -> None:
        def get(url, timeout):
            raise httpx.ConnectError("refused")

        opened: list[str] = []
        clock = iter(range(0, 1000, 50))
        monkeypatch.setattr(httpx, "get", get)
        monkeypatch.setattr(desktop.webbrowser, "open", opened.append)
        monkeypatch.setattr(desktop.time, "sleep", lambda _s: None)
        monkeypatch.setattr(desktop.time, "monotonic", lambda: next(clock))
        thread = desktop.open_chat_when_ready("127.0.0.1", 11434, timeout=120)
        thread.join(5)
        assert not thread.is_alive()
        assert opened == []


class TestLogToFile:
    def test_stdout_and_stderr_go_to_the_file(self, tmp_path) -> None:
        log = tmp_path / "Logs" / "HFL" / "hfl.log"
        saved = os.dup(1), os.dup(2)
        try:
            assert desktop.log_to_file(log) == log
            os.write(1, b"out\n")
            os.write(2, b"err\n")
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        assert log.read_text() == "out\nerr\n"

    def test_a_log_past_the_limit_starts_afresh(self, tmp_path, monkeypatch) -> None:
        log = tmp_path / "hfl.log"
        log.write_text("old\n" * 10)
        monkeypatch.setattr(desktop, "_LOG_LIMIT", 10)
        saved = os.dup(1), os.dup(2)
        try:
            desktop.log_to_file(log)
            os.write(1, b"new\n")
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        assert log.read_text() == "new\n"


class _Kernel32:
    def __init__(self, pids: list[int], window: int = 77) -> None:
        self._pids, self._window = pids, window

    def GetConsoleProcessList(self, buffer, size):  # noqa: N802
        for i, pid in enumerate(self._pids[:size]):
            buffer[i] = pid
        return len(self._pids)

    def GetConsoleWindow(self):  # noqa: N802
        return self._window


@pytest.fixture
def windows(monkeypatch):
    """Windows' console API faked; returns the windows ShowWindow hid."""
    import ctypes

    hidden: list[tuple[int, int]] = []
    monkeypatch.setattr(sys, "platform", "win32")

    def install(pids: list[int], window: int = 77) -> list[tuple[int, int]]:
        user32 = types.SimpleNamespace(ShowWindow=lambda w, how: hidden.append((w, how)))
        windll = types.SimpleNamespace(kernel32=_Kernel32(pids, window), user32=user32)
        monkeypatch.setattr(ctypes, "windll", windll, raising=False)
        return hidden

    return install


class TestHideOwnConsole:
    def test_a_console_of_its_own_is_hidden(self, windows) -> None:
        hidden = windows([os.getpid()])
        assert desktop.hide_own_console() is True
        assert hidden == [(77, 0)]

    def test_the_one_file_bootloader_is_not_somebody_else(self, windows, monkeypatch) -> None:
        hidden = windows([os.getppid(), os.getpid()])
        monkeypatch.setattr(desktop, "_is_own_bootloader", lambda pid: pid == os.getppid())
        assert desktop.hide_own_console() is True
        assert hidden == [(77, 0)]

    def test_a_shared_terminal_stays(self, windows, monkeypatch) -> None:
        hidden = windows([4242, os.getpid()])
        monkeypatch.setattr(desktop, "_is_own_bootloader", lambda pid: False)
        assert desktop.hide_own_console() is False
        assert hidden == []

    def test_no_console_window(self, windows) -> None:
        hidden = windows([os.getpid()], window=0)
        assert desktop.hide_own_console() is False
        assert hidden == []

    def test_not_windows(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        assert desktop.hide_own_console() is False


class TestIsOwnBootloader:
    def test_the_parent_running_this_executable(self, monkeypatch) -> None:
        psutil = pytest.importorskip("psutil")
        monkeypatch.setattr(sys, "executable", psutil.Process(os.getppid()).exe())
        assert desktop._is_own_bootloader(os.getppid()) is True

    def test_the_parent_running_something_else(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "executable", "/nowhere/hfl.exe")
        assert desktop._is_own_bootloader(os.getppid()) is False

    def test_not_the_parent(self) -> None:
        assert desktop._is_own_bootloader(os.getpid()) is False


class TestCliMain:
    def test_finder_runs_serve_under_the_tray_and_opens_the_chat(self, monkeypatch) -> None:
        from hfl.cli import main

        monkeypatch.setattr(sys, "argv", [APP_EXE])
        monkeypatch.setattr(desktop, "opened_as_mac_app", lambda argv: True)
        logged: list[bool] = []
        monkeypatch.setattr(desktop, "log_to_file", lambda: logged.append(True))
        seen: list[list[str]] = []
        monkeypatch.setattr(main, "app", lambda: seen.append(list(sys.argv)))
        main.cli_main()
        assert seen == [[APP_EXE, "serve", "--tray", "--open"]]
        assert logged == [True]

    def test_a_terminal_keeps_its_arguments(self, monkeypatch) -> None:
        from hfl.cli import main

        monkeypatch.setattr(sys, "argv", ["hfl", "list"])
        monkeypatch.setattr(desktop, "opened_as_mac_app", lambda argv: False)
        seen: list[list[str]] = []
        monkeypatch.setattr(main, "app", lambda: seen.append(list(sys.argv)))
        main.cli_main()
        assert seen == [["hfl", "list"]]


class TestServeOpen:
    def test_serve_open_opens_the_chat_page(self, monkeypatch) -> None:
        from typer.testing import CliRunner

        from hfl.api import server
        from hfl.cli import main

        opened: list[tuple[str, int]] = []
        monkeypatch.setattr(desktop, "open_chat_when_ready", lambda h, p: opened.append((h, p)))
        monkeypatch.setattr(server, "start_server", lambda **kw: None)
        result = CliRunner().invoke(main.app, ["serve", "--port", "18080", "--open"])
        assert result.exit_code == 0, result.output
        assert opened == [("127.0.0.1", 18080)]

    def test_serve_without_open_does_not(self, monkeypatch) -> None:
        from typer.testing import CliRunner

        from hfl.api import server
        from hfl.cli import main

        opened: list[tuple[str, int]] = []
        monkeypatch.setattr(desktop, "open_chat_when_ready", lambda h, p: opened.append((h, p)))
        monkeypatch.setattr(server, "start_server", lambda **kw: None)
        result = CliRunner().invoke(main.app, ["serve", "--port", "18080"])
        assert result.exit_code == 0, result.output
        assert opened == []

    def test_tray_open_reaches_the_tray(self, monkeypatch) -> None:
        from typer.testing import CliRunner

        from hfl.cli import main

        trays: list[tuple] = []
        monkeypatch.setattr(main, "_run_tray", lambda *a: trays.append(a))
        result = CliRunner().invoke(main.app, ["serve", "--port", "18080", "--tray", "--open"])
        assert result.exit_code == 0, result.output
        assert trays and trays[0][1] == 18080 and trays[0][-1] is True

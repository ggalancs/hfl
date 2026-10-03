# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The tray icon with pystray, Pillow and AppKit faked in ``sys.modules``
(CI installs none of them): the drawn icon, the menu and its actions, the
icon following the server's status, and the stop on a macOS quit."""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hfl.tray import icon as tray_icon
from hfl.tray.controller import ServerStatus, TrayServerController

# -- fakes ---------------------------------------------------------------------------


class _Image:
    def __init__(self, mode, size, color) -> None:
        self.mode, self.size, self.color = mode, size, color


class _Draw:
    def __init__(self, img: _Image) -> None:
        self.img = img
        self.calls: list[tuple] = []
        img.draw = self  # type: ignore[attr-defined]

    def ellipse(self, box, fill) -> None:
        self.calls.append(("ellipse", box, fill))

    def textbbox(self, xy, text, font) -> tuple[int, int, int, int]:
        return (0, 2, 20, 26)

    def text(self, xy, text, fill, font) -> None:
        self.calls.append(("text", xy, text, fill, font))


class _Menu:
    SEPARATOR = object()

    def __init__(self, *items) -> None:
        self.items = items


class _MenuItem:
    def __init__(self, text, action, enabled=True) -> None:
        self.text, self.action, self.enabled = text, action, enabled


class _Icon:
    def __init__(self, name, icon, title, menu) -> None:
        self.name, self.icon, self.title, self.menu = name, icon, title, menu
        self.ran = 0
        self.stopped = 0

    def run(self) -> None:
        self.ran += 1

    def stop(self) -> None:
        self.stopped += 1


@pytest.fixture
def gui(monkeypatch):
    """Fake PIL and pystray; returns the fonts truetype is allowed to load."""
    fonts: dict[str, bool] = {"truetype": True}

    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image.new = _Image  # type: ignore[attr-defined]
    draw = types.ModuleType("PIL.ImageDraw")
    draw.Draw = _Draw  # type: ignore[attr-defined]
    font = types.ModuleType("PIL.ImageFont")

    def truetype(name, size):
        if not fonts["truetype"]:
            raise OSError("cannot open resource")
        return ("truetype", name, size)

    font.truetype = truetype  # type: ignore[attr-defined]
    font.load_default = lambda: ("default",)  # type: ignore[attr-defined]
    font.FreeTypeFont = font.ImageFont = object  # type: ignore[attr-defined]
    pil.Image, pil.ImageDraw, pil.ImageFont = image, draw, font  # type: ignore[attr-defined]

    pystray = types.ModuleType("pystray")
    pystray.Menu = _Menu  # type: ignore[attr-defined]
    pystray.MenuItem = _MenuItem  # type: ignore[attr-defined]
    pystray.Icon = _Icon  # type: ignore[attr-defined]

    for name, mod in {
        "PIL": pil, "PIL.Image": image, "PIL.ImageDraw": draw, "PIL.ImageFont": font,
        "pystray": pystray,
    }.items():  # fmt: skip
        monkeypatch.setitem(sys.modules, name, mod)
    return fonts


class _Ctrl:
    """A controller whose status and start/stop answers the test sets."""

    def __init__(self, status=ServerStatus.STOPPED, start=True, stop=True) -> None:
        self.status = status
        self.error_message: str | None = None
        self.url = "http://127.0.0.1:11434"
        self._start, self._stop = start, stop
        self.started = self.stopped = 0

    def start(self) -> bool:
        self.started += 1
        return self._start

    def stop(self) -> bool:
        self.stopped += 1
        return self._stop


@pytest.fixture
def sync_threads(monkeypatch):
    """threading.Thread that records its target instead of starting it, and a
    time.sleep that returns at once (no real waiting in these tests)."""
    targets: list = []

    class _Thread:
        def __init__(self, target, daemon=False, name=None) -> None:
            self.target = target
            assert daemon is True  # never keeps the tray process alive

        def start(self) -> None:
            targets.append(self.target)

    monkeypatch.setattr(threading, "Thread", _Thread)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return targets


# -- the icon ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "color"),
    [
        (ServerStatus.RUNNING, "#22C55E"),
        (ServerStatus.STOPPED, "#808080"),
        (ServerStatus.ERROR, "#EF4444"),
        (ServerStatus.STARTING, "#FFA500"),
    ],
)
def test_the_icon_is_a_coloured_circle_with_a_centred_h(gui, status, color) -> None:
    img = tray_icon._generate_icon_image(status)
    assert (img.mode, img.size, img.color) == ("RGBA", (64, 64), (0, 0, 0, 0))
    ellipse, text = img.draw.calls
    assert ellipse == ("ellipse", [4, 4, 60, 60], color)
    # bbox (0, 2, 20, 26): x = (64 - 20) // 2, y = (64 - 24) // 2 - 2
    assert text == ("text", (22, 18), "H", "white", ("truetype", "Arial", 32))


def test_without_arial_the_default_font_is_used(gui) -> None:
    gui["truetype"] = False
    img = tray_icon._generate_icon_image(ServerStatus.RUNNING)
    assert img.draw.calls[1][-1] == ("default",)


# -- the menu ------------------------------------------------------------------------


def _items(menu: _Menu) -> dict:
    named = {}
    for item in menu.items:
        if isinstance(item, _MenuItem):
            key = item.text if isinstance(item.text, str) else item.text.__name__
            named[key] = item
    return named


def test_the_menu_and_what_is_enabled(gui) -> None:
    ctrl = _Ctrl(ServerStatus.STOPPED)
    menu = tray_icon._build_menu(ctrl)  # type: ignore[arg-type]
    items = _items(menu)
    assert menu.items.count(_Menu.SEPARATOR) == 2
    start, stop = items["Start Server"], items["Stop Server"]
    assert start.enabled(None) is True and stop.enabled(None) is False
    ctrl.status = ServerStatus.ERROR
    assert start.enabled(None) is True
    ctrl.status = ServerStatus.RUNNING
    assert start.enabled(None) is False and stop.enabled(None) is True
    assert items["url_text"].text(None) == "http://127.0.0.1:11434"
    from hfl import __version__

    assert f"HFL v{__version__}" in items


def test_the_status_line_says_why_it_failed(gui) -> None:
    ctrl = _Ctrl(ServerStatus.RUNNING)
    status = _items(tray_icon._build_menu(ctrl))["status_text"]  # type: ignore[arg-type]
    assert status.text(None) == "Status: Running"
    ctrl.status = ServerStatus.ERROR
    assert status.text(None) == "Status: Error"
    ctrl.error_message = "Address already in use"
    assert status.text(None) == "Status: Error (Address already in use)"


def test_start_shows_starting_and_follows_the_status(gui, monkeypatch) -> None:
    scheduled: list = []
    monkeypatch.setattr(tray_icon, "_schedule_icon_update", lambda i, c: scheduled.append((i, c)))
    ctrl = _Ctrl(start=True)
    icon = SimpleNamespace(icon=None)
    _items(tray_icon._build_menu(ctrl))["Start Server"].action(icon, None)  # type: ignore[arg-type]
    assert icon.icon.draw.calls[0][2] == "#FFA500"
    assert scheduled == [(icon, ctrl)]


def test_start_refused_changes_nothing(gui, monkeypatch) -> None:
    monkeypatch.setattr(tray_icon, "_schedule_icon_update", lambda i, c: pytest.fail("scheduled"))
    icon = SimpleNamespace(icon="old")
    ctrl = _Ctrl(start=False)
    _items(tray_icon._build_menu(ctrl))["Start Server"].action(icon, None)  # type: ignore[arg-type]
    assert icon.icon == "old" and ctrl.started == 1


@pytest.mark.parametrize("stopped", [True, False])
def test_stop_turns_the_icon_grey_only_when_it_stopped(gui, stopped) -> None:
    icon = SimpleNamespace(icon="old")
    _items(tray_icon._build_menu(_Ctrl(stop=stopped)))["Stop Server"].action(icon, None)  # type: ignore[arg-type]
    if stopped:
        assert icon.icon.draw.calls[0][2] == "#808080"
    else:
        assert icon.icon == "old"


def test_exit_stops_the_server_then_the_icon(gui) -> None:
    ctrl = _Ctrl(ServerStatus.RUNNING)
    icon = MagicMock()
    _items(tray_icon._build_menu(ctrl))["Exit"].action(icon, None)  # type: ignore[arg-type]
    assert ctrl.stopped == 1
    icon.stop.assert_called_once()


# -- the icon follows the server ---------------------------------------------------------


def test_the_icon_updates_once_the_status_settles(gui, sync_threads) -> None:
    ctrl = _Ctrl(ServerStatus.STARTING)
    icon = SimpleNamespace(icon=None)
    tray_icon._schedule_icon_update(icon, ctrl)  # type: ignore[arg-type]
    (update,) = sync_threads
    assert icon.icon is None  # only on its thread
    ctrl.status = ServerStatus.RUNNING
    update()
    assert icon.icon.draw.calls[0][2] == "#22C55E"


def test_a_status_that_never_settles_still_updates_after_the_wait(
    gui, sync_threads, monkeypatch
) -> None:
    naps: list[float] = []
    monkeypatch.setattr(time, "sleep", naps.append)
    ctrl = _Ctrl(ServerStatus.STARTING)
    icon = SimpleNamespace(icon=None)
    tray_icon._schedule_icon_update(icon, ctrl)  # type: ignore[arg-type]
    sync_threads[0]()
    assert sum(naps) == 15.0  # 30 × 0.5 s, then gives up
    assert icon.icon.draw.calls[0][2] == "#FFA500"


# -- a macOS quit stops the server ----------------------------------------------------------


def test_off_macos_nothing_is_observed(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert tray_icon._stop_on_macos_quit(_Ctrl()) is None  # type: ignore[arg-type]


def test_macos_without_appkit_is_not_observed(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "AppKit", None)
    assert tray_icon._stop_on_macos_quit(_Ctrl()) is None  # type: ignore[arg-type]


@pytest.fixture
def appkit(monkeypatch):
    observed: list[tuple] = []
    appkit = types.ModuleType("AppKit")
    appkit.NSApplicationWillTerminateNotification = "WillTerminate"  # type: ignore[attr-defined]
    foundation = types.ModuleType("Foundation")

    class _Center:
        def addObserverForName_object_queue_usingBlock_(self, name, obj, queue, block):
            observed.append((name, obj, queue, block))
            return "observer-token"

    foundation.NSNotificationCenter = SimpleNamespace(defaultCenter=_Center)  # type: ignore[attr-defined]
    unloaded: list[int] = []
    llama = types.ModuleType("hfl.engine.llama_cpp")
    llama.unload_all = lambda: unloaded.append(1)  # type: ignore[attr-defined]
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setitem(sys.modules, "Foundation", foundation)
    monkeypatch.setitem(sys.modules, "hfl.engine.llama_cpp", llama)
    return SimpleNamespace(observed=observed, unloaded=unloaded, llama=llama)


def test_a_macos_quit_stops_the_server_and_unloads_models(appkit) -> None:
    ctrl = _Ctrl(ServerStatus.RUNNING)
    assert tray_icon._stop_on_macos_quit(ctrl) == "observer-token"  # type: ignore[arg-type]
    ((name, obj, queue, block),) = appkit.observed
    assert (name, obj, queue) == ("WillTerminate", None, None)
    block("notification")
    assert ctrl.stopped == 1 and appkit.unloaded == [1]


def test_a_macos_quit_survives_failures_while_stopping(appkit) -> None:
    ctrl = _Ctrl(ServerStatus.RUNNING)

    def broken() -> bool:
        raise RuntimeError("already gone")

    ctrl.stop = broken  # type: ignore[method-assign]
    appkit.llama.unload_all = lambda: (_ for _ in ()).throw(RuntimeError("metal"))
    tray_icon._stop_on_macos_quit(ctrl)  # type: ignore[arg-type]
    appkit.observed[0][3]("notification")  # must not raise into AppKit


# -- the tray --------------------------------------------------------------------------------


def test_the_tray_runs_an_icon_for_the_current_status(gui, monkeypatch, caplog) -> None:
    observers: list = []
    monkeypatch.setattr(tray_icon, "_stop_on_macos_quit", lambda c: observers.append(c) or "obs")
    ctrl = _Ctrl(ServerStatus.ERROR)
    tray = tray_icon.HFLTrayIcon(ctrl)  # type: ignore[arg-type]
    tray.stop()  # before run: nothing to stop, no error
    with caplog.at_level(logging.INFO, logger="hfl.tray.icon"):
        tray.run()
    icon = tray._icon
    assert (icon.name, icon.title, icon.ran) == ("hfl", "HFL Server", 1)
    assert icon.icon.draw.calls[0][2] == "#EF4444"
    assert isinstance(icon.menu, _Menu)
    assert observers == [ctrl] and tray._terminate_observer == "obs"
    assert "Starting tray icon" in caplog.text
    tray.stop()
    assert icon.stopped == 1


def test_run_tray_starts_the_server_and_follows_its_icon(gui, monkeypatch, sync_threads) -> None:
    made: list[dict] = []
    ctrl = _Ctrl(ServerStatus.STARTING)

    def controller(**kw):
        made.append(kw)
        return ctrl

    scheduled: list = []
    monkeypatch.setattr(tray_icon, "TrayServerController", controller)
    monkeypatch.setattr(tray_icon, "_stop_on_macos_quit", lambda c: None)
    monkeypatch.setattr(tray_icon, "_schedule_icon_update", lambda i, c: scheduled.append((i, c)))
    tray_icon.run_tray(host="0.0.0.0", port=8080, api_key="k", model="m", log_level="debug",
                       json_logs=True)  # fmt: skip
    assert made == [{"host": "0.0.0.0", "port": 8080, "api_key": "k", "model": "m",
                     "log_level": "debug", "json_logs": True}]  # fmt: skip
    assert ctrl.started == 1
    (deferred,) = sync_threads
    deferred()  # run() has created the icon by now
    assert len(scheduled) == 1 and scheduled[0][1] is ctrl and isinstance(scheduled[0][0], _Icon)


def test_run_tray_whose_icon_never_appears_gives_up(gui, monkeypatch, sync_threads) -> None:
    ctrl = _Ctrl()
    monkeypatch.setattr(tray_icon, "TrayServerController", lambda **kw: ctrl)
    monkeypatch.setattr(tray_icon, "_schedule_icon_update", lambda i, c: pytest.fail("scheduled"))
    monkeypatch.setattr(tray_icon.HFLTrayIcon, "run", lambda self: None)  # never builds an icon
    tray_icon.run_tray()
    sync_threads[0]()  # waits its 20 turns, schedules nothing


def test_run_tray_without_auto_start(gui, monkeypatch, sync_threads) -> None:
    ctrl = _Ctrl()
    monkeypatch.setattr(tray_icon, "TrayServerController", lambda **kw: ctrl)
    monkeypatch.setattr(tray_icon, "_stop_on_macos_quit", lambda c: None)
    tray_icon.run_tray(auto_start=False)
    assert ctrl.started == 0 and sync_threads == []


# -- the controller's preload, refused by the memory budget ------------------------------------


def test_a_preload_over_the_memory_budget_is_skipped(monkeypatch, caplog) -> None:
    manifest = SimpleNamespace(name="big", local_path="/models/big")
    registry = MagicMock()
    registry.get.return_value = manifest
    state = SimpleNamespace(context_size_override=4096, engine=None, current_model=None)
    seen: list[tuple] = []

    def check(path, n_ctx):
        seen.append((path, n_ctx))
        plan = SimpleNamespace(fits=False, used_after=40 * 1024**3)
        return SimpleNamespace(plan=plan, footprint=30 * 1024**3, budget=0.8)

    ctrl = TrayServerController(model="big")
    with (
        patch("hfl.models.registry.ModelRegistry", return_value=registry),
        patch("hfl.api.state.get_state", return_value=state),
        patch("hfl.engine.residency.check_standalone_load", side_effect=check),
        patch("hfl.engine.selector.select_engine") as select,
        caplog.at_level(logging.WARNING, logger="hfl.tray.controller"),
    ):
        ctrl._preload_model()
    assert seen == [("/models/big", 4096)]
    select.assert_not_called()
    assert state.engine is None
    assert "Not pre-loading big" in caplog.text and "HFL_MEMORY_BUDGET (80%)" in caplog.text

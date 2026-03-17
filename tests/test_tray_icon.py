# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for HFLTrayIcon and icon utilities."""

from unittest.mock import MagicMock, patch

import pytest

from hfl.tray.controller import ServerStatus, TrayServerController


class TestGenerateIconImage:
    """Test icon image generation."""

    @pytest.fixture(autouse=True)
    def _check_pillow(self):
        pytest.importorskip("PIL", reason="Pillow not installed")

    def test_generates_valid_image_for_each_status(self):
        from hfl.tray.icon import _generate_icon_image

        for status in ServerStatus:
            img = _generate_icon_image(status)
            assert img.size == (64, 64)
            assert img.mode == "RGBA"

    def test_running_icon_has_green_pixels(self):
        from hfl.tray.icon import _generate_icon_image

        img = _generate_icon_image(ServerStatus.RUNNING)
        pixels = list(img.getdata())
        # The center should be green-ish (#22C55E = 34, 197, 94)
        center = pixels[32 * 64 + 32]  # center pixel
        assert center[1] > 100  # Green channel is dominant

    def test_stopped_icon_has_gray_pixels(self):
        from hfl.tray.icon import _generate_icon_image

        img = _generate_icon_image(ServerStatus.STOPPED)
        pixels = list(img.getdata())
        center = pixels[32 * 64 + 32]
        # Gray: R ~= G ~= B ~= 128
        assert abs(center[0] - center[1]) < 30
        assert abs(center[1] - center[2]) < 30

    def test_error_icon_has_red_pixels(self):
        from hfl.tray.icon import _generate_icon_image

        img = _generate_icon_image(ServerStatus.ERROR)
        pixels = list(img.getdata())
        center = pixels[32 * 64 + 32]
        assert center[0] > 150  # Red channel is dominant


class TestBuildMenu:
    """Test menu construction."""

    @pytest.fixture(autouse=True)
    def _check_pystray(self):
        pytest.importorskip("pystray", reason="pystray not installed")

    def test_menu_has_expected_items(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        menu = _build_menu(ctrl)
        # pystray.Menu items are accessible
        items = list(menu)
        # Should have: Start, Stop, separator, Status, URL, separator, Exit
        assert len(items) == 7

    def test_start_enabled_when_stopped(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl._status = ServerStatus.STOPPED
        menu = _build_menu(ctrl)
        items = list(menu)
        start_item = items[0]
        # pystray resolves callables via property
        assert start_item.enabled is True

    def test_start_disabled_when_running(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl._status = ServerStatus.RUNNING
        menu = _build_menu(ctrl)
        items = list(menu)
        start_item = items[0]
        assert start_item.enabled is False

    def test_stop_enabled_when_running(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl._status = ServerStatus.RUNNING
        menu = _build_menu(ctrl)
        items = list(menu)
        stop_item = items[1]
        assert stop_item.enabled is True

    def test_stop_disabled_when_stopped(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl._status = ServerStatus.STOPPED
        menu = _build_menu(ctrl)
        items = list(menu)
        stop_item = items[1]
        assert stop_item.enabled is False

    def test_start_enabled_when_error(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl._status = ServerStatus.ERROR
        menu = _build_menu(ctrl)
        items = list(menu)
        start_item = items[0]
        assert start_item.enabled is True

    def test_status_text_shows_running(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl._status = ServerStatus.RUNNING
        menu = _build_menu(ctrl)
        items = list(menu)
        status_item = items[3]
        # pystray resolves text callable via property
        assert "Running" in status_item.text

    def test_status_text_shows_error_message(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl._status = ServerStatus.ERROR
        ctrl._error_message = "Port in use"
        menu = _build_menu(ctrl)
        items = list(menu)
        status_item = items[3]
        assert "Error" in status_item.text
        assert "Port in use" in status_item.text

    def test_url_text(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController(host="0.0.0.0", port=8080)
        menu = _build_menu(ctrl)
        items = list(menu)
        url_item = items[4]
        assert url_item.text == "http://0.0.0.0:8080"


class TestMenuCallbacks:
    """Test menu callback behavior."""

    @pytest.fixture(autouse=True)
    def _check_pystray(self):
        pytest.importorskip("pystray", reason="pystray not installed")

    @patch("hfl.tray.icon._schedule_icon_update")
    def test_start_callback_calls_controller_start(self, mock_schedule):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl.start = MagicMock(return_value=True)

        menu = _build_menu(ctrl)
        items = list(menu)
        start_item = items[0]

        mock_icon = MagicMock()
        # pystray's __call__(icon) calls action(icon, self)
        start_item(mock_icon)

        ctrl.start.assert_called_once()

    def test_stop_callback_calls_controller_stop(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl._status = ServerStatus.RUNNING
        ctrl.stop = MagicMock(return_value=True)

        menu = _build_menu(ctrl)
        items = list(menu)
        stop_item = items[1]

        mock_icon = MagicMock()
        stop_item(mock_icon)

        ctrl.stop.assert_called_once()

    def test_exit_callback_stops_server_and_icon(self):
        from hfl.tray.icon import _build_menu

        ctrl = TrayServerController()
        ctrl.stop = MagicMock(return_value=True)

        menu = _build_menu(ctrl)
        items = list(menu)
        exit_item = items[6]

        mock_icon = MagicMock()
        exit_item(mock_icon)

        ctrl.stop.assert_called_once()
        mock_icon.stop.assert_called_once()


class TestHFLTrayIcon:
    """Test HFLTrayIcon class."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        pytest.importorskip("pystray", reason="pystray not installed")
        pytest.importorskip("PIL", reason="Pillow not installed")

    @patch("pystray.Icon")
    def test_run_creates_icon(self, mock_icon_cls):
        from hfl.tray.icon import HFLTrayIcon

        ctrl = TrayServerController()
        tray = HFLTrayIcon(ctrl)

        mock_icon = MagicMock()
        mock_icon_cls.return_value = mock_icon

        tray.run()

        mock_icon_cls.assert_called_once()
        call_kwargs = mock_icon_cls.call_args
        assert call_kwargs.kwargs.get("name") or call_kwargs[1].get("name") == "hfl"
        mock_icon.run.assert_called_once()

    @patch("pystray.Icon")
    def test_stop_calls_icon_stop(self, mock_icon_cls):
        from hfl.tray.icon import HFLTrayIcon

        ctrl = TrayServerController()
        tray = HFLTrayIcon(ctrl)

        mock_icon = MagicMock()
        mock_icon_cls.return_value = mock_icon
        tray.run()

        tray.stop()
        mock_icon.stop.assert_called()

    def test_stop_before_run_is_safe(self):
        from hfl.tray.icon import HFLTrayIcon

        ctrl = TrayServerController()
        tray = HFLTrayIcon(ctrl)
        tray.stop()  # Should not raise


class TestRunTray:
    """Test the convenience run_tray function."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        pytest.importorskip("pystray", reason="pystray not installed")
        pytest.importorskip("PIL", reason="Pillow not installed")

    @patch("hfl.tray.icon.HFLTrayIcon")
    @patch("hfl.tray.icon.TrayServerController")
    def test_run_tray_auto_start(self, mock_ctrl_cls, mock_tray_cls):
        from hfl.tray.icon import run_tray

        mock_ctrl = MagicMock()
        mock_ctrl_cls.return_value = mock_ctrl
        mock_tray = MagicMock()
        mock_tray_cls.return_value = mock_tray

        run_tray(host="0.0.0.0", port=9090, auto_start=True)

        mock_ctrl.start.assert_called_once()
        mock_tray.run.assert_called_once()

    @patch("hfl.tray.icon.HFLTrayIcon")
    @patch("hfl.tray.icon.TrayServerController")
    def test_run_tray_no_auto_start(self, mock_ctrl_cls, mock_tray_cls):
        from hfl.tray.icon import run_tray

        mock_ctrl = MagicMock()
        mock_ctrl_cls.return_value = mock_ctrl
        mock_tray = MagicMock()
        mock_tray_cls.return_value = mock_tray

        run_tray(auto_start=False)

        mock_ctrl.start.assert_not_called()
        mock_tray.run.assert_called_once()

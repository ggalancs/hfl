# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for TrayServerController."""

import threading
from unittest.mock import MagicMock, patch

from hfl.tray.controller import ServerStatus, TrayServerController


class TestTrayServerControllerInit:
    """Test controller initialization."""

    def test_default_values(self):
        ctrl = TrayServerController()
        assert ctrl.host == "127.0.0.1"
        assert ctrl.port == 11434
        assert ctrl.api_key is None
        assert ctrl.model is None
        assert ctrl.log_level == "info"
        assert ctrl.json_logs is False
        assert ctrl.status == ServerStatus.STOPPED
        assert ctrl.is_running is False
        assert ctrl.error_message is None

    def test_custom_values(self):
        ctrl = TrayServerController(
            host="0.0.0.0",
            port=8080,
            api_key="secret",
            model="test-model",
            log_level="debug",
            json_logs=True,
        )
        assert ctrl.host == "0.0.0.0"
        assert ctrl.port == 8080
        assert ctrl.api_key == "secret"
        assert ctrl.model == "test-model"
        assert ctrl.log_level == "debug"
        assert ctrl.json_logs is True

    def test_url_property(self):
        ctrl = TrayServerController(host="localhost", port=9090)
        assert ctrl.url == "http://localhost:9090"


class TestTrayServerControllerStart:
    """Test server start behavior."""

    @patch("hfl.tray.controller.TrayServerController._run_server")
    def test_start_returns_true(self, mock_run):
        ctrl = TrayServerController()
        result = ctrl.start()
        assert result is True
        assert ctrl.status == ServerStatus.STARTING

    @patch("hfl.tray.controller.TrayServerController._run_server")
    def test_start_creates_daemon_thread(self, mock_run):
        ctrl = TrayServerController()
        ctrl.start()
        assert ctrl._thread is not None
        assert ctrl._thread.daemon is True
        assert ctrl._thread.name == "hfl-server"
        # Wait for thread to finish (mock returns immediately)
        ctrl._thread.join(timeout=2)

    @patch("hfl.tray.controller.TrayServerController._run_server")
    def test_start_when_already_running_returns_false(self, mock_run):
        ctrl = TrayServerController()
        ctrl._status = ServerStatus.RUNNING
        result = ctrl.start()
        assert result is False

    @patch("hfl.tray.controller.TrayServerController._run_server")
    def test_start_when_starting_returns_false(self, mock_run):
        ctrl = TrayServerController()
        ctrl._status = ServerStatus.STARTING
        result = ctrl.start()
        assert result is False

    @patch("hfl.tray.controller.TrayServerController._run_server")
    def test_start_clears_previous_error(self, mock_run):
        ctrl = TrayServerController()
        ctrl._status = ServerStatus.ERROR
        ctrl._error_message = "previous error"
        result = ctrl.start()
        assert result is True
        assert ctrl.error_message is None


class TestTrayServerControllerStop:
    """Test server stop behavior."""

    def test_stop_when_stopped_returns_false(self):
        ctrl = TrayServerController()
        result = ctrl.stop()
        assert result is False

    def test_stop_when_error_returns_false(self):
        ctrl = TrayServerController()
        ctrl._status = ServerStatus.ERROR
        result = ctrl.stop()
        assert result is False

    def test_stop_when_running_signals_exit(self):
        ctrl = TrayServerController()
        ctrl._status = ServerStatus.RUNNING

        mock_server = MagicMock()
        ctrl._server = mock_server
        ctrl._thread = threading.Thread(target=lambda: None)
        ctrl._thread.start()
        ctrl._thread.join()

        result = ctrl.stop()
        assert result is True
        assert mock_server.should_exit is True
        assert ctrl.status == ServerStatus.STOPPED
        assert ctrl._server is None

    def test_stop_when_stopping_returns_false(self):
        ctrl = TrayServerController()
        ctrl._status = ServerStatus.STOPPING
        result = ctrl.stop()
        assert result is False


class TestTrayServerControllerRunServer:
    """Test the _run_server method."""

    @patch("hfl.tray.controller.TrayServerController._preload_model")
    def test_run_server_sets_status_to_running(self, mock_preload):
        """Test that _run_server transitions through RUNNING then to STOPPED."""
        ctrl = TrayServerController(model="test")

        statuses_seen = []
        original_lock = ctrl._lock

        # Track status transitions
        class StatusTracker:
            def __enter__(self):
                original_lock.acquire()
                return self

            def __exit__(self, *args):
                statuses_seen.append(ctrl._status)
                original_lock.release()

        mock_server = MagicMock()

        mock_config = MagicMock()

        with (
            patch("uvicorn.Server", return_value=mock_server),
            patch("uvicorn.Config", return_value=mock_config),
            patch("hfl.tray.controller.asyncio.new_event_loop") as mock_loop_factory,
            patch("hfl.tray.controller.asyncio.set_event_loop"),
            patch("hfl.api.state.get_state") as mock_get_state,
        ):
            mock_loop = MagicMock()
            mock_loop_factory.return_value = mock_loop

            mock_state = MagicMock()
            mock_get_state.return_value = mock_state

            ctrl._run_server()

        # After server.serve() completes, status should be STOPPED (clean exit)
        assert ctrl.status == ServerStatus.STOPPED
        mock_preload.assert_called_once()

    def test_run_server_handles_os_error(self):
        """Test that OSError (port in use) sets ERROR status."""
        ctrl = TrayServerController()

        with (
            patch("uvicorn.Config", side_effect=OSError("Address already in use")),
            patch("hfl.api.state.get_state") as mock_get_state,
        ):
            mock_get_state.return_value = MagicMock()
            ctrl._run_server()

        assert ctrl.status == ServerStatus.ERROR
        assert "Address already in use" in ctrl.error_message

    def test_run_server_handles_generic_exception(self):
        """Test that generic exceptions set ERROR status."""
        ctrl = TrayServerController()

        with (
            patch("uvicorn.Config", side_effect=RuntimeError("unexpected")),
            patch("hfl.api.state.get_state") as mock_get_state,
        ):
            mock_get_state.return_value = MagicMock()
            ctrl._run_server()

        assert ctrl.status == ServerStatus.ERROR
        assert "unexpected" in ctrl.error_message


class TestTrayServerControllerPreload:
    """Test model pre-loading."""

    def test_preload_no_model(self):
        ctrl = TrayServerController()
        # Should not raise
        ctrl._preload_model()

    @patch("hfl.models.registry.ModelRegistry")
    def test_preload_model_not_found(self, mock_registry_cls):
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry_cls.return_value = mock_registry

        ctrl = TrayServerController(model="nonexistent")
        ctrl._preload_model()
        # No error, just a warning

    @patch("hfl.api.state.get_state")
    @patch("hfl.engine.selector.select_engine")
    @patch("hfl.models.registry.ModelRegistry")
    def test_preload_model_success(self, mock_registry_cls, mock_select, mock_get_state):
        mock_manifest = MagicMock()
        mock_manifest.local_path = "/fake/path"
        mock_manifest.name = "test-model"

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_registry_cls.return_value = mock_registry

        mock_engine = MagicMock()
        mock_select.return_value = mock_engine

        mock_state = MagicMock()
        mock_state.context_size_override = 0
        mock_get_state.return_value = mock_state

        ctrl = TrayServerController(model="test-model")
        ctrl._preload_model()

        mock_engine.load.assert_called_once_with("/fake/path", n_ctx=0)
        assert mock_state.engine == mock_engine
        assert mock_state.current_model == mock_manifest

    @patch("hfl.models.registry.ModelRegistry")
    def test_preload_model_failure_does_not_raise(self, mock_registry_cls):
        mock_manifest = MagicMock()
        mock_manifest.local_path = "/fake/path"
        mock_manifest.name = "test-model"

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_manifest
        mock_registry_cls.return_value = mock_registry

        with patch("hfl.engine.selector.select_engine", side_effect=RuntimeError("boom")):
            ctrl = TrayServerController(model="test-model")
            # Should not raise
            ctrl._preload_model()


class TestTrayServerControllerConcurrency:
    """Test thread safety of start/stop operations."""

    @patch("hfl.tray.controller.TrayServerController._run_server")
    def test_concurrent_starts_only_one_succeeds(self, mock_run):
        ctrl = TrayServerController()
        results = []

        def try_start():
            results.append(ctrl.start())

        threads = [threading.Thread(target=try_start) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert results.count(True) == 1
        assert results.count(False) == 4

    def test_stop_without_start_is_safe(self):
        ctrl = TrayServerController()
        assert ctrl.stop() is False
        assert ctrl.status == ServerStatus.STOPPED

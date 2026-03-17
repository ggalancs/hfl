# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Server lifecycle controller for tray integration.

Manages uvicorn server in a background thread, providing
start/stop/status from a non-async context (tray callbacks).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ServerStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class TrayServerController:
    """Manages uvicorn server lifecycle from a non-async context."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        api_key: str | None = None,
        model: str | None = None,
        log_level: str = "info",
        json_logs: bool = False,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.model = model
        self.log_level = log_level
        self.json_logs = json_logs

        self._status = ServerStatus.STOPPED
        self._lock = threading.Lock()
        self._server: object | None = None  # uvicorn.Server
        self._thread: threading.Thread | None = None
        self._error_message: str | None = None

    @property
    def status(self) -> ServerStatus:
        return self._status

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def is_running(self) -> bool:
        return self._status == ServerStatus.RUNNING

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> bool:
        """Start the server in a background thread.

        Returns True if start was initiated, False if already running.
        """
        with self._lock:
            if self._status in (ServerStatus.RUNNING, ServerStatus.STARTING):
                return False
            self._status = ServerStatus.STARTING
            self._error_message = None

        self._thread = threading.Thread(
            target=self._run_server,
            name="hfl-server",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> bool:
        """Stop the server gracefully.

        Returns True if stop was initiated, False if already stopped.
        """
        with self._lock:
            if self._status in (ServerStatus.STOPPED, ServerStatus.STOPPING, ServerStatus.ERROR):
                return False
            self._status = ServerStatus.STOPPING

        if self._server is not None:
            self._server.should_exit = True  # type: ignore[attr-defined]

        if self._thread is not None:
            self._thread.join(timeout=35)  # 30s graceful + 5s margin
            self._thread = None

        with self._lock:
            self._status = ServerStatus.STOPPED
            self._server = None

        logger.info("Server stopped")
        return True

    def _run_server(self) -> None:
        """Run uvicorn server in a new event loop (background thread)."""
        try:
            import uvicorn

            from hfl.api.server import app
            from hfl.api.state import get_state

            # Set API key
            get_state().api_key = self.api_key

            # Pre-load model if specified
            if self.model:
                self._preload_model()

            config = uvicorn.Config(
                app=app,
                host=self.host,
                port=self.port,
                log_level=self.log_level.lower(),
                timeout_graceful_shutdown=30,
            )
            server = uvicorn.Server(config)
            self._server = server

            with self._lock:
                self._status = ServerStatus.RUNNING

            logger.info("Server started on %s:%d", self.host, self.port)

            # Run in a fresh event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(server.serve())
            finally:
                loop.close()

        except OSError as e:
            logger.error("Server failed to start: %s", e)
            with self._lock:
                self._status = ServerStatus.ERROR
                self._error_message = str(e)
        except Exception as e:
            logger.error("Server error: %s", e)
            with self._lock:
                self._status = ServerStatus.ERROR
                self._error_message = str(e)
        finally:
            with self._lock:
                if self._status == ServerStatus.RUNNING:
                    self._status = ServerStatus.STOPPED

    def _preload_model(self) -> None:
        """Pre-load a model before starting the server."""
        from pathlib import Path

        from hfl.api.state import get_state
        from hfl.engine.selector import select_engine
        from hfl.models.registry import ModelRegistry

        if not self.model:
            return

        registry = ModelRegistry()
        manifest = registry.get(self.model)
        if not manifest:
            logger.warning("Model not found for pre-loading: %s", self.model)
            return

        try:
            state = get_state()
            engine = select_engine(Path(manifest.local_path))
            engine.load(manifest.local_path)
            state.engine = engine
            state.current_model = manifest
            logger.info("Model pre-loaded: %s", manifest.name)
        except Exception as e:
            logger.warning("Failed to pre-load model %s: %s", self.model, e)

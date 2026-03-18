# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Cross-platform system tray icon for HFL server.

Uses pystray for macOS/Windows/Linux support.
Icon is generated programmatically with Pillow (no asset files needed).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hfl.tray.controller import ServerStatus, TrayServerController

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Color constants for icon states
_COLORS = {
    ServerStatus.STOPPED: "#808080",  # Gray
    ServerStatus.STARTING: "#FFA500",  # Orange
    ServerStatus.RUNNING: "#22C55E",  # Green
    ServerStatus.STOPPING: "#FFA500",  # Orange
    ServerStatus.ERROR: "#EF4444",  # Red
}

_ICON_SIZE = 64


def _generate_icon_image(status: ServerStatus) -> Any:
    """Generate a tray icon image for the given server status.

    Creates a 64x64 image with a colored circle and "H" letter.

    Returns:
        PIL.Image.Image
    """
    from PIL import Image, ImageDraw, ImageFont

    color = _COLORS.get(status, "#808080")
    img = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw filled circle
    margin = 4
    draw.ellipse(
        [margin, margin, _ICON_SIZE - margin, _ICON_SIZE - margin],
        fill=color,
    )

    # Draw "H" letter centered
    try:
        font = ImageFont.truetype("Arial", 32)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), "H", font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (_ICON_SIZE - text_w) // 2
    y = (_ICON_SIZE - text_h) // 2 - bbox[1]
    draw.text((x, y), "H", fill="white", font=font)

    return img


def _build_menu(controller: TrayServerController) -> Any:
    """Build the tray context menu.

    Returns:
        pystray.Menu
    """
    import pystray

    def on_start(icon: Any, item: Any) -> None:
        if controller.start():
            icon.icon = _generate_icon_image(ServerStatus.STARTING)
            # Poll for status change to update icon
            _schedule_icon_update(icon, controller)

    def on_stop(icon: Any, item: Any) -> None:
        if controller.stop():
            icon.icon = _generate_icon_image(ServerStatus.STOPPED)

    def on_exit(icon: Any, item: Any) -> None:
        controller.stop()
        icon.stop()

    def is_start_enabled(item: Any) -> bool:
        return controller.status == ServerStatus.STOPPED or controller.status == ServerStatus.ERROR

    def is_stop_enabled(item: Any) -> bool:
        return controller.status == ServerStatus.RUNNING

    def status_text(item: Any) -> str:
        status = controller.status.value.capitalize()
        if controller.status == ServerStatus.ERROR and controller.error_message:
            return f"Status: {status} ({controller.error_message})"
        return f"Status: {status}"

    def url_text(item: Any) -> str:
        return controller.url

    return pystray.Menu(
        pystray.MenuItem("Start Server", on_start, enabled=is_start_enabled),
        pystray.MenuItem("Stop Server", on_stop, enabled=is_stop_enabled),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(status_text, None, enabled=False),
        pystray.MenuItem(url_text, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Exit", on_exit),
    )


def _schedule_icon_update(icon: Any, controller: TrayServerController) -> None:
    """Schedule a delayed icon update after status change."""
    import threading

    def _update() -> None:
        # Wait briefly for status to settle
        import time

        for _ in range(30):  # up to 15 seconds
            time.sleep(0.5)
            settled = (ServerStatus.RUNNING, ServerStatus.ERROR, ServerStatus.STOPPED)
            if controller.status in settled:
                break
        icon.icon = _generate_icon_image(controller.status)

    t = threading.Thread(target=_update, daemon=True)
    t.start()


class HFLTrayIcon:
    """Cross-platform system tray icon for HFL."""

    def __init__(self, controller: TrayServerController):
        self._controller = controller
        self._icon: Any = None

    def run(self) -> None:
        """Run the tray icon (blocks the calling thread).

        On macOS, this MUST be called from the main thread.
        """
        import pystray

        self._icon = pystray.Icon(
            name="hfl",
            icon=_generate_icon_image(self._controller.status),
            title="HFL Server",
            menu=_build_menu(self._controller),
        )

        logger.info("Starting tray icon")
        self._icon.run()

    def stop(self) -> None:
        """Stop the tray icon."""
        if self._icon is not None:
            self._icon.stop()


def run_tray(
    host: str = "127.0.0.1",
    port: int = 11434,
    api_key: str | None = None,
    model: str | None = None,
    log_level: str = "info",
    json_logs: bool = False,
    auto_start: bool = True,
) -> None:
    """Convenience function to create controller + tray and run.

    Args:
        host: Server host address
        port: Server port
        api_key: Optional API key
        model: Optional model to pre-load
        log_level: Logging level
        json_logs: Enable JSON log format
        auto_start: Start server automatically on launch
    """
    controller = TrayServerController(
        host=host,
        port=port,
        api_key=api_key,
        model=model,
        log_level=log_level,
        json_logs=json_logs,
    )

    tray = HFLTrayIcon(controller)

    if auto_start:
        controller.start()
        # Schedule icon update so it transitions from STARTING to RUNNING
        if tray._icon is None:
            # Icon not yet created; run() will create it with STARTING status.
            # We schedule the update after a brief delay to give run() time to set _icon.
            import threading

            def _deferred_update() -> None:
                import time

                # Wait for the icon to be created by run()
                for _ in range(20):
                    time.sleep(0.25)
                    if tray._icon is not None:
                        break
                if tray._icon is not None:
                    _schedule_icon_update(tray._icon, controller)

            threading.Thread(target=_deferred_update, daemon=True).start()

    tray.run()

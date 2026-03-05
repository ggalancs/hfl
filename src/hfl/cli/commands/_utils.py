# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Shared utilities for CLI commands."""

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from rich.console import Console
from rich.status import Status

from hfl.i18n import t

if TYPE_CHECKING:
    from hfl.converter.formats import ModelType
    from hfl.models.manifest import ModelManifest

console = Console()


@contextmanager
def progress_spinner(message: str) -> Iterator[Status]:
    """Context manager for showing a spinner during long operations.

    Usage:
        with progress_spinner("Loading model..."):
            do_long_operation()
    """
    with console.status(message, spinner="dots") as status:
        yield status


def show_progress(message: str, finished_message: str | None = None) -> Any:
    """Decorator for functions that take a while.

    Args:
        message: Message to show during execution
        finished_message: Optional message when done (replaces spinner)
    """

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with progress_spinner(message):
                result = func(*args, **kwargs)
            if finished_message:
                console.print(finished_message)
            return result

        return wrapper

    return decorator


def get_model_type(manifest: "ModelManifest") -> "ModelType":
    """Get model type from manifest or detect it.

    Returns:
        ModelType enum value
    """
    from pathlib import Path

    from hfl.converter.formats import ModelType, detect_model_type

    # Try to get from manifest first
    if manifest.model_type:
        try:
            return ModelType(manifest.model_type)
        except ValueError:
            pass

    # Detect from path
    return detect_model_type(Path(manifest.local_path))


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    if size_bytes == 0:
        return t("inspect.na")
    gb = size_bytes / (1024**3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024**2)
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{size_bytes} B"


def get_key() -> str:
    """Read a key without requiring Enter to be pressed."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def extract_params_from_name(model_id: str) -> str | None:
    """Extract the number of parameters from the model name (e.g.: '70B', '7B')."""
    name = model_id.lower()
    # Patterns: 70b, 7b, 1.5b, 0.5b, 405b, etc.
    patterns = [
        r"(\d+\.?\d*)b(?:[-_]|$)",  # 70b, 7b, 1.5b
        r"(\d+)b-",  # 70b-instruct
        r"-(\d+\.?\d*)b",  # model-7b
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1) + "B"
    return None


def estimate_model_size(params_str: str | None, quantization: str = "Q4") -> str:
    """Estimate model size based on parameters and quantization."""
    if not params_str:
        return "?"

    try:
        # Parse params (e.g., "70B" -> 70, "1.5B" -> 1.5)
        params = float(params_str.replace("B", "").replace("b", ""))

        # Bytes per parameter depends on quantization
        # Q4_K_M ~ 4.5 bits/param, Q8_0 ~ 8 bits/param, F16 ~ 16 bits
        bits_per_param = {
            "Q2": 2.5,
            "Q3": 3.5,
            "Q4": 4.5,
            "Q5": 5.5,
            "Q6": 6.5,
            "Q8": 8.0,
            "F1": 16.0,  # F16
        }
        bits = bits_per_param.get(quantization[:2].upper(), 4.5)
        size_gb = (params * 1e9 * bits / 8) / (1024**3)

        if size_gb >= 100:
            return f"{size_gb:.0f}GB"
        elif size_gb >= 10:
            return f"{size_gb:.0f}GB"
        else:
            return f"{size_gb:.1f}GB"
    except Exception:
        return "?"


def get_params_value(model_id: str) -> float | None:
    """Extract the numeric value of parameters from the name (e.g.: 70 for '70B')."""
    params = extract_params_from_name(model_id)
    if not params:
        return None
    try:
        return float(params.replace("B", "").replace("b", ""))
    except ValueError:
        return None


def display_model_row(model: Any, index: int, show_index: bool = True) -> None:
    """Display a formatted model row."""
    # Get model information
    model_id = model.id
    downloads = getattr(model, "downloads", 0) or 0
    likes = getattr(model, "likes", 0) or 0

    # Detect if it has GGUF
    has_gguf = False
    siblings = getattr(model, "siblings", None)
    if siblings:
        has_gguf = any(s.rfilename.endswith(".gguf") for s in siblings)

    pipeline_tag = getattr(model, "pipeline_tag", None)

    # Format icon
    format_icon = "[green]●[/] GGUF" if has_gguf else "[dim]○[/] HF"

    # Format downloads
    if downloads >= 1_000_000:
        dl_str = f"{downloads / 1_000_000:.1f}M"
    elif downloads >= 1_000:
        dl_str = f"{downloads / 1_000:.1f}K"
    else:
        dl_str = str(downloads)

    # Extract parameters and estimate size
    params = extract_params_from_name(model_id)
    size_q4 = estimate_model_size(params, "Q4")
    size_str = f"[magenta]~{size_q4}[/]" if params else ""

    # Index number
    idx_str = f"[dim]{index:3}.[/] " if show_index else ""

    console.print(
        f"{idx_str}[bold cyan]{model_id}[/]  "
        f"{format_icon}  "
        f"[yellow]↓{dl_str}[/]  "
        f"[red]♥{likes}[/]  "
        f"{size_str}  "
        f"[dim]{pipeline_tag or ''}[/]"
    )

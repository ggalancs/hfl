# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl install llama-server``: llama.cpp's server, without installing llama.cpp.

GGUF models answer several requests at once only through ``llama-server``,
which had to be installed apart (Homebrew, winget, a build). This fetches
the official build of the pinned release (``hfl.engine.llama_server_dist``)
after saying what and from where, and asks first: it puts an executable on
the machine.
"""

from __future__ import annotations

import shutil

from rich.console import Console
from rich.markup import escape

from hfl.i18n import t
from hfl.utils.terminal import stdin_is_terminal

console = Console()


def install_llama_server(*, variant: str | None, assume_yes: bool, force: bool) -> int:
    """The command's work; its exit code."""
    import typer

    from hfl.engine import llama_server_dist as dist

    own = shutil.which("llama-server") or dist.bundled_binary()
    if own and not force:
        console.print(t("install.llama_server.already", path=escape(own)))
        return 0
    managed = dist.managed_binary()
    if managed and not force:
        console.print(t("install.llama_server.installed_before", path=escape(managed)))
        return 0
    offered = dist.variants()
    variant = variant or dist.default_variant()
    if variant not in offered:
        builds = ", ".join(offered) or "-"
        message = t("install.llama_server.no_build", variant=variant or "", offered=builds)
        console.print(f"[red]{message}[/]")
        return 1
    console.print(
        t(
            "install.llama_server.what",
            release=dist.RELEASE,
            variant=variant,
            mb=dist.size_of(variant) / 1e6,
            folder=escape(str(dist.install_dir())),
        )
    )
    if not assume_yes:
        if not stdin_is_terminal():
            console.print(f"[yellow]{t('install.llama_server.needs_yes')}[/]")
            return 1
        if not typer.confirm(t("install.llama_server.confirm"), default=True):
            return 1
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    with Progress(
        "{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn(), console=console
    ) as bar:
        task = bar.add_task("llama.cpp", total=dist.size_of(variant))
        try:
            server = dist.install(variant, progress=lambda n: bar.advance(task, n))
        except dist.InstallError as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            return 1
    console.print(f"[green]{t('install.llama_server.done', path=escape(str(server)))}[/]")
    return 0

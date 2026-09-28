#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Record a real ``hfl run`` and draw it as an animated terminal SVG.

    python scripts/demo_svg.py docs/assets/hfl-run-demo.svg

Runs ``hfl run <model>`` in a pseudo-terminal, in a fresh HFL home (so the
download shows), types one question, and records what the terminal printed
and when. The SVG shows those lines in that order; long waits (the
download) are shortened, which the caption says. Nothing is typed into the
picture that was not typed into the run, and no line is written by hand.
POSIX only (pty); needs ``pip install pyte`` (a terminal emulator).
"""

from __future__ import annotations

import html
import os
import pty
import re
import select
import sys
import tempfile
import time
from pathlib import Path

MODEL = "hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"  # Apache-2.0, ~0.5 GB
QUESTION = "What is the capital of France? Answer in one sentence."
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
WIDTH = 92  # columns


def record(hfl: str) -> list[tuple[float, str]]:
    """``(seconds, line)`` for every line of the terminal at the end of the
    run, each timed by when it first showed its final text. The output goes
    through a terminal emulator (pyte): rich redraws progress bars and the
    streamed answer in place, so splitting the raw bytes on newlines loses
    most of it."""
    import pyte

    home = tempfile.mkdtemp(prefix="hfl-demo-")
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("HFL_", "OLLAMA_"))},
        "HFL_HOME": home, "HF_HOME": os.path.join(home, "hf"), "HFL_LANG": "en",
        "NO_COLOR": "1", "COLUMNS": str(WIDTH), "LINES": "60", "TERM": "xterm",
    }  # fmt: skip
    env.pop("HF_TOKEN", None)
    screen = pyte.Screen(WIDTH, 200)
    stream = pyte.Stream(screen)
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(hfl, [hfl, "run", MODEL], env)
    start, typed = time.monotonic(), 0
    snapshots: list[tuple[float, list[str]]] = []
    deadline = start + 900
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(fd, 4096).decode(errors="replace")
            except OSError:
                break
            if not chunk:
                break
            stream.feed(chunk.replace("\n", "\r\n") if "\r\n" not in chunk else chunk)
            snapshots.append((time.monotonic() - start, [r.rstrip() for r in screen.display]))
        cursor_line = screen.display[screen.cursor.y].rstrip()
        if cursor_line.endswith(">>>") and typed < 2:
            time.sleep(0.8)  # a person reads the prompt before typing
            os.write(fd, ((QUESTION, "/exit")[typed] + "\n").encode())
            typed += 1
    os.waitpid(pid, 0)
    if not snapshots:
        return []
    final = snapshots[-1][1]
    last = max((i for i, row in enumerate(final) if row.strip()), default=-1)
    out = []
    for i in range(last + 1):
        when = next((t for t, rows in snapshots if rows[i] == final[i]), snapshots[-1][0])
        out.append((when, final[i]))
    return out


def svg(lines: list[tuple[float, str]], title: str) -> str:
    """The terminal, each line appearing at its (shortened) time."""
    shown: list[tuple[float, str]] = []
    clock, last = 0.4, 0.0
    for t, line in lines:
        clock += min(max(t - last, 0.05), 1.2)  # waits over 1.2 s are shortened
        last = t
        shown.append((clock, line[:WIDTH]))
    line_h, top = 19, 58
    height = top + line_h * len(shown) + 24
    total = shown[-1][0] + 4 if shown else 4
    # One cycle for every line, each with the moment it appears: lines
    # started with their own delays would drift apart after the first loop.
    rows, frames = [], []
    for i, (at, line) in enumerate(shown):
        pct = 100 * at / total
        frames.append(
            f"@keyframes a{i} {{ 0%, {pct:.2f}% {{ opacity: 0; }} "
            f"{pct + 0.01:.2f}%, 97% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}"
        )
        rows.append(
            f'<text x="20" y="{top + i * line_h}" class="l" '
            f'style="animation-name:a{i}">{html.escape(line) or " "}</text>'
        )
    font = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}" '
        f'viewBox="0 0 980 {height}" role="img" aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        f".l {{ font: 13.5px {font}; fill: #c9d1d9; white-space: pre; opacity: 0;",
        f"     animation-duration: {total:.1f}s; animation-iteration-count: infinite; }}",
        *frames,
        "</style>",
        f'<rect x="0.5" y="0.5" width="979" height="{height - 1}" rx="10" fill="#0d1117" '
        'stroke="#30363d"/>',
        '<rect x="0.5" y="0.5" width="979" height="36" rx="10" fill="#161b22"/>',
        '<rect x="0.5" y="26" width="979" height="11" fill="#161b22"/>',
        '<circle cx="22" cy="18" r="6" fill="#ff5f57"/>',
        '<circle cx="42" cy="18" r="6" fill="#febc2e"/>',
        '<circle cx="62" cy="18" r="6" fill="#28c840"/>',
        '<text x="490" y="23" text-anchor="middle" style="font: 12px ui-sans-serif, '
        f'system-ui, sans-serif; fill: #8b949e">{html.escape(title)}</text>',
    ]
    return "\n".join([*head, *rows, "</svg>", ""])


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    hfl = str(Path(sys.executable).with_name("hfl"))
    lines = record(hfl)
    # Abridged, never rewritten: lines are only left out — the Hub's
    # "unauthenticated requests" warning (and its wrapped end), and the
    # temporary folder the download went to.
    kept: list[tuple[float, str]] = []
    skipping = False
    for t, x in lines:
        if x.startswith("Downloaded to:"):
            skipping = True
            continue
        if skipping and not x.startswith("Model ready"):
            continue
        skipping = False
        if x.strip() and "unauthenticated" not in x and "rate limits" not in x:
            kept.append((t, x))
    kept.insert(0, (0.0, f"$ hfl run {MODEL}"))
    title = "hfl run — real output, abridged, waits shortened"
    Path(sys.argv[1]).write_text(svg(kept, title), encoding="utf-8")
    for t, line in kept:
        print(f"{t:6.1f}s  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Section C: each install extra, from the built wheel into a fresh venv.

The wheel is the one ``--setup`` built (``<work>/wheel.txt``). Each extra
gets a venv of its own; its modules must import and ``hfl version`` run.
"""

from __future__ import annotations

import os
import subprocess
import sys

from local_audit import APPLE_SILICON, Audit, Uncheckable, check, expect

MODULES = {
    "all": ["llama_cpp", "transformers", "mlx_lm", "faster_whisper", "diffusers", "mcp", "vllm"],
    "audio": ["sounddevice", "soundfile"],
    "build": ["PyInstaller"],
    "convert": ["gguf"],
    "coqui": ["TTS"],
    "dev": ["pytest", "mypy"],
    "imagegen": ["diffusers", "torch"],
    "llama": ["llama_cpp"],
    "mcp": ["mcp"],
    "mlx": ["mlx_lm", "llguidance"],
    "otel": ["opentelemetry.sdk"],
    "rocm": ["llama_cpp"],
    "stt": ["faster_whisper"],
    "transformers": ["transformers", "torch", "accelerate", "llguidance"],
    "tray": ["pystray", "PIL"],
    "tts": ["transformers", "torchaudio", "soundfile"],
    "vllm": ["vllm"],
    "vulkan": ["llama_cpp"],
    "structured": ["llguidance"],
}
LINUX_CUDA = {"vllm"}
# Extras added after the audit was first run: numbered after the others, so
# the existing checks keep their ids.
LATER = ("structured",)
# Each extra's check id, as the loop at the bottom numbers them.
EXTRA_ID = {
    name: f"C{index}" for index, name in enumerate([*sorted(set(MODULES) - set(LATER)), *LATER], 1)
}


def _extra(a: Audit, extra: str) -> str:
    wheel = a.wheel()
    venv = a.work / "extras" / extra
    installed = venv / ".audit-installed"
    if not installed.exists():
        # New, or an install that never finished (a full disk left [coqui]
        # with half a torch: "module 'torch' has no attribute 'Tensor'"
        # on every later run): start this venv again.
        subprocess.run(["uv", "venv", "-q", "--clear", "--python", "3.12", str(venv)], check=True)
    log = a.work / "logs" / f"extra-{extra}.log"
    done = subprocess.run(
        # --reinstall-package: a rebuilt wheel keeps its version; install it anyway.
        [
            "uv",
            "pip",
            "install",
            "--reinstall-package",
            "hfl",
            "--python",
            str(venv / "bin" / "python"),
            f"{wheel}[{extra}]",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    log.write_text(done.stdout + done.stderr)
    if done.returncode == 0:
        installed.write_text("")
    if done.returncode != 0:
        tail = " ".join((done.stdout + done.stderr).split())[-300:]
        if extra in LINUX_CUDA:
            raise Uncheckable(f"does not install here (Linux + CUDA only): {tail}")
        expect(False, f"pip install hfl[{extra}] failed: {tail}")
    linux = sys.platform.startswith("linux")
    modules = [
        m for m in MODULES[extra] if (APPLE_SILICON or m != "mlx_lm") and (linux or m != "vllm")
    ]
    if not modules:  # a platform marker installs nothing here
        why = "vLLM is a Linux backend" if extra == "vllm" else "MLX is macOS on Apple Silicon only"
        raise Uncheckable(why)
    # pystray reads the display as it is imported: on a Linux with none (a
    # container) whether it is installed is the question here, not a desktop.
    headless = linux and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    code = "; ".join(
        f"import importlib.util as u; assert u.find_spec({m!r})"
        if headless and m == "pystray"
        else f"import {m}"
        for m in modules
    )
    if extra == "coqui":
        # As HFL imports it: coqui-tts needs the helper HFL's coqui engine
        # supplies under transformers 5 (a bare `import TTS` fails there).
        code = "from hfl.engine.coqui_engine import _transformers5_compat as c; c(); " + code
    imported = subprocess.run(
        [str(venv / "bin" / "python"), "-c", code], capture_output=True, text=True, timeout=300
    )
    expect(imported.returncode == 0, f"installed, but: {imported.stderr.strip()[-300:]}")
    version = subprocess.run(
        [str(venv / "bin" / "hfl"), "version"], capture_output=True, text=True, timeout=120
    )
    expect(version.returncode == 0, version.stderr[-200:])
    return f"installs; {', '.join(modules)} import; hfl runs"


for name, cid in EXTRA_ID.items():
    check(cid, f"[{name}]")(lambda a, name=name: _extra(a, name))

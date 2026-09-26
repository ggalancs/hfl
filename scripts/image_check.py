#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A Docker image of HFL, checked before anyone pulls it.

    python scripts/image_check.py ghcr.io/ggalancs/hfl:dev

The 0.21.0 image was published unable to start (its default command
stopped at the network-exposure prompt) and unable to load llama.cpp
(``libgomp.so.1`` missing). This runs what a user would:

1. the image as is, its default command, port published on loopback: it
   must answer ``/healthz`` from the host;
2. with ``HFL_API_KEY`` in its environment: 401 without the key, 200 with it;
3. llama.cpp importable in it;
4. ``scripts/platform_check.py`` inside it — models pulled, served and
   answered over every API.

Every container it starts is removed, whatever happens. Exit 0 only when
all four passed.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

CHECK = Path(__file__).resolve().with_name("platform_check.py")


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker(*args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )  # fmt: skip


class Checks:
    def __init__(self) -> None:
        self.failed = False

    def check(self, label: str, ok: bool, detail: object = "") -> bool:
        self.failed |= not ok
        print(f"{'OK ' if ok else 'BAD'} {label}: {str(detail)[:300]}", flush=True)
        return ok


def _serving(image: str, name: str, env: dict[str, str]) -> tuple[int, str]:
    """Start ``image`` with its default command; its port and whether
    ``/healthz`` answered within a minute (else the container's log)."""
    port = _port()
    flags = [f"-e{key}={value}" for key, value in env.items()]
    _docker("run", "-d", "--name", name, "-p", f"127.0.0.1:{port}:11434", *flags, image)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2).status_code == 200:
                return port, ""
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return port, _docker("logs", name).stdout[-400:] + _docker("logs", name).stderr[-400:]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    image = sys.argv[1]
    run = Checks()
    names = ["hfl-image-check-default", "hfl-image-check-key"]
    try:
        port, log = _serving(image, names[0], {})
        run.check("starts with its default command", not log, log or f"port {port}")

        port, log = _serving(image, names[1], {"HFL_API_KEY": "image-check"})
        if run.check("starts with HFL_API_KEY", not log, log or f"port {port}"):
            url = f"http://127.0.0.1:{port}/api/tags"
            without = httpx.get(url, timeout=10).status_code
            with_key = httpx.get(
                url, headers={"Authorization": "Bearer image-check"}, timeout=10
            ).status_code
            run.check("requires the key", (without, with_key) == (401, 200), (without, with_key))

        imported = _docker(
            "run", "--rm", "--entrypoint", "/opt/venv/bin/python", image,
            "-c", "import llama_cpp; print(llama_cpp.__version__)",
        )  # fmt: skip
        run.check(
            "llama.cpp loads",
            imported.returncode == 0,
            imported.stdout.strip() or imported.stderr.strip()[-300:],
        )

        checked = _docker(
            "run", "--rm", "--name", "hfl-image-check-platform",
            "-v", f"{CHECK}:/check/platform_check.py:ro",
            "--entrypoint", "/opt/venv/bin/python", image, "-u", "/check/platform_check.py",
            timeout=1500,
        )  # fmt: skip
        lines = [line for line in checked.stdout.splitlines() if line[:3] in ("OK ", "BAD")]
        run.check(
            "platform_check.py inside",
            checked.returncode == 0,
            "; ".join(line.split(":")[0] for line in lines if line.startswith("BAD"))
            or f"{len(lines)} checks passed",
        )
    except subprocess.TimeoutExpired as exc:
        run.check("docker", False, f"timed out: {exc.cmd}")
    finally:
        for name in [*names, "hfl-image-check-platform"]:
            _docker("rm", "-f", name)
    print("ALL OK" if not run.failed else "FAILURES", flush=True)
    return 1 if run.failed else 0


if __name__ == "__main__":
    sys.exit(main())

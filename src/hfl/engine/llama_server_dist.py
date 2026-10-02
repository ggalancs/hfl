# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""llama.cpp's own ``llama-server``, without installing llama.cpp.

HFL serves several requests at once on a GGUF model only through
``llama-server``, which users had to install themselves (Homebrew, winget, a
build); a ``pip install``, the Docker image and the installers answered one
request at a time. This module fetches the official build of a pinned
release from llama.cpp's GitHub releases — the archive's sha256 checked
against the one written here — and keeps ``llama-server`` (and
``llama-quantize``, which GGUF conversion uses) with their libraries in
``~/.hfl/bin/llama.cpp-<release>/``. The installers bundle the same files.

Nothing is downloaded unless asked (``hfl install llama-server``). Lookup
order (``hfl.engine.llama_server.binary``): ``HFL_LLAMA_SERVER_BIN``, the
PATH (an install of the user's own), the copy bundled with an executable,
the one installed here.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

RELEASE = "b10964"
_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{release}/{name}"

# (system, machine, variant) -> the archives to unpack together: name,
# size and sha256 as llama.cpp's release publishes them (GitHub's digest).
_A = tuple[str, int, str]
ASSETS: dict[tuple[str, str, str], list[_A]] = {
    ("darwin", "arm64", "metal"): [
        ("llama-b10964-bin-macos-arm64.tar.gz", 11149739,
         "033c845c1df9bf945ff37bb193238b40910b2244be3e1e637b2ceb5878f1a6f5"),
    ],
    ("darwin", "x64", "cpu"): [
        ("llama-b10964-bin-macos-x64.tar.gz", 11199948,
         "03430a394d0a169a5e6d8f01c09f48cf58eb026af6fc95940a4a528e2e50cf38"),
    ],
    ("linux", "x64", "cpu"): [
        ("llama-b10964-bin-ubuntu-x64.tar.gz", 16825086,
         "9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443"),
    ],
    ("linux", "x64", "vulkan"): [
        ("llama-b10964-bin-ubuntu-vulkan-x64.tar.gz", 30166472,
         "55d1e58e14c11eedea090bf088fdeefbfe7b4b09ee03bf6dba9834651769afcf"),
    ],
    ("linux", "arm64", "cpu"): [
        ("llama-b10964-bin-ubuntu-arm64.tar.gz", 13451337,
         "5f0e9c95d970892e43380f82ebcab960edfd20a1cd0f7abffa13b29fdb924949"),
    ],
    ("linux", "arm64", "vulkan"): [
        ("llama-b10964-bin-ubuntu-vulkan-arm64.tar.gz", 24215545,
         "f7864baa0edf5a059fb42c5efb5aceb96075aa1f41e6c3142b71ca69286cb0bb"),
    ],
    ("windows", "x64", "cpu"): [
        ("llama-b10964-bin-win-cpu-x64.zip", 18427629,
         "917f39c076402c421224824607397af20f53625a60defc20e8dd22446bf4c5d7"),
    ],
    # The Vulkan build carries the CPU backends too: any GPU, or none.
    ("windows", "x64", "vulkan"): [
        ("llama-b10964-bin-win-vulkan-x64.zip", 31674542,
         "1ee3ad952f4ba71f438bd6d7bebef19e1c7af04adcaa35d08b4ddabb27d4c642"),
    ],
    ("windows", "x64", "cuda"): [
        ("llama-b10964-bin-win-cuda-12.4-x64.zip", 254067651,
         "264f20d7ee3860aecca9ec12418357a9f3e80349a2b186f66c63859ded1a9593"),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", 391443627,
         "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
    ],
    ("windows", "arm64", "cpu"): [
        ("llama-b10964-bin-win-cpu-arm64.zip", 11996956,
         "4b6a004b076eea47c318bea35cf1db2ff2bf037738b04645646ae8d7c3159478"),
    ],
}  # fmt: skip

# Per platform, what ``hfl install llama-server`` takes when not told.
DEFAULT_VARIANT = {"darwin": {"arm64": "metal"}, "windows": {"x64": "vulkan"}}

# The programs kept (the rest of llama.cpp's tools are left out).
PROGRAMS = ("llama-server", "llama-quantize")


class InstallError(RuntimeError):
    """The download, the check or the unpacking failed; nothing changed."""


def platform_key() -> tuple[str, str]:
    """``(system, machine)`` as ``ASSETS`` names them."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64"}.get(machine, machine)
    return system, machine


def variants(key: tuple[str, str] | None = None) -> list[str]:
    """The builds llama.cpp publishes for this platform."""
    system, machine = key or platform_key()
    return [v for (s, m, v) in ASSETS if (s, m) == (system, machine)]


def default_variant(key: tuple[str, str] | None = None) -> str | None:
    system, machine = key or platform_key()
    choice = DEFAULT_VARIANT.get(system, {}).get(machine)
    available = variants((system, machine))
    if choice in available:
        return choice
    return "cpu" if "cpu" in available else (available[0] if available else None)


def exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def install_dir() -> Path:
    from hfl.config import config

    return Path(config.home_dir) / "bin" / f"llama.cpp-{RELEASE}"


def _runnable(path: Path) -> str | None:
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def managed_binary(name: str = "llama-server") -> str | None:
    """The copy ``hfl install llama-server`` put in place, if any."""
    return _runnable(install_dir() / exe(name))


def bundled_binary(name: str = "llama-server") -> str | None:
    """The copy an executable (PyInstaller, the DMG, the MSI) carries."""
    if not getattr(sys, "frozen", False):
        return None
    roots = [Path(getattr(sys, "_MEIPASS", "")), Path(sys.executable).parent]
    for root in roots:
        found = _runnable(root / "llama.cpp" / exe(name))
        if found:
            return found
    return None


def _kept(member: str) -> str | None:
    """The file name to keep from an archive member, or None. Only a name:
    no directory from the archive reaches the disk (no path traversal)."""
    name = member.replace("\\", "/").rsplit("/", 1)[-1]
    if not name or name.startswith("."):
        return None
    stem = name[:-4] if name.lower().endswith(".exe") else name
    if stem in PROGRAMS:
        return name
    lowered = name.lower()
    if lowered.startswith("license"):
        return name
    if lowered.endswith((".dll", ".dylib")) or ".so" in lowered:
        # Each tool's own code (libllama-cli-impl.so…): only ours.
        if "-impl" in lowered and not any(program in lowered for program in PROGRAMS):
            return None
        return name
    return None


def _download(name: str, sha256: str, dest: Path, progress: Callable[[int], None] | None) -> Path:
    import httpx

    url = _URL.format(release=RELEASE, name=name)
    path = dest / name
    digest = hashlib.sha256()
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=httpx.Timeout(60.0, connect=30.0)
        ) as response:
            response.raise_for_status()
            with open(path, "wb") as out:
                for chunk in response.iter_bytes(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
                    if progress is not None:
                        progress(len(chunk))
    except httpx.HTTPError as exc:
        raise InstallError(f"could not download {url}: {exc}") from exc
    if digest.hexdigest() != sha256:
        raise InstallError(f"{name}: sha256 {digest.hexdigest()} is not the published {sha256}")
    return path


def _unpack(archive: Path, into: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                kept = _kept(info.filename)
                if kept and not info.is_dir():
                    (into / kept).write_bytes(zf.read(info))
        return
    with tarfile.open(archive) as tf:
        links = []
        for member in tf.getmembers():
            kept = _kept(member.name)
            if not kept:
                continue
            if member.issym():  # libfoo.dylib -> libfoo.0.dylib: keep it a link, by name
                target = member.linkname.replace("\\", "/").rsplit("/", 1)[-1]
                links.append((kept, target))
            elif member.isfile():
                source = tf.extractfile(member)
                if source is not None:
                    (into / kept).write_bytes(source.read())
        for name, target in links:
            link = into / name
            if not link.exists():
                link.symlink_to(target)


def check(server: Path) -> str:
    """``llama-server --version`` of an unpacked build; the build it names
    must be the pinned release. Raises with what the system said otherwise
    (an older glibc, a missing library)."""
    try:
        done = subprocess.run(
            [str(server), "--version"], capture_output=True, text=True, timeout=60
        )
    except OSError as exc:
        raise InstallError(f"llama-server does not start here: {exc}") from exc
    said = (done.stdout + done.stderr).strip()
    if f"build {RELEASE.lstrip('b')}" not in said:
        tail = said.splitlines()[-3:] if said else ["(no output)"]
        raise InstallError("llama-server does not run here: " + " / ".join(tail))
    return said


def install(
    variant: str | None = None,
    *,
    progress: Callable[[int], None] | None = None,
    target: Path | None = None,
) -> Path:
    """Download, check and put in place this platform's build; the path of
    its ``llama-server``. All or nothing: a failure leaves the previous
    install (or none) as it was."""
    system, machine = platform_key()
    variant = variant or default_variant((system, machine))
    assets = ASSETS.get((system, machine, variant or ""))
    if not assets:
        offered = ", ".join(variants((system, machine))) or "none"
        raise InstallError(
            f"llama.cpp publishes no {variant or ''} build for {system}/{machine} (builds here: "
            f"{offered})"
        )
    target = target or install_dir()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    downloads = Path(tempfile.mkdtemp(prefix=".downloads-", dir=target.parent))
    try:
        for name, _size, sha256 in assets:
            _unpack(_download(name, sha256, downloads, progress), staging)
        for program in PROGRAMS:
            path = staging / exe(program)
            if path.exists():
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        server = staging / exe("llama-server")
        if not server.exists():
            raise InstallError("the archive holds no llama-server")
        check(server)
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
        return target / exe("llama-server")
    finally:
        shutil.rmtree(downloads, ignore_errors=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def size_of(variant: str | None = None) -> int:
    """Bytes ``install`` downloads for ``variant`` (0 when there is none)."""
    system, machine = platform_key()
    variant = variant or default_variant((system, machine))
    return sum(size for _name, size, _sha in ASSETS.get((system, machine, variant or ""), []))

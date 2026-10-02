# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl install llama-server``: llama.cpp's official build, checked, in place.

GGUF models answer several requests at once only through llama-server,
which users had to install apart; a pip install, the Docker image and the
installers answered one at a time. These pin the install: the published
sha256 is checked, only llama-server, llama-quantize and their libraries
leave the archive (by name: no path traversal), a build that does not run
here changes nothing, and HFL finds what was installed.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hfl.engine import llama_server_dist as dist

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX test programs")

FAKE_SERVER = "#!/bin/sh\necho 'version: 0.4.1 (build {build}, commit x)'\n"


def _archive(tmp_path: Path, build: str = "10964", extra: dict[str, bytes] | None = None) -> Path:
    members = {
        "llama-b10964/llama-server": FAKE_SERVER.format(build=build).encode(),
        "llama-b10964/llama-quantize": b"#!/bin/sh\n",
        "llama-b10964/llama-cli": b"#!/bin/sh\n",  # another tool: left out
        "llama-b10964/libllama.0.dylib": b"lib",
        "llama-b10964/LICENSE": b"MIT",
        **(extra or {}),
    }
    path = tmp_path / "llama.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("llama-b10964/libllama.dylib")
        link.type, link.linkname = tarfile.SYMTYPE, "libllama.0.dylib"
        tf.addfile(link)
    return path


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """``install`` against a local archive registered for this platform."""

    def setup(archive: Path, sha: str | None = None):
        digest = sha or hashlib.sha256(archive.read_bytes()).hexdigest()
        key = (*dist.platform_key(), "test")
        monkeypatch.setitem(dist.ASSETS, key, [(archive.name, archive.stat().st_size, digest)])

        def fetch(name, sha256, dest, progress):
            data = archive.read_bytes()
            if hashlib.sha256(data).hexdigest() != sha256:
                raise dist.InstallError(f"{name}: sha256 mismatch")
            (dest / name).write_bytes(data)
            return dest / name

        monkeypatch.setattr(dist, "_download", fetch)
        return tmp_path / "home" / "llama.cpp-b10964"

    return setup


def test_only_the_server_quantizer_libraries_and_license_leave_the_archive(offline, tmp_path):
    target = offline(_archive(tmp_path))
    server = dist.install("test", target=target)
    kept = sorted(p.name for p in target.iterdir())
    assert kept == sorted(
        ["LICENSE", "libllama.0.dylib", "libllama.dylib", "llama-quantize", "llama-server"]
    )
    assert os.access(server, os.X_OK) and (target / "libllama.dylib").is_symlink()


def test_no_path_from_the_archive_reaches_the_disk(offline, tmp_path):
    evil = {"../../escaped.so": b"x", "/abs/also.dylib": b"x"}
    target = offline(_archive(tmp_path, extra=evil))
    dist.install("test", target=target)
    assert (target / "escaped.so").exists() and (target / "also.dylib").exists()
    assert not (tmp_path / "escaped.so").exists() and not Path("/abs/also.dylib").exists()


def test_a_wrong_sha256_installs_nothing(offline, tmp_path):
    target = offline(_archive(tmp_path), sha="0" * 64)
    with pytest.raises(dist.InstallError, match="sha256"):
        dist.install("test", target=target)
    assert not target.exists() and not any(target.parent.iterdir())


def test_a_build_that_does_not_run_here_leaves_the_previous_install(offline, tmp_path):
    target = offline(_archive(tmp_path))
    dist.install("test", target=target)
    before = (target / "llama-server").read_text()
    offline(_archive(tmp_path, build="999"))  # runs, but is not the pinned release
    with pytest.raises(dist.InstallError, match="does not run here"):
        dist.install("test", target=target)
    assert (target / "llama-server").read_text() == before
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]  # no staging left


def test_binary_lookup_order(monkeypatch, tmp_path):
    from hfl.engine import llama_server

    monkeypatch.delenv("HFL_LLAMA_SERVER_BIN", raising=False)
    monkeypatch.setattr(dist, "bundled_binary", lambda name="llama-server": "/bundle/llama-server")
    monkeypatch.setattr(dist, "managed_binary", lambda name="llama-server": "/home/llama-server")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llama-server")
    assert llama_server.binary() == "/usr/bin/llama-server"  # the user's own first
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert llama_server.binary() == "/bundle/llama-server"
    monkeypatch.setattr(dist, "bundled_binary", lambda name="llama-server": None)
    assert llama_server.binary() == "/home/llama-server"
    explicit = tmp_path / "mine"
    explicit.write_text("")
    monkeypatch.setenv("HFL_LLAMA_SERVER_BIN", str(explicit))
    assert llama_server.binary() == str(explicit)


def test_every_platform_has_a_default_build():
    for system, machine in {(s, m) for (s, m, _v) in dist.ASSETS}:
        assert dist.default_variant((system, machine)) in dist.variants((system, machine))
    assert dist.default_variant(("windows", "x64")) == "vulkan"
    assert dist.default_variant(("linux", "x64")) == "cpu"


def test_the_pins_name_the_pinned_release():
    for (_s, _m, _v), assets in dist.ASSETS.items():
        for name, size, sha in assets:
            assert (dist.RELEASE in name or name.startswith("cudart-")) and size > 0
            assert len(sha) == 64 and int(sha, 16) >= 0


class TestTheCommand:
    def _run(self, monkeypatch, args, *, own=None, managed=None, tty=False):
        from hfl.cli.main import app

        installed: list[str] = []
        monkeypatch.setattr("shutil.which", lambda name: own)
        monkeypatch.setattr(dist, "bundled_binary", lambda name="llama-server": None)
        monkeypatch.setattr(dist, "managed_binary", lambda name="llama-server": managed)
        monkeypatch.setattr(
            dist, "install", lambda variant, progress=None: installed.append(variant) or Path("/x")
        )
        monkeypatch.setattr("hfl.cli.commands.install.stdin_is_terminal", lambda: tty)
        result = CliRunner().invoke(app, ["install", "llama-server", *args])
        return result, installed

    def test_without_a_terminal_it_needs_yes(self, monkeypatch):
        result, installed = self._run(monkeypatch, [])
        assert result.exit_code == 1 and installed == [] and "--yes" in result.output

    def test_yes_installs_the_default_build(self, monkeypatch):
        result, installed = self._run(monkeypatch, ["--yes"])
        assert result.exit_code == 0 and installed == [dist.default_variant()]

    def test_an_existing_llama_server_is_kept(self, monkeypatch):
        result, installed = self._run(monkeypatch, ["--yes"], own="/opt/homebrew/bin/llama-server")
        assert result.exit_code == 0 and installed == []
        result, installed = self._run(monkeypatch, ["--yes", "--force"], own="/opt/x/llama-server")
        assert installed == [dist.default_variant()]

    def test_an_unknown_build_is_refused(self, monkeypatch):
        result, installed = self._run(monkeypatch, ["--yes", "--variant", "quantum"])
        assert result.exit_code == 1 and installed == []


def test_the_converter_quantizes_with_the_installed_tool(monkeypatch, tmp_path):
    from hfl.converter.gguf_converter import GGUFConverter

    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(dist, "bundled_binary", lambda name="llama-server": None)
    monkeypatch.setattr(dist, "managed_binary", lambda name="llama-server": f"/home/bin/{name}")
    converter = GGUFConverter.__new__(GGUFConverter)
    converter.quantize_bin = tmp_path / "missing" / "llama-quantize"
    assert converter._quantizer() == ["/home/bin/llama-quantize"]


def test_the_real_download_checks_the_sha256(monkeypatch, tmp_path):
    """``_download`` itself, its HTTP replaced: a wrong digest is refused."""
    import contextlib

    payload = b"llama.cpp release bytes"

    class _Response:
        def raise_for_status(self):
            pass

        def iter_bytes(self, size):
            yield payload

    @contextlib.contextmanager
    def stream(method, url, **kwargs):
        yield _Response()

    monkeypatch.setattr("httpx.stream", stream)
    good = hashlib.sha256(payload).hexdigest()
    assert dist._download("x.tar.gz", good, tmp_path, None).read_bytes() == payload
    with pytest.raises(dist.InstallError, match="not the published"):
        dist._download("x.tar.gz", "f" * 64, tmp_path, None)

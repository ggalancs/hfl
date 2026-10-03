# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""hfl.utils: the Windows ctypes paths (with a fake kernel32), reading
another process's executable, and a cgroup found through /proc/self/cgroup."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hfl.utils import cgroup, self_exec, terminal

# -- a fake kernel32 -------------------------------------------------------------


class _Func:
    """A ctypes function stand-in: records calls, has a settable restype."""

    def __init__(self, impl):
        self.impl = impl
        self.calls: list[tuple] = []
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.impl(*args)


class _Kernel32:
    def __init__(self, *, handle=1234, image="/opt/hfl/hfl.exe", query_ok=True, console=True):
        def query(handle_ptr, flags, buffer, size_ref):
            buffer.value = image
            return 1 if query_ok else 0

        def get_console_mode(handle_ptr, mode_ref):
            return 1 if console else 0

        self.OpenProcess = _Func(lambda access, inherit, pid: handle)
        self.QueryFullProcessImageNameW = _Func(query)
        self.CloseHandle = _Func(lambda h: 1)
        self.WaitForSingleObject = _Func(lambda h, ms: 0)
        self.GetConsoleMode = _Func(get_console_mode)


@pytest.fixture
def kernel32(monkeypatch):
    """Installs a fake ``ctypes.WinDLL``; returns a setter for its kernel32."""
    box: dict[str, _Kernel32] = {"k": _Kernel32()}
    opened: list[tuple] = []

    def win_dll(name, use_last_error=False):
        opened.append((name, use_last_error))
        return box["k"]

    monkeypatch.setattr(ctypes, "WinDLL", win_dll, raising=False)

    def use(k: _Kernel32) -> _Kernel32:
        box["k"] = k
        return k

    use.opened = opened  # type: ignore[attr-defined]
    return use


def test_kernel32_is_loaded_with_last_error_and_a_pointer_restype(kernel32) -> None:
    k = kernel32(_Kernel32())
    assert self_exec._kernel32() is k
    assert kernel32.opened == [("kernel32", True)]
    assert k.OpenProcess.restype is ctypes.c_void_p


def test_windows_image_of_reads_the_full_image_name_and_closes(kernel32) -> None:
    k = kernel32(_Kernel32(image="/opt/hfl/hfl.exe"))
    assert self_exec._windows_image_of(42) == os.path.realpath("/opt/hfl/hfl.exe")
    access, inherit, pid = k.OpenProcess.calls[0]
    assert (access, inherit, pid) == (0x1000, False, 42)  # query-limited-information
    assert len(k.CloseHandle.calls) == 1  # the handle never leaks


def test_windows_image_of_a_failed_query_is_none_and_still_closes(kernel32) -> None:
    k = kernel32(_Kernel32(query_ok=False))
    assert self_exec._windows_image_of(42) is None
    assert len(k.CloseHandle.calls) == 1


def test_windows_image_of_a_process_it_cannot_open(kernel32) -> None:
    k = kernel32(_Kernel32(handle=0))
    assert self_exec._windows_image_of(42) is None
    assert k.QueryFullProcessImageNameW.calls == [] and k.CloseHandle.calls == []


def test_windows_waiter_waits_on_a_handle_opened_up_front(kernel32) -> None:
    k = kernel32(_Kernel32(handle=77))
    wait = self_exec._windows_waiter(9)
    assert wait is not None
    assert k.OpenProcess.calls == [(0x00100000, False, 9)]  # SYNCHRONIZE, opened now
    assert k.WaitForSingleObject.calls == []  # not until called
    wait()
    ((handle, timeout),) = k.WaitForSingleObject.calls
    assert handle.value == 77 and timeout == 0xFFFFFFFF


def test_windows_waiter_for_a_process_already_gone(kernel32) -> None:
    kernel32(_Kernel32(handle=0))
    assert self_exec._windows_waiter(9) is None


def test_executable_of_on_windows_asks_kernel32(monkeypatch) -> None:
    monkeypatch.setattr(self_exec, "_windows_image_of", lambda pid: f"img-{pid}")
    monkeypatch.setattr(os, "name", "nt")
    try:
        assert self_exec._executable_of(5) == "img-5"
    finally:
        monkeypatch.setattr(os, "name", "posix")


def test_executable_of_on_linux_reads_proc(monkeypatch, tmp_path) -> None:
    target = tmp_path / "python-real"
    target.write_text("")
    read: list[str] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "readlink", lambda p: read.append(p) or str(target))
    assert self_exec._executable_of(321) == os.path.realpath(target)
    assert read == ["/proc/321/exe"]


def test_executable_of_on_linux_unreadable_is_none(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def denied(path):
        raise PermissionError(path)

    monkeypatch.setattr(os, "readlink", denied)
    assert self_exec._executable_of(321) is None


def test_executable_of_elsewhere_asks_ps(monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(argv, **kw):
        calls.append(argv)
        return types.SimpleNamespace(stdout="/usr/bin/true\n")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", run)
    assert self_exec._executable_of(77) == os.path.realpath("/usr/bin/true")
    assert calls == [["ps", "-o", "comm=", "-p", "77"]]


def test_executable_of_a_pid_ps_does_not_know(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: types.SimpleNamespace(stdout=""))
    assert self_exec._executable_of(77) is None


def test_executable_of_when_ps_times_out(monkeypatch) -> None:
    def run(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 5)

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", run)
    assert self_exec._executable_of(77) is None


def test_executable_of_this_very_process_for_real() -> None:
    """No fake: the running interpreter's own PID names a real file."""
    found = self_exec._executable_of(os.getpid())
    assert found is not None and os.path.exists(found)


def test_a_launcher_that_is_init_is_never_watched(monkeypatch) -> None:
    monkeypatch.setattr(self_exec, "is_frozen", lambda: True)
    monkeypatch.setattr(os, "getppid", lambda: 1)
    looked: list[int] = []
    monkeypatch.setattr(self_exec, "_executable_of", lambda pid: looked.append(pid))
    assert self_exec.watch_onefile_launcher() is False
    assert looked == []  # an orphan is not even inspected


# -- terminal: a real Windows console -----------------------------------------------


@pytest.fixture
def msvcrt(monkeypatch):
    fake = types.ModuleType("msvcrt")
    fake.get_osfhandle = lambda fd: 4000 + fd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    return fake


def test_windows_console_true_for_a_console(kernel32, msvcrt) -> None:
    k = kernel32(_Kernel32(console=True))
    assert terminal._windows_console(0) is True
    ((handle, _mode),) = k.GetConsoleMode.calls
    assert handle.value == 4000


def test_windows_console_false_for_nul(kernel32, msvcrt) -> None:
    kernel32(_Kernel32(console=False))
    assert terminal._windows_console(0) is False


def test_windows_console_false_for_a_closed_descriptor(kernel32, msvcrt) -> None:
    def bad(fd):
        raise OSError(9, "Bad file descriptor")

    msvcrt.get_osfhandle = bad
    k = kernel32(_Kernel32())
    assert terminal._windows_console(7) is False
    assert k.GetConsoleMode.calls == []


def test_stdin_on_windows_goes_through_the_console_check(monkeypatch, kernel32, msvcrt) -> None:
    kernel32(_Kernel32(console=False))
    monkeypatch.setattr(sys, "stdin", MagicMock(isatty=lambda: True, fileno=lambda: 0))
    monkeypatch.setattr(os, "name", "nt")
    try:
        answer = terminal.stdin_is_terminal()
    finally:
        monkeypatch.setattr(os, "name", "posix")
    assert answer is False  # a tty that is NUL: nobody to answer


def test_stdin_whose_fileno_fails_is_not_a_terminal(monkeypatch) -> None:
    def closed():
        raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "stdin", MagicMock(isatty=closed))
    assert terminal.stdin_is_terminal() is False


# -- cgroup: this process's own directory ---------------------------------------------


@pytest.fixture
def proc_cgroup(monkeypatch):
    """Redirects reads of /proc/self/cgroup to a given text (None: unreadable)."""
    content: dict[str, str | None] = {"text": None}
    real = Path.read_text

    def read_text(self, *a, **kw):
        if str(self) == "/proc/self/cgroup":
            if content["text"] is None:
                raise FileNotFoundError(str(self))
            return content["text"]
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(cgroup, "_on_linux", lambda: True)
    return content


def test_own_dirs_come_from_proc_self_cgroup(tmp_path, monkeypatch, proc_cgroup) -> None:
    monkeypatch.setattr(cgroup, "ROOT", tmp_path)
    proc_cgroup["text"] = (
        "0::/system.slice/hfl.service\n4:memory:/docker/abc\n5:cpu:/x\n1:name=systemd:/\n"
    )
    assert cgroup._own_dirs() == [
        tmp_path / "system.slice/hfl.service",
        tmp_path / "memory" / "system.slice/hfl.service",
        tmp_path / "docker/abc",
        tmp_path / "memory" / "docker/abc",
        tmp_path,
        tmp_path / "memory",
    ]


def test_own_dirs_without_proc_is_the_mount_root(tmp_path, monkeypatch, proc_cgroup) -> None:
    monkeypatch.setattr(cgroup, "ROOT", tmp_path)
    proc_cgroup["text"] = None
    assert cgroup._own_dirs() == [tmp_path, tmp_path / "memory"]


def test_the_most_specific_cgroup_limit_wins(tmp_path, monkeypatch, proc_cgroup) -> None:
    own = tmp_path / "svc"
    own.mkdir()
    (own / "memory.max").write_text("1000\n")
    (own / "memory.current").write_text("600\n")
    (own / "memory.stat").write_text("inactive_file 100\n")
    (tmp_path / "memory.max").write_text("999999\n")
    monkeypatch.setattr(cgroup, "ROOT", tmp_path)
    proc_cgroup["text"] = "0::/svc\n"
    assert cgroup.limit_bytes() == 1000
    assert cgroup.in_use_bytes() == 500


def test_unparseable_files_are_skipped(tmp_path, monkeypatch, proc_cgroup) -> None:
    (tmp_path / "memory.max").write_text("garbage")
    (tmp_path / "memory.limit_in_bytes").write_text("2048")
    (tmp_path / "memory.current").write_text("300")
    (tmp_path / "memory.stat").write_text("inactive_file notanumber\n")
    monkeypatch.setattr(cgroup, "ROOT", tmp_path)
    assert cgroup.limit_bytes() == 2048  # the v1 name, after a garbled v2 one
    assert cgroup.in_use_bytes() == 300  # an unreadable stat drops nothing


def test_usage_without_a_stat_file_and_cache_above_usage(
    tmp_path, monkeypatch, proc_cgroup
) -> None:
    (tmp_path / "memory.current").write_text("300")
    monkeypatch.setattr(cgroup, "ROOT", tmp_path)
    assert cgroup.in_use_bytes() == 300
    (tmp_path / "memory.stat").write_text("total_inactive_file 900\n")
    assert cgroup.in_use_bytes() == 0  # never negative


def test_no_cgroup_files_at_all(tmp_path, monkeypatch, proc_cgroup) -> None:
    monkeypatch.setattr(cgroup, "ROOT", tmp_path)
    assert cgroup.limit_bytes() is None
    assert cgroup.in_use_bytes() is None


def test_off_linux_there_is_no_cgroup(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert cgroup._on_linux() is False
    assert cgroup.limit_bytes() is None and cgroup.in_use_bytes() is None

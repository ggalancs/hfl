# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A container's memory limit is the machine's memory.

psutil reads the host's: on Modal a 48 GB container saw 339.6 GB, ``hfl
pull`` judged a 235B model would fit split GPU + RAM and downloaded 512 GB,
and the residency planner would admit models past the limit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hfl.utils import cgroup

GIB = 1024**3


@pytest.fixture
def container(tmp_path, monkeypatch):
    """A cgroup v2 mount with a 48 GiB limit, 10 GiB used of which 2 GiB is
    droppable cache, and a host psutil sees as 339.6 GiB."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(str(48 * GIB))
    (root / "memory.current").write_text(str(10 * GIB))
    (root / "memory.stat").write_text(f"anon 1\ninactive_file {2 * GIB}\nactive_file 5\n")
    monkeypatch.setattr(cgroup, "ROOT", root)
    monkeypatch.setattr(cgroup, "_on_linux", lambda: True)
    return root


@pytest.fixture
def host_psutil(monkeypatch):
    """psutil reading a 339.6 GiB host (skipped where psutil is absent)."""
    pytest.importorskip("psutil")
    host = SimpleNamespace(total=int(339.6 * GIB), available=int(300 * GIB))
    monkeypatch.setattr("psutil.virtual_memory", lambda: host)


def test_the_limit_and_what_is_in_use(container) -> None:
    assert cgroup.limit_bytes() == 48 * GIB
    assert cgroup.in_use_bytes() == 8 * GIB


def test_no_limit_is_none(container) -> None:
    (container / "memory.max").write_text("max")
    assert cgroup.limit_bytes() is None


def test_cgroup_v1_unlimited_is_none(tmp_path, monkeypatch) -> None:
    root = tmp_path / "v1"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "memory.limit_in_bytes").write_text(str(9223372036854771712))
    monkeypatch.setattr(cgroup, "ROOT", root)
    monkeypatch.setattr(cgroup, "_on_linux", lambda: True)
    assert cgroup.limit_bytes() is None


def test_the_hardware_profile_sees_the_container(container, host_psutil) -> None:
    from hfl.hub.hw_profile import _system_ram_gb

    assert _system_ram_gb() == 48.0


def test_the_residency_planner_sees_the_container(container, host_psutil) -> None:
    from hfl.engine.residency import current_memory

    view = current_memory()
    assert view is not None and view.total == 48 * GIB and view.in_use == 8 * GIB


def test_off_linux_nothing_is_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cgroup, "_on_linux", lambda: False)
    monkeypatch.setattr(cgroup, "ROOT", tmp_path)  # would hold nothing anyway
    assert cgroup.limit_bytes() is None and cgroup.in_use_bytes() is None

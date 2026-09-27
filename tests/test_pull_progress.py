# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``/api/pull`` reports real download progress: the size of the files it
fetches (from the Hub) and the bytes on disk so far. Every event said
``total: 0, completed: 0`` until the end, so a client (Open WebUI, HFL's own
chat page) could only show an indeterminate bar (plan 0.22 P1-11)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.hub import downloader
from hfl.hub.resolver import ResolvedModel

GGUF = ResolvedModel(
    repo_id="acme/m-GGUF",
    format="gguf",
    filename="m-Q4.gguf",
    parts=["m-Q4-2.gguf"],
    projector="mmproj.gguf",
)
FILES = ["m-Q4.gguf", "m-Q4-2.gguf", "mmproj.gguf", "m-Q8.gguf", "README.md"]


def test_the_files_a_pull_fetches() -> None:
    assert downloader._planned(GGUF, FILES) == ["m-Q4.gguf", "m-Q4-2.gguf", "mmproj.gguf"]
    safetensors = ResolvedModel(repo_id="acme/m", format="safetensors")
    names = ["model.safetensors", "config.json", "README.md", "pytorch_model.bin"]
    assert downloader._planned(safetensors, names) == ["model.safetensors", "config.json"]


@pytest.mark.hub_sizes
def test_their_sizes_come_from_the_hub(monkeypatch) -> None:
    siblings = [SimpleNamespace(rfilename=n, size=10 * (i + 1)) for i, n in enumerate(FILES)]
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(siblings=siblings)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    monkeypatch.setattr(downloader, "ensure_auth", lambda repo: None)
    assert downloader.expected_files(GGUF) == {
        "m-Q4.gguf": 10,
        "m-Q4-2.gguf": 20,
        "mmproj.gguf": 30,
    }
    assert api.model_info.call_args.kwargs["files_metadata"] is True


@pytest.mark.hub_sizes
def test_no_answer_is_an_unknown_total(monkeypatch) -> None:
    api = MagicMock()
    api.model_info.side_effect = OSError("offline")
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    monkeypatch.setattr(downloader, "ensure_auth", lambda repo: None)
    assert downloader.expected_files(GGUF) == {}


def test_bytes_on_disk_count_partial_downloads(temp_config, monkeypatch) -> None:
    monkeypatch.setattr(downloader, "config", temp_config)
    folder = downloader.model_dir_for(GGUF)
    (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (folder / "m-Q4.gguf").write_bytes(b"x" * 10)
    (folder / "m-Q8.gguf").write_bytes(b"x" * 999)  # another quant, not this pull
    (folder / ".cache" / "huggingface" / "download" / "abc.incomplete").write_bytes(b"x" * 7)
    assert downloader.bytes_done(GGUF, {"m-Q4.gguf": 10, "m-Q4-2.gguf": 20}) == 17


@pytest.mark.slow
def test_the_stream_carries_them(temp_config, monkeypatch) -> None:
    """A download slower than the 2 s heartbeat: the heartbeat in between
    reports the partial bytes against the total."""
    from hfl.api.server import app
    from hfl.hub.license_checker import LicenseInfo, LicenseRisk

    monkeypatch.setattr(downloader, "config", temp_config)
    resolved = ResolvedModel(repo_id="acme/m-GGUF", format="gguf", filename="m.gguf")
    folder = downloader.model_dir_for(resolved)

    def slow_pull(r):
        partial = folder / ".cache" / "huggingface" / "download"
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "x.incomplete").write_bytes(b"x" * 40)
        time.sleep(2.6)
        (partial / "x.incomplete").unlink()
        (folder / "m.gguf").write_bytes(b"x" * 100)
        return folder / "m.gguf"

    license_ok = LicenseInfo(
        license_id="apache-2.0",
        license_name="Apache 2.0",
        risk=LicenseRisk.PERMISSIVE,
        restrictions=[],
        url="",
        gated=False,
    )
    with (
        patch("hfl.hub.resolver.resolve", return_value=resolved),
        patch("hfl.hub.downloader.pull_model", side_effect=slow_pull),
        patch("hfl.hub.downloader.expected_files", return_value={"m.gguf": 100}),
        patch("hfl.hub.license_checker.check_model_license", return_value=license_ok),
    ):
        body = (
            TestClient(app, client=("127.0.0.1", 5555))
            .post("/api/pull", json={"model": "acme/m-GGUF", "stream": True})
            .text
        )
    events = [json.loads(line) for line in body.splitlines() if line.strip()]
    progress = [(e["completed"], e["total"]) for e in events if e.get("status") == "downloading"]
    assert (0, 100) == progress[0] and (40, 100) in progress and progress[-1] == (100, 100)
    assert events[-1]["status"] == "success"


def test_a_pulled_model_is_listed_by_the_same_server(temp_config, monkeypatch) -> None:
    """The pull registered through a fresh ModelRegistry: on disk, but the
    server's own registry (what /api/tags reads) did not see it until a
    restart. Found using the chat page's model manager."""
    from hfl.api.server import app
    from hfl.core.container import reset_container
    from hfl.hub.license_checker import LicenseInfo, LicenseRisk

    reset_container()
    monkeypatch.setattr(downloader, "config", temp_config)
    resolved = ResolvedModel(repo_id="acme/tiny-GGUF", format="gguf", filename="tiny.gguf")
    folder = downloader.model_dir_for(resolved)
    folder.mkdir(parents=True)
    (folder / "tiny.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    license_ok = LicenseInfo(
        license_id="apache-2.0",
        license_name="Apache 2.0",
        risk=LicenseRisk.PERMISSIVE,
        restrictions=[],
        url="",
        gated=False,
    )
    client = TestClient(app, client=("127.0.0.1", 5555))
    names = lambda: [m["name"] for m in client.get("/api/tags").json()["models"]]  # noqa: E731
    before = names()  # the server's registry is loaded now, before the pull
    with (
        patch("hfl.hub.resolver.resolve", return_value=resolved),
        patch("hfl.hub.downloader.pull_model", return_value=folder / "tiny.gguf"),
        patch("hfl.hub.license_checker.check_model_license", return_value=license_ok),
    ):
        assert (
            client.post("/api/pull", json={"model": "acme/tiny-GGUF", "stream": False}).status_code
            == 200
        )
    added = set(names()) - set(before)
    assert len(added) == 1 and "tiny" in added.pop()
    reset_container()

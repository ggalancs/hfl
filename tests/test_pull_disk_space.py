# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A pull the disk cannot hold is refused before anything is downloaded.

Nothing checked the free space: on a 512 GB disk a 235B model's 470 GB
download began, filled the disk and failed at the end, taking every other
write on that disk with it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from hfl.hub.pull_service import disk_space
from hfl.hub.resolver import ResolvedModel

GB = 10**9


def _hub(monkeypatch, files: dict[str, int], free: int) -> None:
    siblings = [SimpleNamespace(rfilename=n, size=s) for n, s in files.items()]
    info = SimpleNamespace(siblings=siblings)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda self, *a, **k: info)
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=free))


SAFETENSORS = {
    "model-00001-of-00002.safetensors": 10 * GB,
    "model-00002-of-00002.safetensors": 8 * GB,
    "config.json": 1000,
    "README.md": 5000,  # not downloaded
    "original/consolidated.pth": 18 * GB,  # not downloaded
}


def test_counts_only_what_is_downloaded(temp_config, monkeypatch) -> None:
    _hub(monkeypatch, SAFETENSORS, free=100 * GB)
    space = disk_space(ResolvedModel(repo_id="org/m", format="safetensors"), None)
    assert space is not None and space.download == 18 * GB + 1000 and space.conversion == 0


def test_a_conversion_adds_the_intermediate_and_the_result(temp_config, monkeypatch) -> None:
    _hub(monkeypatch, SAFETENSORS, free=100 * GB)
    resolved = ResolvedModel(repo_id="org/m", format="safetensors")
    q8 = disk_space(resolved, "Q8_0")
    f16 = disk_space(resolved, "F16")
    assert q8.conversion == 18 * GB + int(18 * GB * 8.5 / 16)
    assert f16.conversion == 18 * GB  # the intermediate is the result


def test_a_gguf_counts_its_file_and_parts(temp_config, monkeypatch) -> None:
    _hub(monkeypatch, {"m-q4.gguf": 4 * GB, "m-q8.gguf": 8 * GB, "mmproj.gguf": GB}, free=GB)
    resolved = ResolvedModel(
        repo_id="org/m", format="gguf", filename="m-q4.gguf", projector="mmproj.gguf"
    )
    space = disk_space(resolved, None)
    assert space.download == 5 * GB and not space.fits


def test_no_sizes_from_the_hub_goes_on_as_before(temp_config, monkeypatch) -> None:
    def offline(self, *a, **k):
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", offline)
    assert disk_space(ResolvedModel(repo_id="org/m", format="safetensors"), None) is None


def test_hfl_pull_refuses_before_downloading(temp_config, monkeypatch) -> None:
    from hfl.cli.main import app

    _hub(monkeypatch, SAFETENSORS, free=20 * GB)  # holds the download, not the Q8_0 conversion
    monkeypatch.setattr(
        "hfl.hub.params.estimate_params",
        lambda repo, api=None: SimpleNamespace(total_b=9, active_b=None),
    )
    monkeypatch.setattr("hfl.engine.selector._mlx_preferred", lambda: False)
    monkeypatch.setattr("hfl.cli.main.stdin_is_terminal", lambda: False)
    resolved = ResolvedModel(
        repo_id="org/m", format="safetensors", quantization="Q4_K_M", pipeline_tag="text-generation"
    )
    downloads: list[str] = []
    with (
        patch("hfl.hub.resolver.resolve", return_value=resolved),
        patch("hfl.hub.downloader.pull_model", side_effect=lambda r, *a, **k: downloads.append(r)),
    ):
        result = CliRunner().invoke(app, ["pull", "org/m", "--skip-license", "-q", "Q8_0"])
    assert result.exit_code == 1 and downloads == []
    assert "Not enough disk space" in result.output and "to convert" in result.output


def test_api_pull_refuses_before_downloading(monkeypatch) -> None:
    import asyncio
    import json

    from hfl.api import routes_pull
    from hfl.hub.pull_service import DiskSpace

    monkeypatch.setattr(
        "hfl.hub.pull_service.disk_space", lambda resolved, convert_to: DiskSpace(30 * GB, 0, GB)
    )
    downloads: list[object] = []
    monkeypatch.setattr("hfl.hub.downloader.pull_model", lambda r: downloads.append(r))
    state = SimpleNamespace(local_path=None)

    async def run() -> list[str]:
        resolved = ResolvedModel(repo_id="org/m", format="safetensors")
        return [e async for e in routes_pull._download_stage(resolved, "d", dict, state)]

    events = asyncio.run(run())
    body = json.loads(events[0].strip().removeprefix("data:").strip()) if events else {}
    assert downloads == [] and state.local_path is None
    assert body.get("code") == "no_disk_space" and "30.0 GB" in body.get("error", "")


def test_a_sharded_model_comes_with_its_index_and_template() -> None:
    """The download patterns left out model.safetensors.index.json: every
    sharded safetensors model downloaded but could not load (an L4,
    Yi-1.5-9B-Chat: "no file named model.safetensors")."""
    from fnmatch import fnmatch

    from hfl.hub.downloader import _SAFETENSORS_FILES

    def fetched(name: str) -> bool:
        return any(fnmatch(name, pattern) for pattern in _SAFETENSORS_FILES)

    for needed in (
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "chat_template.json",
    ):
        assert fetched(needed), needed
    for not_needed in ("README.md", "original/consolidated.00.pth", "pytorch_model.bin"):
        assert not fetched(not_needed), not_needed

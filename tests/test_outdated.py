# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl outdated``: newer versions on the Hub, told apart from "cannot tell".

Checked for real (2026-09-26) with nomic-embed-text-v1.5-GGUF recorded at
two of its old commits: a README edit since → up to date (the repo changed,
not the file); the first quants → newer on the Hub, and the ``hfl pull`` it
printed brought the entry up to date with its alias kept. With no network
(unresolvable host, and a blackholed one) every model said "could not
check", exit 1, the blackhole bounded to one timeout for all.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hfl.hub.outdated import Check, check, check_all, local_files, update_command
from hfl.models.manifest import ModelManifest

OLD, NEW = "a" * 40, "b" * 40
REPO = "org/model-GGUF"


class FakeHub:
    """``model_info`` at a revision: the commit and each file's blob id."""

    def __init__(self, commits: dict[str, dict[str, str]], head: str = NEW, error=None):
        self.commits, self.head, self.error, self.calls = commits, head, error, 0

    def model_info(self, repo_id, revision, files_metadata):
        self.calls += 1
        if self.error is not None:
            raise self.error
        sha = self.head if revision == "main" else revision
        files = self.commits[sha]
        siblings = [SimpleNamespace(rfilename=f, blob_id=b) for f, b in files.items()]
        return SimpleNamespace(sha=sha, siblings=siblings)


def _pulled(models: Path, files: list[str], **fields) -> ModelManifest:
    root = models / REPO.replace("/", "--")
    for name in files:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"GGUF")
    (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
    (root / ".cache" / "huggingface" / "x.lock").write_text("")
    defaults = {
        "name": "m",
        "repo_id": REPO,
        "local_path": str(root / files[0]),
        "format": "gguf",
        "quantization": "Q4_K_M",
        "commit_sha": OLD,
        "revision": "main",
    }
    return ModelManifest(**{**defaults, **fields})


class TestLocalFiles:
    def test_a_split_gguf_its_parts_and_projector(self, tmp_path):
        parts = ["q/m-q4-00001-of-00002.gguf", "q/m-q4-00002-of-00002.gguf"]
        manifest = _pulled(tmp_path, [*parts, "q/mmproj-m-f16.gguf", "q/other-q8.gguf"])
        assert local_files(manifest, tmp_path) == [*parts, "q/mmproj-m-f16.gguf"]

    def test_a_folder_every_file_but_the_hidden_ones(self, tmp_path):
        manifest = _pulled(tmp_path, ["config.json", "model.safetensors"])
        manifest.local_path = str(tmp_path / REPO.replace("/", "--"))
        assert local_files(manifest, tmp_path) == ["config.json", "model.safetensors"]

    def test_converted_here_nothing_to_compare(self, tmp_path):
        manifest = _pulled(tmp_path, ["m.gguf"], local_path=str(tmp_path / "converted.gguf"))
        assert local_files(manifest, tmp_path) == []


class TestCheck:
    def test_the_same_commit_is_up_to_date(self, tmp_path):
        manifest = _pulled(tmp_path, ["m.gguf"])
        assert check(manifest, FakeHub({OLD: {}}, head=OLD), tmp_path).status == "current"

    def test_the_repo_moved_but_not_its_files(self, tmp_path):
        manifest = _pulled(tmp_path, ["m.gguf"])
        hub = FakeHub(
            {OLD: {"m.gguf": "1", "README.md": "x"}, NEW: {"m.gguf": "1", "README.md": "y"}}
        )
        result = check(manifest, hub, tmp_path)
        assert (result.status, result.command) == ("files_current", None)

    def test_its_file_changed(self, tmp_path):
        manifest = _pulled(tmp_path, ["m.gguf"])
        hub = FakeHub({OLD: {"m.gguf": "1"}, NEW: {"m.gguf": "2"}})
        result = check(manifest, hub, tmp_path)
        assert (result.status, result.changed) == ("update", ["m.gguf"])
        assert result.command == f"hfl pull {REPO} -q Q4_K_M"

    def test_its_file_is_gone(self, tmp_path):
        manifest = _pulled(tmp_path, ["m.gguf"])
        hub = FakeHub({OLD: {"m.gguf": "1"}, NEW: {"renamed.gguf": "1"}})
        result = check(manifest, hub, tmp_path)
        assert (result.status, result.changed, result.command) == ("gone", ["m.gguf"], None)

    def test_converted_here_the_repo_moved(self, tmp_path):
        manifest = _pulled(tmp_path, ["m.gguf"], local_path=str(tmp_path / "converted.gguf"))
        result = check(manifest, FakeHub({OLD: {}, NEW: {}}), tmp_path)
        assert result.status == "update_repo" and result.command

    @pytest.mark.parametrize(
        ("fields", "status"),
        [
            ({"repo_id": "local/m"}, "local"),
            ({"revision": OLD}, "pinned"),
            ({"commit_sha": None}, "unrecorded"),  # no command: it could be tens of GB
        ],
    )
    def test_what_needs_no_hub(self, tmp_path, fields, status):
        hub = FakeHub({}, error=AssertionError("the Hub was asked"))
        result = check(_pulled(tmp_path, ["m.gguf"], **fields), hub, tmp_path)
        assert (result.status, result.command, hub.calls) == (status, None, 0)

    @pytest.mark.parametrize(
        ("error", "why"),
        [
            ("GatedRepoError", "gated"),
            ("RepositoryNotFoundError", "not_found"),
            ("RevisionNotFoundError", "no_revision"),
            ("ConnectError", "unreachable"),
        ],
    )
    def test_cannot_tell_is_not_up_to_date(self, tmp_path, error, why):
        import httpx
        import huggingface_hub.errors as hub_errors

        if error == "ConnectError":
            exc: Exception = httpx.ConnectError("down")
        else:
            response = httpx.Response(404, request=httpx.Request("GET", "https://hf.co"))
            exc = getattr(hub_errors, error)("no", response=response)
        result = check(_pulled(tmp_path, ["m.gguf"]), FakeHub({}, error=exc), tmp_path)
        assert (result.status, result.detail, result.error) == ("unchecked", why, error)


def test_once_the_hub_is_down_the_rest_are_not_tried(tmp_path):
    import httpx

    hub = FakeHub({}, error=httpx.ConnectTimeout("down"))
    manifests = [
        _pulled(tmp_path, ["a.gguf"], name="a"),
        _pulled(tmp_path, ["b.gguf"], name="b"),
        _pulled(tmp_path, ["c.gguf"], name="c", repo_id="local/c"),
    ]
    results = check_all(manifests, hub, tmp_path)
    assert [(r.name, r.status, r.detail) for r in results] == [
        ("a", "unchecked", "unreachable"),
        ("b", "unchecked", "unreachable"),
        ("c", "local", ""),
    ]
    assert hub.calls == 1


def test_the_command_for_each_kind():
    gguf = ModelManifest("m", REPO, "/x", "gguf", quantization="Q8_0", revision="v2")
    folder = ModelManifest("m", REPO, "/x", "safetensors")
    assert update_command(gguf) == f"hfl pull {REPO} -q Q8_0 --revision v2"
    assert update_command(folder) == f"hfl pull {REPO} --format safetensors"


def test_the_command_exits_1_when_some_could_not_be_checked(temp_config):
    from typer.testing import CliRunner

    from hfl.cli.main import app
    from hfl.models.registry import ModelRegistry

    ModelRegistry().add(_pulled(temp_config.models_dir, ["m.gguf"], name="m"))
    down = [Check("m", "unchecked", detail="unreachable", error="ConnectError")]
    with patch("hfl.hub.outdated.check_all", return_value=down):
        result = CliRunner().invoke(app, ["outdated"], env={"COLUMNS": "200"})
    assert result.exit_code == 1
    row = next(line for line in result.output.splitlines() if "│ m " in line)
    assert "could not check" in row and "not the same as up to date" in result.output
    fine = [Check("m", "current")]
    with patch("hfl.hub.outdated.check_all", return_value=fine):
        assert CliRunner().invoke(app, ["outdated"]).exit_code == 0


def test_pulling_again_keeps_the_alias(temp_config):
    """The update ``hfl outdated`` prints is a pull; the entry it replaced
    used to lose its alias, and every client using it the model."""
    from typer.testing import CliRunner

    from hfl.cli.main import app
    from hfl.hub.resolver import ResolvedModel
    from hfl.models.registry import ModelRegistry

    path = temp_config.models_dir / "test--model" / "model-Q4_K_M.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF content")
    resolved = ResolvedModel(
        repo_id="test/model", filename=path.name, format="gguf", quantization="Q4_K_M"
    )
    with (
        patch("hfl.hub.resolver.resolve", return_value=resolved),
        patch("hfl.hub.downloader.pull_model", return_value=path),
    ):
        first = CliRunner().invoke(app, ["pull", "test/model", "--skip-license", "-a", "mine"])
        again = CliRunner().invoke(app, ["pull", "test/model", "--skip-license"])
    assert first.exit_code == 0 and again.exit_code == 0, again.output
    assert ModelRegistry().get("model-q4_k_m").alias == "mine"

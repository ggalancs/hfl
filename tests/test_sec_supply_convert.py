# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Converting a pulled model must not run the repo's code, and the
converter that runs is the pinned, checked llama.cpp release."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hfl.converter import gguf_converter
from hfl.exceptions import ConversionError


@pytest.fixture(autouse=True)
def _no_remote_code(monkeypatch):
    monkeypatch.delenv("HFL_ALLOW_REMOTE_CODE", raising=False)


def _model(tmp_path: Path, **files: object) -> Path:
    folder = tmp_path / "Org--model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "llama"}))
    for name, content in files.items():
        path = folder / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content))
    return folder


# -- finding 1: repo code at convert time ---------------------------------


@pytest.mark.parametrize(
    "files, culprit",
    [
        ({"config.json": {"model_type": "qwen", "auto_map": {"AutoConfig": "x.C"}}}, "config.json"),
        (
            {"tokenizer_config.json": {"auto_map": {"AutoTokenizer": ["evil/repo--tok.T", None]}}},
            "tokenizer_config.json",
        ),
        ({"tokenization_qwen.py": "import os"}, "tokenization_qwen.py"),
        ({"sub__modeling.py": "import os"}, "sub/modeling.py"),
    ],
)
def test_the_pull_refuses_to_convert_a_repo_that_carries_code(tmp_path, files, culprit):
    from hfl.hub.pull_service import PullStepError, _convert

    folder = _model(tmp_path, **files)
    with (
        patch.object(gguf_converter.GGUFConverter, "convert") as convert,
        pytest.raises(PullStepError) as caught,
    ):
        _convert(folder, "org/model", "Q4_K_M")
    assert caught.value.key == "errors.cannot_convert_gguf"
    assert culprit in caught.value.values["reason"]
    assert "HFL_ALLOW_REMOTE_CODE" in caught.value.values["reason"]
    convert.assert_not_called()


def test_the_converter_itself_refuses_before_fetching_anything(tmp_path, temp_config):
    folder = _model(tmp_path, **{"tokenizer_config.json": {"auto_map": {"AutoTokenizer": "t.T"}}})
    converter = gguf_converter.GGUFConverter()
    with (
        patch.object(converter, "ensure_tools") as ensure,
        patch.object(gguf_converter.subprocess, "run") as run,
        pytest.raises(ConversionError),
    ):
        converter.convert(folder, tmp_path / "out", "Q4_K_M")
    ensure.assert_not_called()
    run.assert_not_called()


def test_a_plain_repo_and_an_opted_in_operator_are_not_refused(tmp_path, monkeypatch):
    plain = _model(tmp_path)
    (plain / ".cache" / "huggingface").mkdir(parents=True)
    (plain / ".cache" / "huggingface" / "x.py").write_text("")  # hub's cache, not the model
    assert gguf_converter.check_remote_code(plain) is None
    (plain / "modeling.py").write_text("import os")
    assert gguf_converter.check_remote_code(plain) is not None
    monkeypatch.setenv("HFL_ALLOW_REMOTE_CODE", "1")
    assert gguf_converter.check_remote_code(plain) is None


def test_the_converter_runs_offline_and_without_the_hub_token(tmp_path, temp_config, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secret")
    monkeypatch.setenv("NO_LOCAL_GGUF", "1")
    folder = _model(tmp_path)
    converter = gguf_converter.GGUFConverter()
    seen: list[dict] = []

    def run(cmd, **kwargs):
        if str(converter.convert_script) in cmd:
            seen.append(kwargs.get("env"))
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF" * 10)
        return MagicMock(returncode=0)

    with (
        patch.object(converter, "ensure_tools"),
        patch.object(converter, "_check_conversion_environment"),
        patch.object(gguf_converter.subprocess, "run", side_effect=run),
    ):
        converter.convert(folder, tmp_path / "out", "F16")
    assert len(seen) == 1 and seen[0] is not None
    env = seen[0]
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    assert "NO_LOCAL_GGUF" not in env


def test_a_whole_repo_download_leaves_its_python_behind(tmp_path, monkeypatch):
    from huggingface_hub.utils import filter_repo_objects

    from hfl.hub import downloader
    from hfl.hub.resolver import ResolvedModel

    calls: list[dict] = []
    monkeypatch.setattr(downloader, "ensure_auth", lambda repo: None)
    monkeypatch.setattr(downloader, "_rate_limit", lambda: None)
    monkeypatch.setattr(downloader, "_verify_downloads", lambda *a, **k: None)
    monkeypatch.setattr(
        downloader, "snapshot_download", lambda **kw: calls.append(kw) or str(tmp_path)
    )
    monkeypatch.setattr(downloader.config, "home_dir", tmp_path, raising=False)
    downloader.pull_model(ResolvedModel(repo_id="o/r", format="pytorch"))
    ignored = calls[-1]["ignore_patterns"]
    # The Hub client's own filter is what applies the patterns.
    kept = list(
        filter_repo_objects(
            ["pytorch_model.bin", "modeling_x.py", "sub/tok.py", "config.json"],
            ignore_patterns=ignored,
        )
    )
    assert kept == ["pytorch_model.bin", "config.json"]
    planned = downloader._planned(ResolvedModel(repo_id="o/r", format="pytorch"), ["a.py", "b.bin"])
    assert planned == ["b.bin"]

    monkeypatch.setenv("HFL_ALLOW_REMOTE_CODE", "1")
    downloader.pull_model(ResolvedModel(repo_id="o/r", format="pytorch"))
    assert calls[-1]["ignore_patterns"] is None


# -- finding 2: the converter is pinned and checked ----------------------


def test_the_converter_is_the_release_hfl_pins_for_llama_server():
    from hfl.engine.llama_server_dist import RELEASE

    assert gguf_converter.LLAMA_CPP_BRANCH == RELEASE
    assert gguf_converter.LLAMA_CPP_ARCHIVE.endswith(f"/refs/tags/{RELEASE}")
    assert len(gguf_converter.LLAMA_CPP_SHA256) == 64
    assert len(gguf_converter.LLAMA_CPP_COMMIT) == 40


def _serve(monkeypatch, raw: bytes) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            yield raw

    @contextmanager
    def stream(*a, **k):
        yield Response()

    monkeypatch.setattr("httpx.stream", stream)


def _archive() -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        data = b"print(1)"
        info = tarfile.TarInfo("llama.cpp-b1/convert_hf_to_gguf.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def test_an_archive_that_is_not_the_pinned_one_is_not_unpacked(tmp_path, monkeypatch):
    _serve(monkeypatch, _archive())
    target = tmp_path / "tools" / "llama.cpp"
    target.parent.mkdir()
    with pytest.raises(ConversionError, match="sha256"):
        gguf_converter._download_converter(target)
    assert not target.exists()


def test_the_pinned_archive_is_unpacked(tmp_path, monkeypatch):
    raw = _archive()
    _serve(monkeypatch, raw)
    monkeypatch.setattr(gguf_converter, "LLAMA_CPP_SHA256", hashlib.sha256(raw).hexdigest())
    target = tmp_path / "tools" / "llama.cpp"
    target.parent.mkdir()
    gguf_converter._download_converter(target)
    assert (target / "convert_hf_to_gguf.py").read_text() == "print(1)"


def _git(url: str, head: str):
    def run(cmd, **kwargs):
        out = url if cmd[:2] == ["git", "config"] else head
        return MagicMock(returncode=0, stdout=out + "\n")

    return run


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/ggml-org/llama.cpp.gitgit",  # rstrip(".git") ate it all
        "https://github.com/ggml-org/llama.cpptig",
        "https://github.com/evil/llama.cpp.git",
    ],
)
def test_a_clone_from_elsewhere_fails_verification(tmp_path, url):
    with patch.object(
        gguf_converter.subprocess, "run", side_effect=_git(url, gguf_converter.LLAMA_CPP_COMMIT)
    ):
        assert not gguf_converter._verify_git_clone(tmp_path, gguf_converter.LLAMA_CPP_REPO)


def test_a_clone_at_another_commit_fails_verification(tmp_path):
    repo = gguf_converter.LLAMA_CPP_REPO
    with patch.object(gguf_converter.subprocess, "run", side_effect=_git(repo, "0" * 40)):
        assert not gguf_converter._verify_git_clone(tmp_path, repo)
    pinned = gguf_converter.LLAMA_CPP_COMMIT
    for url in (repo, repo.removesuffix(".git")):
        with patch.object(gguf_converter.subprocess, "run", side_effect=_git(url, pinned)):
            assert gguf_converter._verify_git_clone(tmp_path, repo)


def test_a_converter_fetched_before_the_pin_is_fetched_again(tmp_path, temp_config, monkeypatch):
    converter = gguf_converter.GGUFConverter()
    old = converter.llama_cpp_dir
    (old / "gguf-py").mkdir(parents=True)
    converter.convert_script.write_text("# master, unchecked")
    fetched: list[Path] = []

    def download(target: Path) -> None:
        fetched.append(target)
        target.mkdir()
        (target / "convert_hf_to_gguf.py").write_text("# pinned")

    monkeypatch.setattr(gguf_converter.shutil, "which", lambda tool: None)  # no git: archive
    monkeypatch.setattr(gguf_converter, "_download_converter", download)
    converter.ensure_tools()
    assert fetched == [old]
    assert converter.convert_script.read_text() == "# pinned"
    assert (old / ".hfl-release").read_text() == gguf_converter.LLAMA_CPP_BRANCH
    aside = [p for p in old.parent.iterdir() if p.name.startswith("llama.cpp.unpinned-")]
    assert len(aside) == 1  # set aside, not deleted
    converter.ensure_tools()  # now pinned: nothing fetched
    assert fetched == [old]

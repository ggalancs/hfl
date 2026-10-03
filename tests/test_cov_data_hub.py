# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Hub helpers without the network: chat-template repair for MLX pulls,
the local-cache probes, and the downloader's checksum/progress helpers.
Every Hub call is faked; nothing is fetched."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfl.hub import chat_template_repair as ctr
from hfl.hub import downloader as dl
from hfl.hub import local_cache as lc
from hfl.hub.resolver import ResolvedModel

# ----------------------------------------------------------------------
# chat_template_repair
# ----------------------------------------------------------------------


class TestHasChatTemplate:
    def test_sources(self, tmp_path):
        assert ctr.has_chat_template(tmp_path) is False
        (tmp_path / "tokenizer_config.json").write_text("{bad json")
        assert ctr.has_chat_template(tmp_path) is False
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": ""}))
        assert ctr.has_chat_template(tmp_path) is False
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "T"}))
        assert ctr.has_chat_template(tmp_path) is True
        (tmp_path / "tokenizer_config.json").unlink()
        (tmp_path / "chat_template.jinja").write_text("J")
        assert ctr.has_chat_template(tmp_path) is True


def _fake_api(monkeypatch, info=None, error=None):
    import huggingface_hub

    class _Api:
        def model_info(self, repo_id, **kw):
            if error is not None:
                raise error
            return info

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)


class TestBaseRepo:
    @pytest.mark.parametrize(
        "base, expected",
        [(["google/gemma-4-31b-it", "x"], "google/gemma-4-31b-it"), ("org/m", "org/m"), ([], None)],
    )
    def test_card_base_model(self, monkeypatch, base, expected):
        _fake_api(monkeypatch, SimpleNamespace(card_data=SimpleNamespace(base_model=base)))
        assert ctr._base_repo_from_card("mlx-community/x") == expected

    def test_card_missing_or_api_failure(self, monkeypatch):
        _fake_api(monkeypatch, SimpleNamespace(card_data=None))
        assert ctr._base_repo_from_card("a/b") is None
        _fake_api(monkeypatch, error=RuntimeError("401"))
        assert ctr._base_repo_from_card("a/b") is None

    def test_without_huggingface_hub(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        assert ctr._base_repo_from_card("a/b") is None
        assert ctr._try_download("a/b", "f", Path("/nonexistent")) is False

    @pytest.mark.parametrize(
        "repo_id, expected",
        [
            ("mlx-community/gemma-4-31b-it-4bit", "google/gemma-4-31b-it"),
            ("mlx-community/Llama-3.2-1B-MLX-8bit", "meta-llama/Llama-3.2-1B"),
            ("lmstudio-community/Qwen3-8B-MLX", "Qwen/Qwen3-8B"),
            ("mlx-community/Mixtral-8x7B-bf16", "mistralai/Mixtral-8x7B"),
            ("mlx-community/phi-4-4bit", None),  # unknown family
            ("someone/gemma-4bit", None),  # unknown quantiser org
            ("mlx-community/gemma-2b", None),  # no quant suffix
            ("no-slash", None),
        ],
    )
    def test_heuristic(self, repo_id, expected):
        assert ctr._heuristic_base_repo(repo_id) == expected

    def test_resolve_prefers_card(self, monkeypatch):
        monkeypatch.setattr(ctr, "_base_repo_from_card", lambda r: "card/base")
        assert ctr._resolve_base_repo("mlx-community/gemma-4bit") == "card/base"
        monkeypatch.setattr(ctr, "_base_repo_from_card", lambda r: None)
        assert ctr._resolve_base_repo("mlx-community/gemma-4bit") == "google/gemma"


def _fake_download(monkeypatch, files: dict[str, Path | Exception]):
    import huggingface_hub

    calls = []

    def hf_hub_download(repo, filename):
        calls.append((repo, filename))
        found = files.get(filename, FileNotFoundError(filename))
        if isinstance(found, Exception):
            raise found
        return str(found)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    return calls


class TestTryDownload:
    def test_copy_success_and_failures(self, tmp_path, monkeypatch):
        src = tmp_path / "src.jinja"
        src.write_text("TEMPLATE")
        _fake_download(monkeypatch, {"chat_template.jinja": src})
        dest = tmp_path / "out.jinja"
        assert ctr._try_download("b/r", "chat_template.jinja", dest) is True
        assert dest.read_text() == "TEMPLATE"
        assert ctr._try_download("b/r", "missing", dest) is False
        # Copy into a directory that does not exist.
        assert ctr._try_download("b/r", "chat_template.jinja", tmp_path / "no" / "x") is False


class TestEnsureChatTemplate:
    def test_already_present(self, tmp_path):
        (tmp_path / "chat_template.jinja").write_text("x")
        assert ctr.ensure_chat_template(tmp_path, "a/b") is True

    def test_no_base_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ctr, "_resolve_base_repo", lambda r: None)
        assert ctr.ensure_chat_template(tmp_path, "a/b") is False

    def test_jinja_from_base(self, tmp_path, monkeypatch):
        src = tmp_path / "upstream.jinja"
        src.write_text("{{ messages }}")
        model = tmp_path / "model"
        model.mkdir()
        monkeypatch.setattr(ctr, "_resolve_base_repo", lambda r: "google/gemma")
        calls = _fake_download(monkeypatch, {"chat_template.jinja": src})
        assert ctr.ensure_chat_template(model, "mlx-community/gemma-4bit") is True
        assert (model / "chat_template.jinja").read_text() == "{{ messages }}"
        assert calls == [("google/gemma", "chat_template.jinja")]

    def test_merge_from_base_tokenizer_config(self, tmp_path, monkeypatch):
        upstream = tmp_path / "upstream.json"
        upstream.write_text(json.dumps({"chat_template": "T", "eos_token": "<base>"}))
        model = tmp_path / "model"
        model.mkdir()
        (model / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<mlx>"}))
        monkeypatch.setattr(ctr, "_resolve_base_repo", lambda r: "google/gemma")
        _fake_download(monkeypatch, {"tokenizer_config.json": upstream})
        assert ctr.ensure_chat_template(model, "mlx-community/gemma-4bit") is True
        merged = json.loads((model / "tokenizer_config.json").read_text())
        assert merged == {"eos_token": "<mlx>", "chat_template": "T"}  # only the template added
        assert not (model / "_base_tokenizer_config.tmp.json").exists()

    @pytest.mark.parametrize(
        "upstream_text, local_text",
        [
            (None, "{}"),  # base has no tokenizer_config either
            ("{broken", "{}"),  # base config unparseable
            (json.dumps({"eos_token": "x"}), "{}"),  # base has no template
            (json.dumps({"chat_template": "T"}), None),  # no local config to merge into
        ],
    )
    def test_merge_gives_up(self, tmp_path, monkeypatch, upstream_text, local_text):
        model = tmp_path / "model"
        model.mkdir()
        if local_text is not None:
            (model / "tokenizer_config.json").write_text(local_text)
        files: dict = {}
        if upstream_text is not None:
            upstream = tmp_path / "upstream.json"
            upstream.write_text(upstream_text)
            files["tokenizer_config.json"] = upstream
        monkeypatch.setattr(ctr, "_resolve_base_repo", lambda r: "google/gemma")
        _fake_download(monkeypatch, files)
        assert ctr.ensure_chat_template(model, "mlx-community/gemma-4bit") is False
        assert not (model / "_base_tokenizer_config.tmp.json").exists()
        if local_text is not None:
            assert (model / "tokenizer_config.json").read_text() == local_text

    def test_merge_write_failure(self, tmp_path, monkeypatch):
        upstream = tmp_path / "upstream.json"
        upstream.write_text(json.dumps({"chat_template": "T"}))
        model = tmp_path / "model"
        model.mkdir()
        cfg = model / "tokenizer_config.json"
        cfg.write_text("{}")
        monkeypatch.setattr(ctr, "_resolve_base_repo", lambda r: "google/gemma")
        _fake_download(monkeypatch, {"tokenizer_config.json": upstream})
        real_open = Path.open

        def open_(self, mode="r", *a, **kw):
            if self == cfg and "w" in mode:
                raise OSError("read-only")
            return real_open(self, mode, *a, **kw)

        monkeypatch.setattr(Path, "open", open_)
        assert ctr.ensure_chat_template(model, "mlx-community/gemma-4bit") is False


# ----------------------------------------------------------------------
# local_cache
# ----------------------------------------------------------------------


class TestLocalCache:
    def test_hub_model_available_locally(self, monkeypatch):
        import huggingface_hub

        seen = []

        def snapshot_download(model, local_files_only):
            seen.append((model, local_files_only))
            if model == "absent/model":
                raise FileNotFoundError(model)
            return "/cache/x"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
        assert lc.hub_model_available_locally("present/model") is True
        assert lc.hub_model_available_locally("absent/model") is False
        assert seen == [("present/model", True), ("absent/model", True)]

    def test_whisper_via_faster_whisper(self, monkeypatch):
        utils = types.ModuleType("faster_whisper.utils")

        def download_model(model, local_files_only):
            assert local_files_only is True
            if model != "small":
                raise RuntimeError("not cached")
            return "/cache/small"

        utils.download_model = download_model
        monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
        monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
        assert lc.whisper_available_locally("small") is True
        assert lc.whisper_available_locally("large-v3") is False

    def test_whisper_falls_back_to_openai_cache(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        monkeypatch.setitem(sys.modules, "faster_whisper.utils", None)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        (tmp_path / "whisper").mkdir()
        (tmp_path / "whisper" / "tiny.en.pt").write_bytes(b"x")
        assert lc.whisper_available_locally("tiny.en") is True
        assert lc.whisper_available_locally("base") is False
        assert lc.openai_whisper_cached("tiny.en") is True

    @pytest.mark.parametrize(
        "name, ok",
        [("org/model", True), ("large-v3", True), ("/etc/passwd", False), ("../x", False)],
    )
    def test_is_model_id(self, name, ok):
        assert lc.is_model_id(name) is ok


# ----------------------------------------------------------------------
# downloader helpers
# ----------------------------------------------------------------------


class TestDownloaderHelpers:
    def test_retryable_without_httpx_falls_back_to_stdlib(self, monkeypatch):
        def import_module(name):
            raise ImportError(name)

        monkeypatch.setattr(dl.importlib, "import_module", import_module)
        assert dl._retryable() == (ConnectionError, TimeoutError)

    def test_retryable_skips_module_without_transport_error(self, monkeypatch):
        fake = types.ModuleType("httpx")

        class TransportError(Exception):
            pass

        fake.TransportError = TransportError
        modules = {"httpx": fake, "httpx2": types.ModuleType("httpx2")}
        monkeypatch.setattr(dl.importlib, "import_module", lambda name: modules[name])
        assert dl._retryable() == (TransportError,)

    @pytest.mark.hub_sizes  # opt out of conftest's stub of _hub_sha256
    def test_hub_sha256(self, monkeypatch):
        siblings = [
            SimpleNamespace(rfilename="w.safetensors", lfs=SimpleNamespace(sha256="ABC")),
            SimpleNamespace(rfilename="config.json", lfs=None),
            SimpleNamespace(rfilename="odd.bin", lfs=SimpleNamespace(sha256="")),
        ]
        seen = {}
        import huggingface_hub

        class _Api:
            def model_info(self, repo_id, **kw):
                seen.update(kw, repo_id=repo_id)
                return SimpleNamespace(siblings=siblings)

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
        resolved = ResolvedModel(repo_id="o/m", commit_sha="deadbeef")
        assert dl._hub_sha256(resolved, "tok") == {"w.safetensors": "abc"}
        assert seen["revision"] == "deadbeef" and seen["token"] == "tok"
        assert seen["files_metadata"] is True

    @pytest.mark.hub_sizes
    def test_hub_sha256_failure_is_empty(self, monkeypatch):
        import huggingface_hub

        class _Api:
            def model_info(self, repo_id, **kw):
                raise RuntimeError("offline")

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
        assert dl._hub_sha256(ResolvedModel(repo_id="o/m"), None) == {}

    def test_token_quietly(self, monkeypatch):
        def boom(repo_id):
            raise RuntimeError("gated")

        monkeypatch.setattr(dl, "ensure_auth", boom)
        assert dl._token_quietly("o/m") is None
        monkeypatch.setattr(dl, "ensure_auth", lambda repo_id: "hf_x")
        assert dl._token_quietly("o/m") == "hf_x"

    def test_bytes_done_counts_partial_downloads(self, temp_config, monkeypatch):
        resolved = ResolvedModel(repo_id="o/m")
        folder = dl.model_dir_for(resolved)
        partial = folder / ".cache" / "huggingface" / "download"
        partial.mkdir(parents=True)
        (folder / "a.safetensors").write_bytes(b"x" * 5)
        (partial / "b.incomplete").write_bytes(b"y" * 3)
        assert dl.bytes_done(resolved, {"a.safetensors": 5, "b.safetensors": 9}) == 8

        def rglob(self, pattern):
            raise OSError("vanished")

        monkeypatch.setattr(Path, "rglob", rglob)
        assert dl.bytes_done(resolved, {"a.safetensors": 5}) == 5

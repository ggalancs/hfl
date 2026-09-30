# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Deleting a model: ``hfl rm`` and ``DELETE /api/delete`` share one rule.

HFL deletes only what lives in its own models folder. A registry entry
pointing anywhere else — a GGUF of yours registered in place, a leftover
pointing at /tmp — loses its entry and keeps its file. A file another
entry still uses (``hfl cp`` is zero-copy) is kept too.

``/api/delete`` is Ollama's route. It was left out on purpose so the API
could not destroy models; it now exists for the owner only: a loopback
peer, never a remote one (not even with HFL_ALLOW_REMOTE_PULL), and never
from a web page (foreign Origin).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from hfl.models.manifest import ModelManifest


def _register(temp_config, name, path):
    from hfl.models.registry import get_registry

    manifest = ModelManifest(name=name, repo_id=f"org/{name}", local_path=str(path), format="gguf")
    get_registry().add(manifest)
    return manifest


def _gguf(directory, name="m.gguf"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"GGUF")
    return path


class TestTheRule:
    def test_a_file_in_the_models_folder_is_deleted(self, temp_config):
        from hfl.models.registry import get_registry
        from hfl.models.removal import remove_model

        path = _gguf(temp_config.models_dir / "org--m")
        manifest = _register(temp_config, "m", path)
        result = remove_model(get_registry(), manifest)
        assert result.deleted and not path.exists()
        assert get_registry().get("m") is None

    def test_a_folder_in_the_models_folder_is_deleted(self, temp_config):
        from hfl.models.registry import get_registry
        from hfl.models.removal import remove_model

        folder = temp_config.models_dir / "org--tts"
        _gguf(folder, "weights.bin")
        remove_model(get_registry(), _register(temp_config, "tts", folder))
        assert not folder.exists()

    def test_a_file_outside_is_kept(self, temp_config, tmp_path):
        from hfl.models.registry import get_registry
        from hfl.models.removal import remove_model

        mine = _gguf(tmp_path / "my-own-models")
        result = remove_model(get_registry(), _register(temp_config, "mine", mine))
        assert mine.exists() and not result.deleted
        assert result.kept_outside == mine
        assert get_registry().get("mine") is None

    def test_the_models_folder_itself_is_never_deleted(self, temp_config):
        from hfl.models.registry import get_registry
        from hfl.models.removal import remove_model

        _gguf(temp_config.models_dir / "org--other")
        result = remove_model(get_registry(), _register(temp_config, "x", temp_config.models_dir))
        assert temp_config.models_dir.exists() and not result.deleted

    def test_a_shared_file_is_kept(self, temp_config):
        from hfl.models.registry import get_registry
        from hfl.models.removal import remove_model

        path = _gguf(temp_config.models_dir / "org--m")
        manifest = _register(temp_config, "a", path)
        _register(temp_config, "b", path)
        result = remove_model(get_registry(), manifest)
        assert path.exists() and result.shared_with == ["b"]


@pytest.fixture
def owner(temp_config):
    from hfl.api.server import app
    from hfl.api.state import reset_state

    reset_state()
    yield TestClient(app, client=("127.0.0.1", 50000))
    reset_state()


class TestTheRoute:
    def test_the_owner_deletes(self, owner, temp_config):
        path = _gguf(temp_config.models_dir / "org--m")
        _register(temp_config, "m", path)
        response = owner.request("DELETE", "/api/delete", json={"model": "m"})
        assert response.status_code == 200
        assert not path.exists()

    def test_curl_without_a_content_type(self, owner, temp_config):
        path = _gguf(temp_config.models_dir / "org--m")
        _register(temp_config, "m", path)
        response = owner.request(
            "DELETE",
            "/api/delete",
            content=b'{"model": "m"}',
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200 and not path.exists()

    def test_the_legacy_name_field(self, owner, temp_config):
        _register(temp_config, "m", _gguf(temp_config.models_dir / "org--m"))
        assert owner.request("DELETE", "/api/delete", json={"name": "m"}).status_code == 200

    def test_an_unknown_model_is_404(self, owner):
        response = owner.request("DELETE", "/api/delete", json={"model": "nope"})
        assert response.status_code == 404

    def test_a_loaded_model_is_unloaded_first(self, owner, temp_config, monkeypatch):
        from hfl.api.state import get_state

        _register(temp_config, "m", _gguf(temp_config.models_dir / "org--m"))
        evict = AsyncMock(return_value=True)
        monkeypatch.setattr(get_state(), "evict", evict)
        assert owner.request("DELETE", "/api/delete", json={"model": "m"}).status_code == 200
        evict.assert_awaited_once()
        assert evict.await_args.args[0] == "m"

    def test_a_remote_peer_is_refused_even_with_remote_admin_on(self, temp_config, monkeypatch):
        from hfl.api.server import app

        monkeypatch.setattr(temp_config, "allow_remote_pull", True, raising=False)
        import hfl.config

        monkeypatch.setattr(hfl.config.config, "allow_remote_pull", True, raising=False)
        path = _gguf(temp_config.models_dir / "org--m")
        _register(temp_config, "m", path)
        remote = TestClient(app, client=("192.168.1.50", 50000))
        response = remote.request("DELETE", "/api/delete", json={"model": "m"})
        assert response.status_code == 403
        assert path.exists()

    def test_a_web_page_is_refused(self, owner, temp_config):
        path = _gguf(temp_config.models_dir / "org--m")
        _register(temp_config, "m", path)
        response = owner.request(
            "DELETE", "/api/delete", json={"model": "m"}, headers={"Origin": "https://evil.test"}
        )
        assert response.status_code == 403
        assert path.exists()


def _held_open(monkeypatch):
    """Windows' refusal to move or delete a file another process has open
    (WinError 32 / 5, measured on a real Windows 10 with a mapped model)."""
    from pathlib import Path

    def refuse(self, target):
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(Path, "rename", refuse)


class TestInUse:
    """A running ``hfl serve`` maps the model it serves; on Windows ``hfl rm``
    died in rmtree with a traceback, part of the folder already deleted."""

    def test_nothing_is_deleted_and_the_entry_stays(self, temp_config, monkeypatch):
        from hfl.models.registry import get_registry
        from hfl.models.removal import ModelInUse, remove_model

        folder = temp_config.models_dir / "org--m"
        path = _gguf(folder)
        (folder / "README.md").write_text("card")
        manifest = _register(temp_config, "m", folder)
        _held_open(monkeypatch)
        with pytest.raises(ModelInUse):
            remove_model(get_registry(), manifest)
        assert path.exists() and (folder / "README.md").exists()
        assert get_registry().get("m") is not None

    def test_the_cli_says_so_without_a_traceback(self, temp_config, monkeypatch):
        from typer.testing import CliRunner

        from hfl.cli.main import app

        _register(temp_config, "m", _gguf(temp_config.models_dir / "org--m"))
        _held_open(monkeypatch)
        result = CliRunner().invoke(app, ["rm", "m", "--yes"])
        assert result.exit_code == 1 and "hfl stop m" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_the_route_answers_409(self, owner, temp_config, monkeypatch):
        _register(temp_config, "m", _gguf(temp_config.models_dir / "org--m"))
        _held_open(monkeypatch)
        response = owner.request("DELETE", "/api/delete", json={"model": "m"})
        assert response.status_code == 409 and "in use" in response.text


def test_a_file_renamed_but_not_deleted_goes_back(temp_config, monkeypatch):
    """llama.cpp opens a model so Windows lets it be renamed, not deleted
    (measured). The next `hfl rm` then found nothing under its name, dropped
    the entry and left the renamed file on disk for good."""
    from pathlib import Path

    from hfl.models.registry import get_registry
    from hfl.models.removal import ModelInUse, remove_model

    path = _gguf(temp_config.models_dir / "org--m")
    manifest = _register(temp_config, "m", path)

    def refuse(self, missing_ok=False):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "unlink", refuse)
    with pytest.raises(ModelInUse):
        remove_model(get_registry(), manifest)
    assert path.exists() and [p.name for p in path.parent.iterdir()] == [path.name]
    assert get_registry().get("m") is not None


class _Embedder:
    is_loaded = True

    def __init__(self) -> None:
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True


def _serving_embeddings(temp_config, name: str) -> _Embedder:
    from hfl.api.state import get_state

    _register(temp_config, name, _gguf(temp_config.models_dir / "org--e"))
    state, engine = get_state(), _Embedder()
    state._embed_engine, state._embed_model_name = engine, name
    return engine


class TestEmbeddingModels:
    """``hfl stop`` and ``/api/delete`` reached only LLMs and TTS: an
    embedding model stayed loaded (found on Windows, where it then could
    not be deleted)."""

    def test_stop_by_name_unloads_it(self, owner, temp_config):
        from hfl.api.state import get_state

        engine = _serving_embeddings(temp_config, "nomic-embed")
        out = owner.post("/api/stop", json={"model": "nomic-embed"}).json()
        assert out["status"] == "stopped" and engine.unloaded
        assert get_state()._embed_engine is None

    def test_stop_all_unloads_it(self, owner, temp_config):
        engine = _serving_embeddings(temp_config, "nomic-embed")
        out = owner.post("/api/stop", json={}).json()
        assert out["status"] == "stopped" and engine.unloaded

    def test_another_name_leaves_it_loaded(self, owner, temp_config):
        engine = _serving_embeddings(temp_config, "nomic-embed")
        _register(temp_config, "other", _gguf(temp_config.models_dir / "org--o"))
        owner.post("/api/stop", json={"model": "other"})
        assert not engine.unloaded

    def test_delete_unloads_it_first(self, owner, temp_config):
        engine = _serving_embeddings(temp_config, "nomic-embed")
        response = owner.request("DELETE", "/api/delete", json={"model": "nomic-embed"})
        assert response.status_code == 200 and engine.unloaded

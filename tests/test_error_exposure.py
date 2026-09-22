# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""One invariant, across every endpoint that catches a foreign exception.

CodeQL's ``py/stack-trace-exposure`` only sees the handlers that write
straight into a ``JSONResponse`` / ``StreamingResponse``; it does not model
``raise HTTPException(detail=...)``, so it flagged six sites and stayed quiet
about the rest of the same family. These tests pin the family instead of the
six: **no message an HFL endpoint sends a caller may contain the text of an
exception HFL did not write.**

The distinction that decides each case is authorship, not exception type:

* ``ModelNotFoundError``, ``InvalidBlobDigestError``, ``WebSearchError``,
  ``ValidationError`` … carry sentences the project wrote for the caller,
  and those still go out verbatim — turning them into references would
  make the API worse for no gain.
* Anything from the OS, the Hub SDK or an inference backend quotes paths,
  URLs and library internals. Those go to the log, and the caller gets the
  step that failed plus a reference that joins the two.

Where a leak would reach an *unauthenticated remote user* (no
``require_owner``) it is marked below, because that is the case that
actually matters.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import reset_state

LOCAL_PEER = ("127.0.0.1", 5555)

# A path that must never appear in a response body. Every stub below raises
# an exception carrying it.
SECRET_PATH = "/Users/secret/.hfl/models/blobs/sha256-deadbeef"


@pytest.fixture
def client(temp_config):
    reset_state()
    yield TestClient(app, client=LOCAL_PEER)
    reset_state()


def _no_secret(body: str) -> None:
    assert SECRET_PATH not in body, f"server path leaked to the client: {body}"
    assert "/Users/secret" not in body


class TestModelBlobUnreadable:
    """``load_llm`` raises ``ModelNotFoundError`` when a model is not
    registered. A ``FileNotFoundError`` means the *blob* could not be opened,
    and the OS message is the path it tried."""

    @pytest.mark.parametrize(
        ("path", "method", "body", "remote_reachable"),
        [
            ("/api/verify/qwen", "post", None, True),
            ("/api/lora/qwen", "get", None, True),  # no require_owner
            ("/api/snapshot/save", "post", {"model": "qwen", "name": "s1"}, False),
            ("/api/snapshot/load", "post", {"model": "qwen", "name": "s1"}, False),
        ],
    )
    def test_path_never_reaches_the_caller(self, client, path, method, body, remote_reachable):
        async def _unreadable(name, *args, **kwargs):
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{SECRET_PATH}'")

        targets = [
            "hfl.api.routes_verify.load_llm",
            "hfl.api.routes_lora.load_llm",
            "hfl.api.routes_snapshot.load_llm",
        ]
        with (
            patch(targets[0], _unreadable),
            patch(targets[1], _unreadable),
            patch(targets[2], _unreadable),
        ):
            response = getattr(client, method)(path, **({"json": body} if body else {}))

        assert response.status_code == 404
        _no_secret(response.text)
        # The caller still learns which model failed.
        assert "qwen" in response.text


class TestHubFailures:
    """huggingface_hub quotes request URLs — and, on an auth failure, what it
    was sent — in its exception messages. ``/api/discover`` has no owner
    guard, so a remote user reaches this handler."""

    @pytest.mark.parametrize(
        ("path", "target"),
        [
            ("/api/discover?q=llama", "hfl.api.routes_discover.search_hub"),
            ("/api/draft/recommend?model=qwen", "hfl.api.routes_draft.pick_draft_for"),
        ],
    )
    def test_hub_exception_text_is_not_forwarded(self, client, path, target, caplog):
        boom = RuntimeError(
            f"401 Client Error for https://huggingface.co/api?tok=hf_abc {SECRET_PATH}"
        )
        with caplog.at_level(logging.ERROR), patch(target, side_effect=boom):
            response = client.get(path)

        assert response.status_code == 503
        _no_secret(response.text)
        assert "hf_abc" not in response.text
        assert "401 Client Error" not in response.text
        # ...and the operator still gets all of it, joined by the reference.
        ref = response.json()["detail"].split("(ref ")[1].split(")")[0]
        assert ref in caplog.text
        assert "401 Client Error" in caplog.text


class TestAuthoredMessagesSurvive:
    """The other half of the invariant: messages HFL wrote must NOT be
    swallowed, or the sweep would have made the API worse."""

    def test_malformed_blob_digest_keeps_its_message(self, client):
        response = client.request("HEAD", "/api/blobs/not-a-digest")
        assert response.status_code == 400

    def test_unknown_model_keeps_its_message(self, client):
        response = client.post("/api/verify/does-not-exist")
        assert response.status_code == 404
        assert "does-not-exist" in response.text

    def test_lora_scale_validation_keeps_its_message(self, client):
        """``scale must be in [0.0, 5.0]`` is ours and mentions only what the
        caller sent, so it goes out verbatim."""
        response = client.post(
            "/api/lora/apply",
            json={"model": "qwen", "lora_path": "a.safetensors", "scale": 99.0},
        )
        # Rejected by the pydantic bound before the engine is touched.
        assert response.status_code in (400, 422)


class TestPathContainmentMessage:
    """``PathTraversalError`` names the base directory it contained against.
    That is the server's layout."""

    def test_rejection_does_not_print_the_data_dir(self, client):
        response = client.post(
            "/api/lora/apply",
            json={"model": "qwen", "lora_path": "../../../etc/passwd", "scale": 1.0},
        )
        assert response.status_code == 400
        assert "/etc/passwd" not in response.json()["detail"]
        assert ".hfl" not in response.json()["detail"]
        assert "HFL data dir" in response.json()["detail"]


class TestPushMissingFiles:
    """``build_upload_plan``'s ``FileNotFoundError`` prints
    ``manifest.local_path``; its ``ValueError`` is our own sentence about the
    destination the caller supplied. The route must treat them differently."""

    @pytest.fixture
    def registered(self, tmp_path):
        from hfl.core.container import get_registry
        from hfl.models.manifest import ModelManifest

        d = tmp_path / "qwen"
        d.mkdir()
        (d / "model.gguf").write_bytes(b"GGUF")
        manifest = ModelManifest(
            name="qwen-coder-7b",
            repo_id="Qwen/Qwen2.5",
            local_path=str(d),
            format="gguf",
        )
        registry = get_registry()
        registry.add(manifest)
        yield manifest
        registry.remove(manifest.name)

    def test_missing_files_do_not_print_local_path(self, client, registered, monkeypatch):
        def _missing(*args, **kwargs):
            raise FileNotFoundError(f"manifest.local_path does not exist: {SECRET_PATH}")

        monkeypatch.setattr("hfl.hub.uploader.build_upload_plan", _missing)

        response = client.post(
            "/api/push",
            json={"model": registered.name, "destination": "user/repo"},
        )
        assert response.status_code == 400
        _no_secret(response.text)
        assert registered.name in response.text

    def test_bad_destination_keeps_our_own_sentence(self, client, registered, monkeypatch):
        def _bad(*args, **kwargs):
            raise ValueError("target_repo_id must be ``namespace/model``: got 'nope'")

        monkeypatch.setattr("hfl.hub.uploader.build_upload_plan", _bad)

        response = client.post(
            "/api/push",
            json={"model": registered.name, "destination": "user/repo"},
        )
        assert response.status_code == 400
        assert "namespace/model" in response.json()["detail"]


class TestSnapshotWrapper:
    """``save_snapshot`` wraps the backend's failure. ``from exc`` keeps the
    cause for a traceback reader; ``str()`` of the RuntimeError must not."""

    def test_backend_failure_message_is_not_embedded(self):
        from hfl.engine import snapshot as snap

        engine = MagicMock()
        engine.save_state.side_effect = RuntimeError(f"mmap failed on {SECRET_PATH}")

        with pytest.raises(RuntimeError) as caught:
            snap.save_snapshot(engine, name="s1", model_name="qwen")

        assert SECRET_PATH not in str(caught.value)
        # The cause is still attached for anyone holding the traceback.
        assert caught.value.__cause__ is not None
        assert SECRET_PATH in str(caught.value.__cause__)


class TestWebSocketErrors:
    """``/ws/chat`` has no owner guard, and the backend exception is raised
    inside the producer thread — the frame the socket sends is the one built
    in ``routes_ws``."""

    def test_backend_stream_failure_sends_a_reference(self, client, caplog):
        from hfl.api.state import get_state
        from hfl.models.manifest import ModelManifest

        manifest = ModelManifest(
            name="qwen-coder-7b",
            repo_id="Qwen/Qwen2.5",
            local_path="/tmp/qwen.gguf",
            format="gguf",
        )
        engine = MagicMock()
        engine.is_loaded = True
        engine.chat_stream = MagicMock(
            side_effect=RuntimeError(f"CUDA error reading {SECRET_PATH}")
        )
        state = get_state()
        state.engine = engine
        state.current_model = manifest

        with caplog.at_level(logging.ERROR), client.websocket_connect("/ws/chat") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "chat",
                        "model": manifest.name,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            )
            frames = []
            for _ in range(20):
                frame = json.loads(ws.receive_text())
                frames.append(frame)
                if frame["type"] in ("error", "done"):
                    break

        error = next(f for f in frames if f["type"] == "error")
        _no_secret(json.dumps(frames))
        assert "chat stream failed" in error["message"]
        assert "CUDA error" not in error["message"]
        assert "CUDA error" in caplog.text

    def test_unknown_model_keeps_hfl_own_message(self, client):
        """``load_llm`` raises ``ModelNotFoundError`` — an HFL error whose
        sentence was written for the caller. The generic ``except`` around it
        must not swallow it into a reference."""
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "chat",
                        "model": "definitely-not-registered",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            )
            frames = []
            for _ in range(10):
                frame = json.loads(ws.receive_text())
                frames.append(frame)
                if frame["type"] in ("error", "done"):
                    break

        error = next(f for f in frames if f["type"] == "error")
        assert "definitely-not-registered" in error["message"]
        assert "(ref " not in error["message"]

    def test_malformed_frame_keeps_its_own_message(self, client):
        """``_validate_chat_frame`` raises ``ValueError`` with a sentence we
        wrote; the sweep must not have turned it into a reference."""
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "chat", "model": "", "messages": []}))
            frame = json.loads(ws.receive_text())

        assert frame["type"] == "error"
        assert "'model' must be a non-empty string" in frame["message"]

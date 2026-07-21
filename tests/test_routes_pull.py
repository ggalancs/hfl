# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the Ollama-compatible ``POST /api/pull`` endpoint.

Pins the NDJSON status stream shape — every status string
(``pulling manifest``, ``downloading``, ``verifying sha256 digest``,
``writing manifest``, ``success``, ``error``) and the event field
layout (``digest`` / ``total`` / ``completed``) is what Open WebUI
and ollama-python key off to render a progress bar.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app

# Loopback peer == the machine owner: pull is allowed. Any other address is
# a remote *user* and is refused (see hfl.api.admin_guard). The default
# TestClient peer is ("testclient", 50000), which would read as remote, so
# tests pin the peer explicitly.
LOCAL_PEER = ("127.0.0.1", 5555)
REMOTE_PEER = ("203.0.113.7", 5555)


def _permissive_license():
    """A PERMISSIVE ``LicenseInfo`` so the owner license gate lets the
    pull proceed without a real Hub round-trip."""
    from hfl.hub.license_checker import LicenseInfo, LicenseRisk

    return LicenseInfo(
        license_id="apache-2.0",
        license_name="Apache 2.0",
        risk=LicenseRisk.PERMISSIVE,
        restrictions=[],
        url="https://huggingface.co/acme/test-model#license",
        gated=False,
    )


@pytest.fixture
def client(temp_config):
    return TestClient(app, client=LOCAL_PEER)


@pytest.fixture
def mock_pull(tmp_path, temp_config):
    """Patch the downloader + resolver + license checker into a
    predictable local file. ``check_model_license`` returns a permissive
    license so the owner policy gate passes; ``temp_config`` isolates the
    registry/provenance writes the pull now performs."""
    fake_file = tmp_path / "model.gguf"
    fake_file.write_bytes(b"\x00" * 1024)  # 1 KiB dummy content

    fake_resolved = MagicMock()
    fake_resolved.repo_id = "acme/test-model"
    fake_resolved.revision = "sha256:" + "a" * 64
    fake_resolved.commit_sha = "c" * 40
    fake_resolved.quantization = None
    fake_resolved.format = "gguf"
    fake_resolved.filename = "model.gguf"

    with (
        patch("hfl.hub.resolver.resolve", return_value=fake_resolved) as m_resolve,
        patch("hfl.hub.downloader.pull_model", return_value=fake_file) as m_pull,
        patch(
            "hfl.hub.license_checker.check_model_license",
            return_value=_permissive_license(),
        ) as m_lic,
    ):
        yield m_resolve, m_pull, fake_file, m_lic


def _parse_ndjson(body: bytes | str) -> list[dict]:
    """Split NDJSON body into a list of parsed events."""
    text = body if isinstance(body, str) else body.decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestPullStreaming:
    def test_happy_path_emits_ollama_sequence(self, client, mock_pull):
        """The canonical Ollama status sequence must be emitted in order."""
        response = client.post(
            "/api/pull",
            json={"model": "acme/test-model", "stream": True},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")

        events = _parse_ndjson(response.content)
        statuses = [e["status"] for e in events]

        # Minimum required sequence (in order). Additional
        # ``downloading`` heartbeats may appear in between.
        assert statuses[0] == "pulling manifest"
        assert "downloading" in statuses
        assert statuses[-3:] == [
            "verifying sha256 digest",
            "writing manifest",
            "success",
        ]

    def test_downloading_events_carry_digest_and_counters(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": True})
        events = _parse_ndjson(response.content)
        downloads = [e for e in events if e["status"] == "downloading"]
        assert downloads, "Must emit at least one downloading event"

        for event in downloads:
            # Contract fields Open WebUI reads
            assert set(event.keys()) >= {"status", "digest", "total", "completed"}
            assert event["digest"].startswith("sha")
            assert isinstance(event["total"], int)
            assert isinstance(event["completed"], int)

        # Final "downloading" reports the actual on-disk bytes.
        final_dl = downloads[-1]
        assert final_dl["total"] == 1024  # 1 KiB dummy file
        assert final_dl["completed"] == final_dl["total"]

    def test_resolver_failure_emits_error_event(self, client, tmp_path):
        with patch(
            "hfl.hub.resolver.resolve",
            side_effect=ValueError("Model not found: ghost/phantom"),
        ):
            response = client.post("/api/pull", json={"model": "ghost/phantom", "stream": True})

        assert response.status_code == 200
        events = _parse_ndjson(response.content)
        statuses = [e["status"] for e in events]

        # Sequence starts with manifest, then jumps straight to error.
        assert statuses[0] == "pulling manifest"
        assert "error" in statuses
        error_event = [e for e in events if e["status"] == "error"][0]
        assert "ghost/phantom" in error_event["error"]

    def test_download_failure_emits_error_event(self, client, tmp_path):
        """If the download itself raises, the error line follows the
        opening downloading heartbeat."""
        fake_resolved = MagicMock()
        fake_resolved.repo_id = "acme/broken"
        fake_resolved.revision = None
        fake_resolved.format = "gguf"

        with (
            patch("hfl.hub.resolver.resolve", return_value=fake_resolved),
            patch(
                "hfl.hub.license_checker.check_model_license",
                return_value=_permissive_license(),
            ),
            patch(
                "hfl.hub.downloader.pull_model",
                side_effect=OSError("disk full"),
            ),
        ):
            response = client.post("/api/pull", json={"model": "acme/broken", "stream": True})

        events = _parse_ndjson(response.content)
        assert any(e["status"] == "error" for e in events)
        error = next(e for e in events if e["status"] == "error")
        assert "disk full" in error["error"]


class TestPullNonStreaming:
    def test_non_stream_returns_single_json_success(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 200
        # Response is pure JSON (not NDJSON).
        body = response.json()
        assert body["status"] == "success"
        # Duration is attached so scripts can log it.
        assert "_duration_seconds" in body

    def test_non_stream_returns_500_on_error(self, client):
        with patch("hfl.hub.resolver.resolve", side_effect=RuntimeError("boom")):
            response = client.post(
                "/api/pull",
                json={"model": "x/y", "stream": False},
            )
        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert "boom" in body["error"]


class TestPullValidation:
    def test_empty_model_rejected(self, client):
        response = client.post("/api/pull", json={"model": "", "stream": False})
        assert response.status_code == 422

    def test_oversize_model_rejected(self, client):
        response = client.post("/api/pull", json={"model": "x" * 1024, "stream": False})
        assert response.status_code == 422

    def test_insecure_flag_accepted_but_ignored(self, client, mock_pull):
        """``insecure`` is part of Ollama's schema — we accept it for
        compatibility but it's a no-op (hf_hub downloads are always
        HTTPS)."""
        response = client.post(
            "/api/pull",
            json={"model": "acme/test-model", "insecure": True, "stream": False},
        )
        assert response.status_code == 200


class TestPullContractForOpenWebUI:
    """Black-box parity check: the event stream must be parseable by
    a naive NDJSON consumer that just iterates lines and reads the
    ``status`` field."""

    def test_every_line_is_parseable_json(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": True})
        for line in response.content.decode().splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            assert "status" in parsed

    @pytest.mark.asyncio
    async def test_iter_pull_events_forwards_quantization_to_resolver(self, mock_pull):
        """``iter_pull_events(quantization=X)`` must pass X to ``resolve`` — the
        channel smart-pull relies on to honour the variant it planned for the
        host's memory budget (otherwise the resolver re-picks its own default)."""
        from hfl.api.routes_pull import iter_pull_events

        m_resolve, _m_pull, _f, _lic = mock_pull
        async for _line in iter_pull_events("acme/test-model", quantization="q3_k_m"):
            pass

        assert m_resolve.call_args is not None
        args, kwargs = m_resolve.call_args
        passed = kwargs.get("quantization", args[1] if len(args) > 1 else None)
        assert passed == "q3_k_m"

    @pytest.mark.asyncio
    async def test_iter_pull_events_defaults_quantization_to_none(self, mock_pull):
        """Without an explicit quant, the resolver still receives None and
        applies its own extraction/priority — no behaviour regression."""
        from hfl.api.routes_pull import iter_pull_events

        m_resolve, _m_pull, _f, _lic = mock_pull
        async for _line in iter_pull_events("acme/test-model"):
            pass

        args, kwargs = m_resolve.call_args
        passed = kwargs.get("quantization", args[1] if len(args) > 1 else None)
        assert passed is None


def _make_license(risk, license_id="some-license"):
    from hfl.hub.license_checker import LicenseInfo

    return LicenseInfo(
        license_id=license_id,
        license_name=license_id.title(),
        risk=risk,
        restrictions=[],
        url=f"https://huggingface.co/acme/{license_id}",
        gated=False,
    )


class TestPullOwnerGuard:
    """``pull`` is an owner operation: remote API users must be refused
    (403) unless the owner opted into remote administration."""

    def test_remote_caller_is_forbidden(self, temp_config, mock_pull):
        remote = TestClient(app, client=REMOTE_PEER)
        response = remote.post("/api/pull", json={"model": "acme/test-model", "stream": True})
        assert response.status_code == 403
        detail = response.json()["detail"]
        assert detail["code"] == "remote_admin_forbidden"
        # Refused before any download work happened.
        _m_resolve, m_pull, _f, _lic = mock_pull
        m_pull.assert_not_called()

    def test_local_caller_is_allowed(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_remote_allowed_when_owner_opts_in(self, temp_config, mock_pull):
        temp_config.allow_remote_pull = True
        remote = TestClient(app, client=REMOTE_PEER)
        response = remote.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 200
        assert response.json()["status"] == "success"


class TestPullLicensePolicy:
    """The owner's ``license_policy`` decides, without a human in the
    loop, whether a non-interactive pull may proceed."""

    def _run(self, client, risk, license_id="cc-by-nc-4.0", stream=True):
        fake_resolved = MagicMock()
        fake_resolved.repo_id = "acme/test-model"
        fake_resolved.revision = None
        fake_resolved.commit_sha = None
        fake_resolved.quantization = None
        fake_resolved.format = "gguf"
        with (
            patch("hfl.hub.resolver.resolve", return_value=fake_resolved),
            patch(
                "hfl.hub.license_checker.check_model_license",
                return_value=_make_license(risk, license_id),
            ),
            patch("hfl.hub.downloader.pull_model") as m_pull,
        ):
            self._m_pull = m_pull
            return client.post("/api/pull", json={"model": "acme/test-model", "stream": stream})

    def test_non_permissive_blocked_under_default_policy_stream(self, client, temp_config):
        """Default policy is 'permissive' — a non-commercial license is
        refused with a license_not_accepted error event, no download."""
        from hfl.hub.license_checker import LicenseRisk

        response = self._run(client, LicenseRisk.NON_COMMERCIAL)
        events = _parse_ndjson(response.content)
        err = next(e for e in events if e["status"] == "error")
        assert err["code"] == "license_not_accepted"
        assert err["risk"] == "non_commercial"
        # Never reached the download.
        self._m_pull.assert_not_called()
        # And no success event was emitted.
        assert not any(e["status"] == "success" for e in events)

    def test_non_permissive_blocked_returns_403_non_stream(self, client, temp_config):
        from hfl.hub.license_checker import LicenseRisk

        response = self._run(client, LicenseRisk.RESTRICTED, stream=False)
        assert response.status_code == 403
        assert response.json()["code"] == "license_not_accepted"

    def test_conditional_allowed_when_policy_widened(self, client, temp_config, mock_pull):
        """With policy='conditional', a Llama-class (CONDITIONAL) license
        proceeds automatically. Uses ``mock_pull`` for a real download
        path so the success + registration steps run."""
        from hfl.hub.license_checker import LicenseRisk

        temp_config.license_policy = "conditional"
        with patch(
            "hfl.hub.license_checker.check_model_license",
            return_value=_make_license(LicenseRisk.CONDITIONAL, "llama3.1"),
        ):
            response = client.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_all_policy_allows_restricted(self, client, temp_config, mock_pull):
        from hfl.hub.license_checker import LicenseRisk

        temp_config.license_policy = "all"
        with patch(
            "hfl.hub.license_checker.check_model_license",
            return_value=_make_license(LicenseRisk.RESTRICTED, "proprietary"),
        ):
            response = client.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_permissive_emits_verifying_license_event(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": True})
        statuses = [e["status"] for e in _parse_ndjson(response.content)]
        assert "verifying license" in statuses

    def test_license_classification_failure_fails_closed(self, client, temp_config):
        """If the Hub lookup raises, the license is treated as UNKNOWN and
        the default policy refuses it (fail closed)."""
        fake_resolved = MagicMock()
        fake_resolved.repo_id = "acme/test-model"
        fake_resolved.revision = None
        fake_resolved.format = "gguf"
        with (
            patch("hfl.hub.resolver.resolve", return_value=fake_resolved),
            patch(
                "hfl.hub.license_checker.check_model_license",
                side_effect=RuntimeError("hub down"),
            ),
            patch("hfl.hub.downloader.pull_model") as m_pull,
        ):
            response = client.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 403
        assert response.json()["code"] == "license_not_accepted"
        m_pull.assert_not_called()


class TestPullTraceability:
    """A successful server pull must register the model with its license
    and log provenance — the gap that let API pulls slip through
    untracked."""

    def test_successful_pull_registers_manifest_with_license(self, client, mock_pull):
        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 200

        from hfl.models.registry import ModelRegistry

        manifest = ModelRegistry().get("test-model")
        assert manifest is not None
        assert manifest.repo_id == "acme/test-model"
        assert manifest.license == "apache-2.0"
        assert manifest.license_accepted_at is not None

    def test_successful_pull_logs_provenance(self, client, mock_pull, monkeypatch):
        # Reset the cached provenance singleton so it re-binds to the
        # temp_config home for this test.
        import hfl.models.provenance as prov

        monkeypatch.setattr(prov, "_provenance_log", None)

        response = client.post("/api/pull", json={"model": "acme/test-model", "stream": False})
        assert response.status_code == 200

        history = prov.get_provenance_log().get_history("acme/test-model")
        assert history, "expected a provenance record for the pulled repo"
        record = history[-1]
        assert record["license_accepted"] is True
        assert record["original_license"] == "apache-2.0"
        assert "license policy" in record["notes"]

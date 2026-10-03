# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A pull downloads, checks and reads the license of the commit it records;
the license gate classifies by name, not by substring; the license panel
shows Hub text as text."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hfl.hub import downloader
from hfl.hub.license_checker import (
    LicenseInfo,
    LicenseRisk,
    check_model_license,
    require_user_acceptance,
)
from hfl.hub.resolver import ResolvedModel

SHA = "a" * 40

# -- finding 3: the recorded commit is the downloaded one ----------------


@pytest.mark.hub_sizes  # its own fake Hub: the sha256 lookup is under test
def test_a_gguf_pull_downloads_and_checks_every_file_at_the_resolved_commit(
    temp_config, monkeypatch
):
    revisions: list[tuple[str, str | None]] = []
    monkeypatch.setattr(downloader, "ensure_auth", lambda repo: None)
    monkeypatch.setattr(downloader, "_rate_limit", lambda: None)

    def fetch(repo_id, filename, revision, local_dir, token):
        revisions.append((filename, revision))
        (local_dir / filename).write_bytes(b"x")
        return local_dir / filename

    monkeypatch.setattr(downloader, "_download_file", fetch)
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(siblings=[])
    with patch("huggingface_hub.HfApi", return_value=api):
        downloader.pull_model(
            ResolvedModel(
                repo_id="o/r",
                revision="main",
                commit_sha=SHA,
                filename="m-00001-of-00002.gguf",
                format="gguf",
                parts=["m-00002-of-00002.gguf"],
                projector="mmproj-F16.gguf",
            )
        )
    assert revisions and all(rev == SHA for _, rev in revisions)
    assert api.model_info.call_args.kwargs["revision"] == SHA  # sha256 of that commit


def test_a_snapshot_pull_is_at_the_resolved_commit(temp_config, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(downloader, "ensure_auth", lambda repo: None)
    monkeypatch.setattr(downloader, "_rate_limit", lambda: None)
    monkeypatch.setattr(downloader, "_verify_downloads", lambda *a, **k: None)
    monkeypatch.setattr(downloader, "snapshot_download", lambda **kw: calls.append(kw) or ".")
    downloader.pull_model(
        ResolvedModel(repo_id="o/r", revision="main", commit_sha=SHA, format="safetensors")
    )
    assert calls[0]["revision"] == SHA
    # Without a commit (an older caller), the ref asked for is still used.
    downloader.pull_model(ResolvedModel(repo_id="o/r", revision="v2", format="safetensors"))
    assert calls[1]["revision"] == "v2"


def test_a_redownload_after_a_bad_sha256_is_at_the_same_commit(tmp_path, monkeypatch):
    import hashlib

    (tmp_path / "m.gguf").write_bytes(b"bad")
    monkeypatch.setattr(
        downloader,
        "_hub_sha256",
        lambda resolved, token: {"m.gguf": hashlib.sha256(b"good").hexdigest()},
    )
    seen: list[str | None] = []

    def fetch(repo_id, filename, revision, local_dir, token):
        seen.append(revision)
        (local_dir / filename).write_bytes(b"good")
        return local_dir / filename

    monkeypatch.setattr(downloader, "_download_file", fetch)
    resolved = SimpleNamespace(repo_id="o/r", revision="main", commit_sha=SHA)
    downloader._verify_downloads(resolved, tmp_path, None, ["m.gguf"])
    assert seen == [SHA]


def _card(license_id: str, license_name: str | None = None, link: str | None = None):
    info = MagicMock()
    info.card_data = MagicMock()
    info.card_data.license = license_id
    info.card_data.license_name = license_name
    info.card_data.license_link = link
    info.gated = False
    info.tags = []
    return info


def test_the_license_is_read_at_the_pinned_revision():
    with patch("hfl.hub.license_checker.HfApi") as api_class:
        api_class.return_value.model_info.return_value = _card("mit")
        check_model_license("o/r", revision=SHA)
    assert api_class.return_value.model_info.call_args.kwargs["revision"] == SHA


# -- finding 4: no PERMISSIVE by substring ---------------------------------


def _classify(name: str) -> LicenseInfo:
    with patch("hfl.hub.license_checker.HfApi") as api_class:
        api_class.return_value.model_info.return_value = _card("other", name)
        return check_model_license("o/r")


@pytest.mark.parametrize(
    "name", ["acme-limited-noncommercial", "a", "t", "ap", "Submit-Only", "mitigated terms"]
)
def test_a_name_that_only_contains_a_permissive_license_is_not_permissive(name):
    assert _classify(name).risk != LicenseRisk.PERMISSIVE


@pytest.mark.parametrize(
    "name, risk, family",
    [
        ("llama3.1-community", LicenseRisk.CONDITIONAL, "llama3.1"),
        ("qwen2", LicenseRisk.CONDITIONAL, "qwen"),
        ("Gemma Terms", LicenseRisk.CONDITIONAL, "gemma"),
        ("CC BY-NC 4.0", LicenseRisk.NON_COMMERCIAL, "cc-by-nc-4.0"),
        ("Apache 2.0", LicenseRisk.PERMISSIVE, None),
        ("MIT", LicenseRisk.PERMISSIVE, None),
    ],
)
def test_the_variants_it_meant_to_accept_still_classify(name, risk, family):
    from hfl.hub.license_checker import LICENSE_RESTRICTIONS

    info = _classify(name)
    assert info.risk == risk
    assert info.restrictions == LICENSE_RESTRICTIONS.get(family, []) if family else True


def test_restrictions_follow_the_family_not_a_substring():
    from hfl.hub.license_checker import LICENSE_RESTRICTIONS

    with patch("hfl.hub.license_checker.HfApi") as api_class:
        api_class.return_value.model_info.return_value = _card("creativeml-openrail-m")
        info = check_model_license("o/r")
    assert info.restrictions == LICENSE_RESTRICTIONS["openrail"]
    assert _classify("a").restrictions == []  # "a" is inside "llama2": not its terms


# -- finding 6: Hub text in the panel is text -----------------------------

EVIL = (
    "x\x1b[2K\x1b[1A\x1b]8;;https://evil.example\x1b\\here\x1b]8;;\x1b\\"
    "‮[/][bold green]PERMISSIVE[/] [link=https://evil.example]terms[/link]\x9b2J"
)


def _panel(info: LicenseInfo) -> str:
    from rich.console import Console

    sink = io.StringIO()
    console = Console(file=sink, width=400, force_terminal=False, color_system=None)
    with (
        patch("rich.console.Console", return_value=console),
        patch("typer.confirm", return_value=False),
    ):
        require_user_acceptance(info, "o/r")
    return sink.getvalue()


def test_the_panel_strips_control_characters_and_markup_from_hub_text():
    info = LicenseInfo(
        license_id=EVIL,
        license_name=EVIL,
        risk=LicenseRisk.NON_COMMERCIAL,
        restrictions=["non-commercial-only"],
        url="https://example.com/" + EVIL,
        gated=False,
    )
    out = _panel(info)
    assert "NON-COMMERCIAL LICENSE" in out
    for bad in ("\x1b", "\x9b", "‮", "\x07"):
        assert bad not in out
    # The card's markup is printed as text, not interpreted.
    assert "[bold green]PERMISSIVE[/]" in out
    assert "[link=https://evil.example]terms[/link]" in out


def test_the_permissive_line_shows_hub_text_as_text():
    info = LicenseInfo(
        license_id="[red]x[/]\x1b[2K",
        license_name="n",
        risk=LicenseRisk.PERMISSIVE,
        restrictions=[],
        url=None,
        gated=False,
    )
    out = _panel(info)
    assert "[red]x[/]" in out and "\x1b" not in out

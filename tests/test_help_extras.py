# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl help --extras`` lists every extra the package declares (a list kept
by hand missed seven: local audit A13), each described in both languages,
its packages read from the package's own metadata."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
DECLARED = sorted(PROJECT["optional-dependencies"])
USER_EXTRAS = sorted(set(DECLARED) - {"dev", "build"})


@pytest.fixture
def metadata_from_pyproject(monkeypatch):
    """The installed metadata as pyproject.toml declares it (an editable
    install's can lag behind until reinstalled)."""

    class Meta:
        @staticmethod
        def get_all(key):
            return DECLARED if key == "Provides-Extra" else []

    requires = [
        f"{spec}; extra == '{extra}'"
        for extra, specs in PROJECT["optional-dependencies"].items()
        for spec in specs
    ]
    monkeypatch.setattr(importlib.metadata, "metadata", lambda name: Meta())
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: requires)


@pytest.mark.parametrize("lang", ["en", "es"])
def test_every_extra_is_described(lang) -> None:
    extras = json.loads((ROOT / f"src/hfl/i18n/locales/{lang}.json").read_text())["help"]["extras"]
    missing = [e for e in USER_EXTRAS if not extras.get(e, {}).get("summary")]
    assert missing == []


def test_every_declared_extra_is_listed(metadata_from_pyproject) -> None:
    from hfl.cli.main import app

    result = CliRunner().invoke(app, ["help", "--extras"], env={"COLUMNS": "220"})
    assert result.exit_code == 0
    assert [e for e in USER_EXTRAS if f"hfl[{e}]" in result.stdout] == USER_EXTRAS


def test_packages_come_from_the_markers(metadata_from_pyproject) -> None:
    from hfl.cli.main import _declared_extras

    order, packages = _declared_extras()
    assert "dev" not in order and "build" not in order
    assert order[0] == "llama" and order[-1] == "all"
    assert packages["tts"] == PROJECT["optional-dependencies"]["tts"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Records written while 4096 was the manifest default load as "auto".

Until 2026-03-18 ``ModelManifest.context_length`` defaulted to 4096, and
every pull saved that number. The server and ``hfl run`` honour a
recorded context before auto-sizing, so those models kept loading with
4096 tokens forever — a 256K-context coder could not take Claude Code's
20K-token prompt. A 4096 saved before that date is the old default, not
a choice; one saved later came from an explicit ``num_ctx`` and stays.
"""

from __future__ import annotations

import pytest

from hfl.models.manifest import ModelManifest


def _record(context_length, created_at):
    return {
        "name": "m",
        "repo_id": "org/m",
        "local_path": "/nowhere/m.gguf",
        "format": "gguf",
        "context_length": context_length,
        "created_at": created_at,
    }


@pytest.mark.parametrize(
    "created_at", ["2026-02-18T10:00:00", "2026-03-05T23:59:59.123456", "2026-03-17T23:59:59"]
)
def test_the_old_default_means_auto(created_at):
    assert ModelManifest.from_dict(_record(4096, created_at)).context_length == 0


@pytest.mark.parametrize(
    ("context_length", "created_at"),
    [
        (4096, "2026-03-18T00:00:00"),  # explicit num_ctx after the change
        (4096, "2026-09-01T12:00:00"),
        (8192, "2026-02-18T10:00:00"),  # never the default
        (0, "2026-02-18T10:00:00"),
    ],
)
def test_anything_else_is_kept(context_length, created_at):
    assert ModelManifest.from_dict(_record(context_length, created_at)).context_length == (
        context_length
    )


@pytest.mark.parametrize("created_at", [None, "", "not a date"])
def test_an_unreadable_date_keeps_the_value(created_at):
    assert ModelManifest.from_dict(_record(4096, created_at)).context_length == 4096


def test_the_server_then_auto_sizes(temp_config):
    from hfl.api.model_loader import _manifest_ctx

    legacy = ModelManifest.from_dict(_record(4096, "2026-02-19T09:00:00"))
    assert _manifest_ctx(legacy) == 0


def test_the_registry_rehydrates_through_the_same_path(temp_config):
    import json

    from hfl.models.registry import ModelRegistry

    temp_config.registry_path.write_text(json.dumps([_record(4096, "2026-02-19T09:00:00")]))
    assert ModelRegistry().get("m").context_length == 0

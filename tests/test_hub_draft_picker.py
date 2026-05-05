# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for ``hfl/hub/draft_picker.py`` — V4 F5."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import MagicMock

from hfl.hub.draft_picker import DraftPick, pick_draft_for


@dataclass
class _FakeCardData:
    license: str | None = None


@dataclass
class _FakeModelInfo:
    id: str
    likes: int = 0
    downloads: int = 0
    last_modified: datetime | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    gated: bool = False
    tags: list[str] = field(default_factory=list)
    card_data: _FakeCardData | None = None


def _api_with(*infos: _FakeModelInfo):
    api = MagicMock()
    api.list_models = MagicMock(return_value=iter(list(infos)))
    return api


class TestPickDraftFor:
    def test_picks_smaller_llama_when_available(self):
        api = _api_with(
            _FakeModelInfo(
                id="meta-llama/Llama-3.2-1B-Instruct",
                likes=2000,
                tags=["llama", "instruct"],
            ),
            _FakeModelInfo(
                id="meta-llama/Llama-3.1-8B-Instruct",
                likes=10000,
                tags=["llama", "instruct"],
            ),
        )
        pick = pick_draft_for("meta-llama/Llama-3.1-70B-Instruct", api=api)
        assert pick is not None
        # Both candidates are < 70B × 0.25 = 17.5B, so 1B and 8B are
        # both eligible. Smaller wins on the score.
        assert "1B" in pick.repo_id

    def test_excludes_target_repo_itself(self):
        api = _api_with(
            _FakeModelInfo(
                id="meta-llama/Llama-3.1-70B-Instruct",
                likes=10000,
                tags=["llama"],
            ),
            _FakeModelInfo(
                id="meta-llama/Llama-3.2-1B-Instruct",
                likes=2000,
                tags=["llama"],
            ),
        )
        pick = pick_draft_for("meta-llama/Llama-3.1-70B-Instruct", api=api)
        assert pick is not None
        assert pick.repo_id != "meta-llama/Llama-3.1-70B-Instruct"

    def test_max_ratio_filters_out_too_large(self):
        api = _api_with(
            _FakeModelInfo(
                id="meta-llama/Llama-3.1-13B-Instruct",
                likes=5000,
                tags=["llama"],
            ),
        )
        # Strict ratio: 13B / 8B = 1.625 ratio → above 0.25 → reject.
        pick = pick_draft_for(
            "meta-llama/Llama-3.1-8B-Instruct",
            api=api,
            max_ratio=0.25,
        )
        # Hub yielded only the 13B; falls back to canonical Llama draft.
        assert pick is not None
        assert "1B" in pick.repo_id  # the canonical Llama default
        assert "no smaller fork" in pick.rationale

    def test_falls_back_to_canonical_when_hub_empty(self):
        api = _api_with()  # nothing matched

        pick = pick_draft_for("meta-llama/Llama-3.1-70B-Instruct", api=api)
        assert pick is not None
        assert "meta-llama" in pick.repo_id
        assert "no smaller fork" in pick.rationale

    def test_unknown_family_returns_none(self):
        api = _api_with()
        # Repo id with no detectable family AND no parameter signal.
        pick = pick_draft_for("anthropic/notamodel", api=api)
        assert pick is None

    def test_quant_preference_rewards_quantised_forks(self):
        """When two siblings have similar params and likes, the
        quantised one should win."""
        api = _api_with(
            _FakeModelInfo(
                id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                likes=500,
                tags=["llama", "gguf"],
            ),
            _FakeModelInfo(
                id="meta-llama/Llama-3.2-1B-Instruct",
                likes=500,
                tags=["llama"],
            ),
        )
        pick = pick_draft_for("meta-llama/Llama-3.1-70B-Instruct", api=api)
        assert pick is not None
        # The GGUF fork carries the +3 quant bonus.
        assert "GGUF" in pick.repo_id


class TestDraftPickShape:
    def test_dataclass_fields_present(self):
        api = _api_with(
            _FakeModelInfo(
                id="meta-llama/Llama-3.2-1B-Instruct",
                likes=2000,
                tags=["llama"],
            ),
        )
        pick = pick_draft_for("meta-llama/Llama-3.1-70B-Instruct", api=api)
        assert isinstance(pick, DraftPick)
        assert pick.repo_id
        # rationale is mandatory for the API output to be intelligible.
        assert pick.rationale

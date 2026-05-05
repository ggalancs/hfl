# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Unit tests for ``hfl/hub/discovery.py`` — V4.

The HF Hub never gets touched in these tests; ``HfApi.list_models``
is replaced with a fake that returns hand-built ``ModelInfo``-like
objects, so we can exercise the family/quant/multimodal heuristics
deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hfl.hub.discovery import (
    DiscoveryCache,
    DiscoveryQuery,
    _entry_from_model_info,
    _family_for,
    _is_multimodal,
    _parameter_estimate_b,
    _quantization_for,
    format_size_human,
    search_hub,
)


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


# --- family / quant / multimodal classifiers --------------------------------


class TestClassifiers:
    @pytest.mark.parametrize(
        "repo_id,tags,expected",
        [
            ("meta-llama/Llama-3.1-8B-Instruct", ["llama"], "llama"),
            ("Qwen/Qwen2.5-7B-Instruct", ["qwen2"], "qwen"),
            ("google/gemma-3-9b-it", ["gemma"], "gemma"),
            ("mistralai/Mistral-7B-v0.3", ["mistral"], "mistral"),
            ("mistralai/Mixtral-8x7B-Instruct-v0.1", ["mixtral"], "mixtral"),
            ("microsoft/phi-3-mini", ["phi3"], "phi"),
            ("anthropic/notamodel", [], None),  # no signal
        ],
    )
    def test_family_detection(self, repo_id, tags, expected):
        assert _family_for(repo_id, tags) == expected

    @pytest.mark.parametrize(
        "repo_id,tags,expected",
        [
            ("bartowski/Qwen2.5-7B-Instruct-GGUF", ["gguf"], "gguf"),
            ("mlx-community/Llama-3.1-8B-Instruct-4bit", ["mlx"], "mlx"),
            ("user/something-AWQ", [], "awq"),
            ("user/something-GPTQ", [], "gptq"),
            ("user/plain-safetensors", [], None),
        ],
    )
    def test_quantization_detection(self, repo_id, tags, expected):
        assert _quantization_for(repo_id, tags) == expected

    @pytest.mark.parametrize(
        "tags,pipeline_tag,expected",
        [
            (["vision", "llama"], None, True),
            (["multimodal"], None, True),
            (["text"], "image-to-text", True),
            (["vl"], None, True),
            (["llama", "instruct"], "text-generation", False),
        ],
    )
    def test_multimodal_detection(self, tags, pipeline_tag, expected):
        assert _is_multimodal(tags, pipeline_tag) == expected

    @pytest.mark.parametrize(
        "repo_id,tags,expected",
        [
            ("meta-llama/Llama-3.1-8B-Instruct", [], 8.0),
            ("Qwen/Qwen2.5-1.5B", [], 1.5),
            ("google/gemma-3-27b-it", [], 27.0),
            ("mistralai/Mistral-7B-v0.3", [], 7.0),
            # Mixtral name carries 8x7B; we keep the LAST B-suffixed
            # number, which is "7" here.
            ("mistralai/Mixtral-8x7B-Instruct", [], 7.0),
            ("user/no-signal", [], None),
        ],
    )
    def test_parameter_estimate(self, repo_id, tags, expected):
        assert _parameter_estimate_b(repo_id, tags) == expected


# --- entry conversion -------------------------------------------------------


class TestEntryConversion:
    def test_full_metadata_round_trip(self):
        info = _FakeModelInfo(
            id="meta-llama/Llama-3.1-8B-Instruct",
            likes=1234,
            downloads=987_654,
            last_modified=datetime(2026, 1, 15),
            pipeline_tag="text-generation",
            library_name="transformers",
            gated=True,
            tags=["llama", "instruct"],
            card_data=_FakeCardData(license="apache-2.0"),
        )
        entry = _entry_from_model_info(info)

        assert entry.repo_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert entry.likes == 1234
        assert entry.downloads == 987_654
        assert entry.last_modified == "2026-01-15T00:00:00"
        assert entry.pipeline_tag == "text-generation"
        assert entry.library == "transformers"
        assert entry.license == "apache-2.0"
        assert entry.gated is True
        assert entry.family == "llama"
        assert entry.parameter_estimate_b == 8.0

    def test_missing_card_data_is_handled(self):
        info = _FakeModelInfo(id="user/x", card_data=None)
        entry = _entry_from_model_info(info)
        assert entry.license is None

    def test_multimodal_tag_is_appended_when_inferred(self):
        info = _FakeModelInfo(
            id="user/llava-vision",
            tags=["vision"],
            pipeline_tag=None,
        )
        entry = _entry_from_model_info(info)
        assert "multimodal" in entry.tags


# --- search_hub end-to-end --------------------------------------------------


class TestSearchHub:
    def test_results_filtered_by_min_likes(self):
        api = MagicMock()
        api.list_models.return_value = iter(
            [
                _FakeModelInfo(id="user/popular", likes=500, tags=["llama"]),
                _FakeModelInfo(id="user/unknown", likes=2, tags=["llama"]),
            ]
        )
        results = search_hub(DiscoveryQuery(min_likes=100), api=api)
        assert [r.repo_id for r in results] == ["user/popular"]

    def test_family_filter(self):
        api = MagicMock()
        api.list_models.return_value = iter(
            [
                _FakeModelInfo(id="meta-llama/Llama-3-8B", tags=["llama"]),
                _FakeModelInfo(id="Qwen/Qwen-7B", tags=["qwen"]),
            ]
        )
        results = search_hub(DiscoveryQuery(family="qwen"), api=api)
        assert [r.repo_id for r in results] == ["Qwen/Qwen-7B"]

    def test_quantization_filter(self):
        api = MagicMock()
        api.list_models.return_value = iter(
            [
                _FakeModelInfo(id="mlx-community/Llama-4bit", tags=["mlx"]),
                _FakeModelInfo(id="bartowski/Llama-GGUF", tags=["gguf"]),
            ]
        )
        results = search_hub(DiscoveryQuery(quantization="gguf"), api=api)
        assert [r.repo_id for r in results] == ["bartowski/Llama-GGUF"]

    def test_multimodal_only(self):
        api = MagicMock()
        api.list_models.return_value = iter(
            [
                _FakeModelInfo(id="user/llava", tags=["vision", "llama"]),
                _FakeModelInfo(id="user/text-only", tags=["llama"]),
            ]
        )
        results = search_hub(DiscoveryQuery(multimodal=True), api=api)
        assert [r.repo_id for r in results] == ["user/llava"]

    def test_gated_filter_includes_only_gated(self):
        api = MagicMock()
        api.list_models.return_value = iter(
            [
                _FakeModelInfo(id="meta-llama/Llama-3", gated=True),
                _FakeModelInfo(id="user/open-fork", gated=False),
            ]
        )
        results = search_hub(DiscoveryQuery(gated=True), api=api)
        assert [r.repo_id for r in results] == ["meta-llama/Llama-3"]

    def test_page_size_caps_output(self):
        api = MagicMock()
        api.list_models.return_value = iter(
            _FakeModelInfo(id=f"user/m{i}", likes=10) for i in range(50)
        )
        results = search_hub(DiscoveryQuery(page_size=5), api=api)
        assert len(results) == 5

    def test_license_filter_case_insensitive(self):
        api = MagicMock()
        api.list_models.return_value = iter(
            [
                _FakeModelInfo(id="user/a", card_data=_FakeCardData(license="Apache-2.0")),
                _FakeModelInfo(id="user/b", card_data=_FakeCardData(license="MIT")),
            ]
        )
        results = search_hub(DiscoveryQuery(license="apache-2.0"), api=api)
        assert [r.repo_id for r in results] == ["user/a"]


# --- DiscoveryCache ---------------------------------------------------------


class TestDiscoveryCache:
    def test_round_trip(self, tmp_path):
        from hfl.hub.discovery import DiscoveryEntry

        cache = DiscoveryCache(tmp_path / "cache.json", ttl_seconds=300)
        query = DiscoveryQuery(q="qwen")
        entries = [
            DiscoveryEntry(
                repo_id="Qwen/X",
                likes=10,
                downloads=100,
                last_modified=None,
                pipeline_tag=None,
                library=None,
                license=None,
                gated=False,
            )
        ]

        assert cache.get(query) is None
        cache.put(query, entries)
        roundtrip = cache.get(query)
        assert roundtrip is not None
        assert [e.repo_id for e in roundtrip] == ["Qwen/X"]

    def test_expired_entries_return_none(self, tmp_path):
        from hfl.hub.discovery import DiscoveryEntry

        cache = DiscoveryCache(tmp_path / "cache.json", ttl_seconds=0)
        query = DiscoveryQuery(q="qwen")
        entries = [
            DiscoveryEntry(
                repo_id="Qwen/X",
                likes=10,
                downloads=100,
                last_modified=None,
                pipeline_tag=None,
                library=None,
                license=None,
                gated=False,
            )
        ]
        cache.put(query, entries)

        # ttl=0 means everything reads as stale on the next get.
        assert cache.get(query) is None

    def test_corrupt_cache_file_does_not_crash(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("{ this is not json")

        cache = DiscoveryCache(path)
        # Should treat corrupt cache as empty and not raise.
        assert cache.get(DiscoveryQuery()) is None

    def test_cache_evicts_when_oversized(self, tmp_path):
        from hfl.hub.discovery import DiscoveryEntry

        cache = DiscoveryCache(tmp_path / "cache.json")
        # 40 distinct queries, then verify we capped near 32.
        for i in range(40):
            cache.put(
                DiscoveryQuery(q=f"q-{i}"),
                [
                    DiscoveryEntry(
                        repo_id=f"u/m{i}",
                        likes=0,
                        downloads=0,
                        last_modified=None,
                        pipeline_tag=None,
                        library=None,
                        license=None,
                        gated=False,
                    )
                ],
            )
        import json

        data = json.loads(Path(tmp_path / "cache.json").read_text())
        assert len(data) <= 33  # 32 retained + the freshest just inserted


# --- Misc -------------------------------------------------------------------


class TestFormatSizeHuman:
    @pytest.mark.parametrize(
        "b,expected",
        [
            (None, "-"),
            (1.0, "1B"),
            (1.5, "1.5B"),
            (8.0, "8B"),
            (70.0, "70B"),
            (180.0, "180B"),
        ],
    )
    def test_render(self, b, expected):
        assert format_size_human(b) == expected

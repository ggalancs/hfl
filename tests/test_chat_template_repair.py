# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the chat_template repair helper used after ``hfl pull`` on
MLX repos that ship without a chat template."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hfl.hub.chat_template_repair import (
    _heuristic_base_repo,
    ensure_chat_template,
    has_chat_template,
)

# --- has_chat_template --------------------------------------------------------


class TestHasChatTemplate:
    def test_true_when_jinja_file_present(self, tmp_path: Path):
        (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
        assert has_chat_template(tmp_path) is True

    def test_true_when_tokenizer_config_has_template(self, tmp_path: Path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GemmaTokenizer", "chat_template": "tmpl"})
        )
        assert has_chat_template(tmp_path) is True

    def test_false_when_neither(self, tmp_path: Path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GemmaTokenizer"})
        )
        assert has_chat_template(tmp_path) is False

    def test_false_when_tokenizer_config_missing(self, tmp_path: Path):
        assert has_chat_template(tmp_path) is False

    def test_false_when_tokenizer_config_invalid(self, tmp_path: Path):
        (tmp_path / "tokenizer_config.json").write_text("{ not json")
        assert has_chat_template(tmp_path) is False

    def test_empty_template_counts_as_missing(self, tmp_path: Path):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": ""}))
        assert has_chat_template(tmp_path) is False


# --- Heuristic base repo resolution ------------------------------------------


class TestHeuristicBaseRepo:
    @pytest.mark.parametrize(
        "repo_id,expected",
        [
            ("mlx-community/gemma-4-31b-it-4bit", "google/gemma-4-31b-it"),
            ("lmstudio-community/gemma-4-31B-it-MLX-4bit", "google/gemma-4-31B-it"),
            ("mlx-community/Llama-3.3-70B-Instruct-4bit", "meta-llama/Llama-3.3-70B-Instruct"),
            ("mlx-community/Qwen2.5-32B-Instruct-8bit", "Qwen/Qwen2.5-32B-Instruct"),
            ("mlx-community/Mistral-7B-Instruct-v0.3-bf16", "mistralai/Mistral-7B-Instruct-v0.3"),
        ],
    )
    def test_resolves_known_patterns(self, repo_id, expected):
        assert _heuristic_base_repo(repo_id) == expected

    def test_unknown_org_returns_none(self):
        assert _heuristic_base_repo("acme/random-model-4bit") is None

    def test_unstripped_name_returns_none(self):
        # No quantisation suffix at all — can't know the base.
        assert _heuristic_base_repo("mlx-community/gemma-4-31b-it") is None

    def test_unknown_family_returns_none(self):
        # Org is known but family isn't in our heuristic table.
        assert _heuristic_base_repo("mlx-community/some-random-model-4bit") is None

    def test_malformed_repo_id_returns_none(self):
        assert _heuristic_base_repo("just-a-name-4bit") is None


# --- ensure_chat_template end-to-end -----------------------------------------


class TestEnsureChatTemplate:
    def test_noop_when_template_already_present(self, tmp_path: Path):
        (tmp_path / "chat_template.jinja").write_text("tmpl")
        # Should not attempt any network call.
        with patch("huggingface_hub.HfApi") as mock_api:
            assert ensure_chat_template(tmp_path, "repo/id") is True
        # Noop path should skip both HfApi construction and downloads.
        mock_api.assert_not_called()

    def test_fetches_jinja_from_base_repo(self, tmp_path: Path, monkeypatch):
        # Tokenizer config exists but has no chat_template.
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GemmaTokenizer"})
        )

        # ModelCard resolution yields the base repo.
        fake_info = MagicMock()
        fake_info.card_data = MagicMock(base_model="google/gemma-4-31b-it")
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.return_value = fake_info

        with patch("huggingface_hub.HfApi", return_value=mock_api_instance):
            # Fetch succeeds and the hub lib returns a path to a cached
            # file; we stage that file in tmp_path for realism.
            cached = tmp_path / "_cache_chat_template.jinja"
            cached.write_text("{%- macro greet -%}hi{%- endmacro -%}")

            def fake_download(repo, filename):
                assert repo == "google/gemma-4-31b-it"
                assert filename == "chat_template.jinja"
                return str(cached)

            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=fake_download,
                create=True,
            ):
                assert ensure_chat_template(tmp_path, "mlx-community/gemma-4-31b-it-4bit") is True

        assert (tmp_path / "chat_template.jinja").read_text().startswith("{%- macro greet")

    def test_falls_back_to_heuristic_when_model_card_lacks_base(self, tmp_path: Path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GemmaTokenizer"})
        )

        # ModelCard returns info without card_data.base_model.
        fake_info = MagicMock()
        fake_info.card_data = MagicMock(base_model=None)
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.return_value = fake_info

        captured: dict = {}

        def fake_download(repo, filename):
            captured["repo"] = repo
            captured["filename"] = filename
            dst = tmp_path / "_cache.jinja"
            dst.write_text("heuristic-template")
            return str(dst)

        with (
            patch(
                "huggingface_hub.HfApi",
                return_value=mock_api_instance,
                create=True,
            ),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=fake_download,
                create=True,
            ),
        ):
            ok = ensure_chat_template(tmp_path, "mlx-community/gemma-4-31b-it-4bit")

        assert ok is True
        # The heuristic must have mapped mlx-community/...-4bit -> google/...
        assert captured["repo"] == "google/gemma-4-31b-it"

    def test_merges_template_into_tokenizer_config_when_no_jinja_file(self, tmp_path: Path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GemmaTokenizer"})
        )

        fake_info = MagicMock()
        fake_info.card_data = MagicMock(base_model="google/gemma-4-31b-it")
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.return_value = fake_info

        base_cfg = tmp_path / "_base_cfg.json"
        base_cfg.write_text(json.dumps({"chat_template": "RECOVERED"}))

        call_log: list[tuple[str, str]] = []

        def fake_download(repo, filename):
            call_log.append((repo, filename))
            if filename == "chat_template.jinja":
                # Simulate HTTP 404 for the jinja file.
                raise FileNotFoundError("not present in base repo")
            if filename == "tokenizer_config.json":
                return str(base_cfg)
            raise AssertionError(f"unexpected filename {filename}")

        with (
            patch(
                "huggingface_hub.HfApi",
                return_value=mock_api_instance,
                create=True,
            ),
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=fake_download,
                create=True,
            ),
        ):
            ok = ensure_chat_template(tmp_path, "mlx-community/gemma-4-31b-it-4bit")

        assert ok is True
        # Both attempts were made, in order.
        assert ("google/gemma-4-31b-it", "chat_template.jinja") in call_log
        assert ("google/gemma-4-31b-it", "tokenizer_config.json") in call_log
        # The merged template is now in the local tokenizer_config.
        local_cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert local_cfg["chat_template"] == "RECOVERED"
        # And the tokenizer_class survives (merge, not overwrite).
        assert local_cfg["tokenizer_class"] == "GemmaTokenizer"

    def test_returns_false_when_no_base_resolvable(self, tmp_path: Path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "SomeTokenizer"})
        )
        fake_info = MagicMock()
        fake_info.card_data = MagicMock(base_model=None)
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.return_value = fake_info

        with patch(
            "huggingface_hub.HfApi",
            return_value=mock_api_instance,
            create=True,
        ):
            # Unknown-org repo with no base_model and no heuristic match.
            ok = ensure_chat_template(tmp_path, "acme/random-model-foobar")

        assert ok is False

    def test_never_raises_on_network_failure(self, tmp_path: Path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "GemmaTokenizer"})
        )
        mock_api_instance = MagicMock()
        mock_api_instance.model_info.side_effect = RuntimeError("connection refused")

        with patch(
            "huggingface_hub.HfApi",
            return_value=mock_api_instance,
            create=True,
        ):
            # Heuristic still kicks in; fail download too.
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=RuntimeError("network down"),
                create=True,
            ):
                ok = ensure_chat_template(tmp_path, "mlx-community/gemma-4-31b-it-4bit")

        assert ok is False  # graceful fail, not an exception

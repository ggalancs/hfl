# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The interactive Hub search: its filters must search the Hub, not a page.

Measured before these tests (2026-09-24): `hfl search qwen3 --gguf
--max-params 8` answered "No models <8.0B found" while unsloth/Qwen3-4B-GGUF,
Qwen/Qwen3-8B-GGUF and others exist. Both filters ran after the Hub answered,
over the top `--limit` results only (3 GGUF among the first 30, all 27B+).
After the fix the same query lists 30 small Qwen3 GGUF repos.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


def _model(repo, gguf=True):
    m = MagicMock()
    m.id = repo
    m.downloads = 100
    m.likes = 1
    m.pipeline_tag = "text-generation"
    m.siblings = [MagicMock(rfilename="m-Q4_K_M.gguf")] if gguf else []
    return m


def _search(args, results):
    from hfl.cli.main import app

    with patch("huggingface_hub.HfApi") as api_class:
        api = MagicMock()
        api.list_models.return_value = results
        api_class.return_value = api
        result = CliRunner().invoke(app, ["search", *args], input="q\n")
    return result, api.list_models.call_args.kwargs


class TestFiltersReachTheHub:
    def test_gguf_is_filtered_by_the_hub(self, temp_config):
        _, kwargs = _search(["qwen3", "--gguf"], [_model("org/Qwen3-4B-GGUF")])
        assert kwargs.get("filter") == "gguf"

    def test_no_hub_filter_without_the_flag(self, temp_config):
        _, kwargs = _search(["qwen3"], [_model("org/Qwen3-4B")])
        assert "filter" not in kwargs

    def test_a_size_filter_fetches_a_wider_window(self, temp_config):
        _, kwargs = _search(["qwen3", "--max-params", "8", "--limit", "30"], [])
        assert kwargs["limit"] == 300

    def test_the_window_is_capped(self, temp_config):
        _, kwargs = _search(["qwen3", "--min-params", "1", "--limit", "500"], [])
        assert kwargs["limit"] == 1000

    def test_without_a_size_filter_the_limit_is_the_limit(self, temp_config):
        _, kwargs = _search(["qwen3", "--limit", "30"], [])
        assert kwargs["limit"] == 30

    def test_the_answer_is_cut_back_to_the_limit(self, temp_config):
        many = [_model(f"org/Qwen3-{i % 7 + 1}B-GGUF-{i}") for i in range(40)]
        result, _ = _search(
            ["qwen3", "--max-params", "8", "--limit", "5", "--page-size", "10"], many
        )
        assert "5 models found" in result.stdout


class TestSizesAreTotals:
    """What the search shows and filters on is the model's TOTAL size."""

    @pytest.mark.parametrize(
        ("repo", "expected"),
        [
            ("meta-llama/Llama-3.3-70B-Instruct", "70B"),
            ("microsoft/phi-1.5b", "1.5B"),
            ("Qwen/Qwen3-30B-A3B", "30B"),
            ("Qwen/Qwen3.8-2.4T-A95B", "2400B"),
            # Only the active count is in the name: no size beats a wrong one.
            ("meta-llama/Llama-4-Scout-17B-16E-Instruct", None),
            ("tencent/Hunyuan-A13B-Instruct", None),
        ],
    )
    def test_names(self, repo, expected):
        from hfl.cli.commands._utils import extract_params_from_name

        assert extract_params_from_name(repo) == expected

    def test_a_large_moe_is_not_let_through_a_small_size_filter(self, temp_config):
        result, _ = _search(
            ["llama", "--max-params", "20"],
            [_model("meta-llama/Llama-4-Scout-17B-16E-Instruct"), _model("org/Llama-3.2-3B-GGUF")],
        )
        assert "Llama-4-Scout" not in result.stdout
        assert "Llama-3.2-3B" in result.stdout


class TestSelection:
    """A number picks the model on the page and pulls it — on any page."""

    @staticmethod
    def _run(results, keys, page_size="10"):
        from hfl.cli import main

        pressed = iter(keys)
        picked = []
        with patch("huggingface_hub.HfApi") as api_class:
            api = MagicMock()
            api.list_models.return_value = results
            api_class.return_value = api
            with patch.object(main, "get_key", lambda: next(pressed)):
                with patch.object(main, "_pull_selected_model", lambda m: picked.append(m.id)):
                    result = CliRunner().invoke(
                        main.app, ["search", "qwen3", "--page-size", page_size]
                    )
        return result, picked

    def test_a_single_page_of_results_can_be_picked_from(self, temp_config):
        results = [_model("org/Qwen3-4B-GGUF"), _model("org/Qwen3-8B-GGUF")]
        _, picked = self._run(results, ["1"])
        assert picked == ["org/Qwen3-8B-GGUF"]

    def test_the_last_page_can_be_picked_from(self, temp_config):
        results = [_model(f"org/Qwen3-{i}B-GGUF") for i in range(1, 5)]
        _, picked = self._run(results, [" ", "0"], page_size="3")
        assert picked == ["org/Qwen3-4B-GGUF"]

    def test_q_leaves_without_pulling(self, temp_config):
        _, picked = self._run([_model("org/Qwen3-4B-GGUF")], ["q"])
        assert picked == []


class TestWithoutARawKeyboard:
    """A pipe or a console without raw input reads lines instead of keys."""

    @staticmethod
    def _run(results, typed, page_size="10"):
        from hfl.cli import main

        def no_raw_keyboard():
            raise OSError("not a tty")

        picked = []
        with patch("huggingface_hub.HfApi") as api_class:
            api = MagicMock()
            api.list_models.return_value = results
            api_class.return_value = api
            with patch.object(main, "get_key", no_raw_keyboard):
                with patch.object(main, "_pull_selected_model", lambda m: picked.append(m.id)):
                    result = CliRunner().invoke(
                        main.app, ["search", "qwen3", "--page-size", page_size], input=typed
                    )
        return result, picked

    def test_a_typed_number_selects(self, temp_config):
        _, picked = self._run([_model("org/A-GGUF"), _model("org/B-GGUF")], "1\n")
        assert picked == ["org/B-GGUF"]

    def test_enter_on_the_last_page_ends_the_search(self, temp_config):
        result, picked = self._run([_model("org/A-GGUF")], "\n")
        assert result.exit_code == 0 and picked == []

    def test_p_goes_back_a_page(self, temp_config):
        results = [_model(f"org/M{i}-GGUF") for i in range(4)]
        _, picked = self._run(results, "\np\n0\n", page_size="2")
        assert picked == ["org/M0-GGUF"]

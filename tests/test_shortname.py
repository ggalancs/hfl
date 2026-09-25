# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Short model names: ``hfl run qwen3-coder`` without a Hub reference.

The rules were tuned against the real Hub (qwen3-coder, llama3.2, qwen3:8b,
qwen3:0.6b, gemma3, deepseek-r1:8b, gpt-oss...); these tests pin each one
against a fake Hub so the suite stays offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from hfl.hub import shortname
from hfl.hub.shortname import find_options, is_short_name, split_name

GB = 1024**3


class FakeHub:
    """list_models by substring, model_info with file sizes."""

    def __init__(self, repos: dict[str, dict]):
        # repo_id -> {"downloads": int, "files": {filename: size}}
        self.repos = repos
        self.searched: list[str] = []

    def list_models(self, search, filter, sort, limit):
        self.searched.append(search)
        hits = [
            SimpleNamespace(id=r, downloads=v["downloads"])
            for r, v in self.repos.items()
            if search.lower() in r.lower()
        ]
        return sorted(hits, key=lambda m: -m.downloads)[:limit]

    def model_info(self, repo_id, files_metadata=False):
        files = self.repos[repo_id]["files"]
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=f, size=s) for f, s in files.items()]
        )


def _repo(downloads, **quants):
    return {
        "downloads": downloads,
        "files": {f"model-{q}.gguf": int(size * GB) for q, size in quants.items()},
    }


class TestNames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("qwen3-coder", ("qwen3-coder", None, None)),
            ("qwen3:8b", ("qwen3", "8b", None)),
            ("qwen3:0.6b", ("qwen3", "0.6b", None)),
            ("qwen3:Q8_0", ("qwen3", None, "Q8_0")),
            ("qwen3:8b-q8_0", ("qwen3", "8b", "Q8_0")),
            ("Llama3.2", ("llama3.2", None, None)),
        ],
    )
    def test_split(self, name, expected):
        assert split_name(name) == expected

    @pytest.mark.parametrize(
        ("name", "ok"),
        [
            ("qwen3-coder", True),
            ("qwen3:8b", True),
            ("phi", True),
            ("x", False),  # too short: matches half the Hub
            ("invalid:model:format", False),
            ("org/model", False),
            ("../etc", False),
            ("with space", False),
        ],
    )
    def test_what_counts_as_a_short_name(self, name, ok):
        assert is_short_name(name) is ok


class TestSearch:
    def test_the_usual_build_first_derivatives_and_other_tasks_out(self):
        hub = FakeHub(
            {
                "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF": _repo(500, Q4_K_M=17.3),
                "someone/Qwen3-Coder-30B-A3B-abliterated-GGUF": _repo(9000, Q4_K_M=17.3),
                "someone/Qwen3-Coder-ASR-GGUF": _repo(9000, Q4_K_M=1),
                "someone/Delphi-Qwen3-Coder-GGUF": _repo(100, Q4_K_M=5),
            }
        )
        options = find_options("qwen3-coder", api=hub, budget_bytes=100 * GB)
        assert [o.repo_id for o in options] == [
            "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "someone/Delphi-Qwen3-Coder-GGUF",  # "qwen3-coder" starts a word there
        ]
        assert options[0].reference == "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"

    @pytest.mark.parametrize(
        "repo",
        [
            "someone/gpt-oss-20b-Derestricted-GGUF",
            "someone/DeepSeek-R1-Distill-Qwen-1.5B-OBLITERATED",
            "unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF",  # pruned experts
        ],
    )
    def test_derivatives_seen_on_the_hub_are_left_out(self, repo):
        assert shortname._DERIVATIVE.search(repo)

    def test_a_name_must_start_a_word(self):
        hub = FakeHub({"acme/Delphi-7B-GGUF": _repo(10, Q4_K_M=4)})
        assert find_options("phi", api=hub, budget_bytes=100 * GB) == []

    def test_spelling_with_a_dash_is_searched_too(self):
        hub = FakeHub({"unsloth/Llama-3.2-3B-Instruct-GGUF": _repo(10, Q4_K_M=1.9)})
        options = find_options("llama3.2", api=hub, budget_bytes=100 * GB)
        assert options and options[0].repo_id == "unsloth/Llama-3.2-3B-Instruct-GGUF"
        assert "llama-3.2" in hub.searched

    def test_a_size_tag_filters_and_is_searched(self):
        hub = FakeHub(
            {
                "Qwen/Qwen3-8B-GGUF": _repo(100, Q4_K_M=4.7),
                "Qwen/Qwen3-0.6B-GGUF": _repo(10, Q8_0=0.6),
            }
        )
        options = find_options("qwen3:0.6b", api=hub, budget_bytes=100 * GB)
        assert [o.repo_id for o in options] == ["Qwen/Qwen3-0.6B-GGUF"]
        assert options[0].quantization == "Q8_0"  # the only quant it publishes
        assert "qwen3-0.6b" in hub.searched

    def test_one_build_per_model_whoever_published_it(self):
        hub = FakeHub(
            {
                "bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-GGUF": _repo(50, Q4_K_M=4.7),
                "lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF": _repo(80, Q4_K_M=4.7),
                "hugging-quants/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M-GGUF": _repo(10, Q4_K_M=4.7),
            }
        )
        options = find_options("deepseek-r1:8b", api=hub, budget_bytes=100 * GB)
        assert len(options) == 1

    def test_instruct_and_known_publishers_rank_first(self):
        hub = FakeHub(
            {
                "nobody/Gemma-3-4B-GGUF": _repo(1000, Q4_K_M=2.3),
                "unsloth/gemma-3-4b-it-GGUF": _repo(200, Q4_K_M=2.3),
            }
        )
        options = find_options("gemma3", api=hub, budget_bytes=100 * GB)
        assert options[0].repo_id == "unsloth/gemma-3-4b-it-GGUF"  # 200 x3 x2 > 1000


class TestQuant:
    def test_q4_k_m_by_default(self):
        hub = FakeHub({"a/Model-7B-GGUF": _repo(1, Q8_0=7, Q4_K_M=4, Q2_K=2)})
        assert find_options("model", api=hub, budget_bytes=100 * GB)[0].quantization == "Q4_K_M"

    def test_smaller_when_the_default_does_not_fit(self):
        hub = FakeHub({"a/Model-70B-GGUF": _repo(1, Q4_K_M=40, Q3_K_M=31, Q2_K=24)})
        option = find_options("model", api=hub, budget_bytes=35 * GB)[0]
        assert option.quantization == "Q2_K"  # Q3_K_M x 1.15 = 35.7 > 35

    def test_nothing_that_fits_is_nothing(self):
        hub = FakeHub({"a/Model-400B-GGUF": _repo(1, Q4_K_M=200, Q2_K=120)})
        assert find_options("model", api=hub, budget_bytes=100 * GB) == []

    def test_an_explicit_quant_is_honoured(self):
        hub = FakeHub({"a/Model-7B-GGUF": _repo(1, Q8_0=7, Q4_K_M=4)})
        assert find_options("model:q8_0", api=hub, budget_bytes=100 * GB)[0].quantization == "Q8_0"

    def test_split_files_add_up_and_projectors_do_not_count(self):
        files = {
            "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf": 30 * GB,
            "Q4_K_M/model-Q4_K_M-00002-of-00002.gguf": 20 * GB,
            "mmproj-model-f16.gguf": 2 * GB,
        }
        hub = FakeHub({"a/Model-GGUF": {"downloads": 1, "files": files}})
        option = find_options("model", api=hub, budget_bytes=100 * GB)[0]
        assert (option.quantization, option.size_bytes) == ("Q4_K_M", 50 * GB)


@pytest.fixture
def cli(temp_config, monkeypatch):
    from hfl.cli import main

    hub = FakeHub(
        {
            "unsloth/gemma-3-1b-it-GGUF": _repo(900, Q4_K_M=0.8),
            "unsloth/gemma-3-4b-it-GGUF": _repo(500, Q4_K_M=2.3),
        }
    )
    monkeypatch.setattr(shortname, "_default_api", lambda: hub)
    monkeypatch.setattr(shortname, "default_budget_bytes", lambda: 100 * GB)
    pulled: list[dict] = []

    def fake_pull(**kwargs):
        from hfl.models.manifest import ModelManifest
        from hfl.models.registry import ModelRegistry

        pulled.append(kwargs)
        repo, quant = kwargs["model"].removeprefix("hf.co/").rsplit(":", 1)
        ModelRegistry().add(
            ModelManifest(
                name=repo.split("/")[1].lower(),
                repo_id=repo,
                local_path="/nowhere/m.gguf",
                format="gguf",
                quantization=quant,
                alias=kwargs.get("alias"),
            )
        )

    monkeypatch.setattr(main, "pull", fake_pull)
    return main, hub, pulled


def test_the_user_picks_from_the_sizes(cli, monkeypatch):
    main, hub, pulled = cli
    from hfl.models.registry import ModelRegistry

    monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(main.typer, "prompt", lambda *a, **k: "2")
    manifest = main._local_or_pulled("gemma3", ModelRegistry)
    assert pulled[0]["model"] == "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M"
    assert pulled[0]["alias"] == "gemma3"
    assert manifest.repo_id == "unsloth/gemma-3-4b-it-GGUF"


def test_the_choice_is_remembered(cli, monkeypatch):
    """A tagged name is remembered under an alias that differs from what was
    typed (``gemma3:4b`` -> ``gemma3-4b``: ':' is not allowed in a name), so
    the registry's own alias lookup does not find it — the short-name path
    must look the alias up itself before searching again."""
    main, hub, pulled = cli
    from hfl.models.registry import ModelRegistry

    main._local_or_pulled("gemma3:4b", ModelRegistry, assume_yes=True)
    assert pulled[0]["alias"] == "gemma3-4b"
    searches = len(hub.searched)
    again = main._local_or_pulled("gemma3:4b", ModelRegistry, assume_yes=True)
    assert again.repo_id == "unsloth/gemma-3-4b-it-GGUF"
    assert len(hub.searched) == searches and len(pulled) == 1  # no Hub, no download


def test_without_a_terminal_nothing_is_downloaded(cli, monkeypatch):
    main, hub, pulled = cli
    from hfl.models.registry import ModelRegistry

    monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))
    import typer

    with pytest.raises(typer.Exit):
        main._local_or_pulled("gemma3", ModelRegistry)
    assert pulled == []


def test_a_cancelled_choice_downloads_nothing(cli, monkeypatch):
    main, hub, pulled = cli
    from hfl.models.registry import ModelRegistry

    monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(main.typer, "prompt", lambda *a, **k: "n")
    import typer

    with pytest.raises(typer.Exit):
        main._local_or_pulled("gemma3", ModelRegistry)
    assert pulled == []


def test_an_invalid_name_is_not_searched(cli):
    main, hub, pulled = cli
    from hfl.models.registry import ModelRegistry

    assert main._local_or_pulled("invalid:model:format", ModelRegistry) is None
    assert hub.searched == [] and pulled == []


def test_run_says_yes_with_the_flag(cli, monkeypatch):
    main, hub, pulled = cli
    seen = {}
    monkeypatch.setattr(
        main,
        "_local_or_pulled",
        lambda model, cls, **k: seen.update(k) or None,
    )
    CliRunner().invoke(main.app, ["run", "gemma3", "--yes"])
    assert seen.get("assume_yes") is True

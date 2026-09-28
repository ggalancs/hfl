# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl search`` reads a query the way a person writes it. "coding assistant
7b" went to the Hub as a phrase and matched five repo names (7 downloads for
the best); it means a coding model of about 7B."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hfl.hub.query import describe, hub_queries, parse


def test_task_size_and_filler_are_read() -> None:
    intent = parse("coding assistant 7b")
    assert (intent.task, intent.size_b, intent.words) == ("code", 7.0, [])
    low, high = intent.size_range() or (0, 0)
    assert low < 7 < 8 <= high  # Llama 8B and Qwen 7B both count as "7b"
    assert hub_queries(intent) == [{"search": "coder"}, {"search": "code"}]
    assert "code models" in describe(intent)


def test_a_task_with_a_pipeline_and_gguf() -> None:
    intent = parse("vision 3b gguf")
    assert (intent.task, intent.size_b, intent.gguf) == ("vision", 3.0, True)
    assert all(q["pipeline_tag"] == "image-text-to-text" for q in hub_queries(intent))


def test_names_stay_in_the_search() -> None:
    intent = parse("qwen coder 14b")
    assert intent.words == ["qwen"]
    assert hub_queries(intent) == [{"search": "qwen coder"}, {"search": "qwen code"}]


@pytest.mark.parametrize("query", ["qwen2.5", "llama-3.2-1b-instruct", "bartowski phi"])
def test_a_plain_query_is_not_interpreted(query) -> None:
    assert not parse(query).interpreted


def _run(monkeypatch, args):
    import huggingface_hub
    from typer.testing import CliRunner

    from hfl.cli.main import app

    calls: list[dict] = []
    repos = [
        SimpleNamespace(id="Qwen/Qwen2.5-Coder-7B-Instruct", downloads=2_400_000, likes=9,
                        siblings=[], pipeline_tag="text-generation"),
        SimpleNamespace(id="someone/coding-assistant-7b", downloads=7, likes=0,
                        siblings=[], pipeline_tag="text-generation"),
        SimpleNamespace(id="big/Coder-33B", downloads=5_000_000, likes=1,
                        siblings=[], pipeline_tag="text-generation"),
    ]  # fmt: skip

    class Api:
        def list_models(self, **kwargs):
            calls.append(kwargs)
            return iter(repos)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    result = CliRunner().invoke(app, args, input="q\n")
    return result, calls


def test_the_command_searches_what_the_query_means(monkeypatch) -> None:
    result, calls = _run(monkeypatch, ["search", "coding assistant 7b"])
    assert [c.get("search") for c in calls] == ["coder", "code"]
    assert "Qwen2.5-Coder-7B-Instruct" in result.output
    assert "Coder-33B" not in result.output  # outside "about 7B"
    assert "code models" in result.output  # it says how it read the query


def test_literal_searches_the_text_as_written(monkeypatch) -> None:
    _result, calls = _run(monkeypatch, ["search", "coding assistant 7b", "--literal"])
    assert [c.get("search") for c in calls] == ["coding assistant 7b"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""`hfl run <hub reference>`: one command from the Hub's "Use this model".

Apps listed in the Hub's Local Apps menu offer a single line to copy
(``ollama run hf.co/org/model:Q4_K_M``). HFL needed two — ``hfl pull``
then ``hfl run <derived name>`` — so `hfl run` now accepts the reference,
uses a local copy when there is one, and pulls it otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hfl.hub.resolver import ModelSpec, parse_model_spec, strip_hub_prefix


class TestSpecParsing:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("org/model", ModelSpec("org/model", None, None)),
            ("org/model:Q4_K_M", ModelSpec("org/model", "Q4_K_M", None)),
            ("hf.co/org/model:Q4_K_M", ModelSpec("org/model", "Q4_K_M", None)),
            ("huggingface.co/org/model", ModelSpec("org/model", None, None)),
            ("https://huggingface.co/org/model:q8_0", ModelSpec("org/model", "q8_0", None)),
            ("org/model:Q4_K_M@abc123", ModelSpec("org/model", "Q4_K_M", "abc123")),
            ("org/model@main", ModelSpec("org/model", None, "main")),
            ("qwen3-14b-gguf-q4_k_m", ModelSpec(None, None, None)),
        ],
    )
    def test_references(self, spec, expected):
        assert parse_model_spec(spec) == expected

    def test_a_colon_that_is_not_a_quantization_stays(self):
        assert parse_model_spec("org/model:latest") == ModelSpec("org/model:latest", None, None)

    def test_the_prefix_is_case_insensitive(self):
        assert strip_hub_prefix("HF.CO/Org/Model") == "Org/Model"

    def test_resolve_accepts_the_prefix(self, monkeypatch):
        """`hfl pull hf.co/...` must reach the Hub with the bare repo id."""
        from hfl.hub import resolver

        seen = {}

        class Api:
            def model_info(self, repo_id, revision=None):
                seen["repo"] = repo_id
                return SimpleNamespace(
                    siblings=[SimpleNamespace(rfilename="m-Q4_K_M.gguf")],
                    pipeline_tag="text-generation",
                    sha="abc",
                )

        monkeypatch.setattr(resolver, "HfApi", Api)
        resolved = resolver.resolve("hf.co/org/model:Q4_K_M")
        assert seen["repo"] == "org/model" and resolved.repo_id == "org/model"


def _manifest(name, repo, quant, created):
    from hfl.models.manifest import ModelManifest

    return ModelManifest(
        name=name,
        repo_id=repo,
        local_path=f"/x/{name}",
        format="gguf",
        quantization=quant,
        created_at=created,
    )


class TestFindPulled:
    @pytest.fixture
    def registry(self, temp_config):
        from hfl.models.registry import ModelRegistry

        reg = ModelRegistry()
        reg.add(_manifest("m-q4", "Org/Model", "Q4_K_M", "2026-01-01T00:00:00"))
        reg.add(_manifest("m-q8", "org/model", "Q8_0", "2026-02-01T00:00:00"))
        return reg

    def test_matches_repo_and_quant_case_insensitively(self, registry):
        assert registry.find_pulled("org/MODEL", "q4_k_m").name == "m-q4"

    def test_without_a_quant_the_newest_copy(self, registry):
        assert registry.find_pulled("org/model").name == "m-q8"

    def test_another_quant_is_not_a_match(self, registry):
        assert registry.find_pulled("org/model", "Q5_K_M") is None


class TestRunResolution:
    """`_local_or_pulled`: what `hfl run` opens, and whether it pulls."""

    @pytest.fixture
    def cli(self, monkeypatch):
        from hfl.cli import main

        pulls = []

        def fake_pull(**kwargs):
            pulls.append(kwargs)
            State.manifests.append(
                _manifest("pulled", "org/model", kwargs["quantize"], "2026-03-01T00:00:00")
            )

        class State:
            manifests: list = []

        class Registry:
            def get(self, name):
                return next((m for m in State.manifests if m.name == name), None)

            def find_pulled(self, repo, quant=None):
                for m in reversed(State.manifests):
                    if m.repo_id.lower() == repo.lower() and (
                        quant is None or (m.quantization or "").lower() == quant.lower()
                    ):
                        return m
                return None

        monkeypatch.setattr(main, "pull", fake_pull)
        return SimpleNamespace(main=main, pulls=pulls, state=State, registry=Registry)

    def test_a_local_name_opens_without_pulling(self, cli):
        cli.state.manifests.append(_manifest("mine", "org/model", "Q4_K_M", "2026"))
        assert cli.main._local_or_pulled("mine", cli.registry).name == "mine"
        assert cli.pulls == []

    def test_a_reference_already_on_disk_opens_without_pulling(self, cli):
        cli.state.manifests.append(_manifest("mine", "org/model", "Q4_K_M", "2026"))
        got = cli.main._local_or_pulled("hf.co/org/model:Q4_K_M", cli.registry)
        assert got.name == "mine" and cli.pulls == []

    def test_a_missing_reference_is_pulled_with_its_quant_and_revision(self, cli):
        got = cli.main._local_or_pulled("hf.co/org/model:Q8_0@abc", cli.registry)
        assert got.name == "pulled"
        assert cli.pulls == [
            {
                "model": "hf.co/org/model:Q8_0@abc",
                "quantize": "Q8_0",
                "format": "auto",
                "revision": "abc",
                "alias": None,
                "skip_license": False,
            }
        ]

    def test_another_quant_on_disk_does_not_stop_the_pull(self, cli):
        cli.state.manifests.append(_manifest("mine", "org/model", "Q4_K_M", "2026"))
        cli.main._local_or_pulled("org/model:Q8_0", cli.registry)
        assert len(cli.pulls) == 1

    def test_the_license_check_is_never_skipped(self, cli):
        cli.main._local_or_pulled("org/model", cli.registry)
        assert cli.pulls[0]["skip_license"] is False
        assert cli.pulls[0]["quantize"] == "Q4_K_M"

    def test_an_unknown_bare_name_is_not_guessed_at(self, cli):
        assert cli.main._local_or_pulled("not-here", cli.registry) is None
        assert cli.pulls == []

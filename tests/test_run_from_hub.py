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


class TestApiNames:
    """The API accepts the Hub reference clients learned from the Hub or
    Ollama, maps it to the local copy, and never pulls on its own."""

    @pytest.fixture
    def names(self, monkeypatch):
        from hfl.api import model_loader

        manifests = [_manifest("local-q4", "org/model", "Q4_K_M", "2026")]

        class Registry:
            def get(self, name):
                return next((m for m in manifests if m.name == name), None)

            def find_pulled(self, repo, quant=None):
                return next(
                    (
                        m
                        for m in manifests
                        if m.repo_id.lower() == repo.lower()
                        and (quant is None or m.quantization.lower() == quant.lower())
                    ),
                    None,
                )

        monkeypatch.setattr(model_loader, "get_registry", lambda: Registry())
        return model_loader._canonical_model_name

    def test_a_local_name_is_itself(self, names):
        assert names("local-q4") == "local-q4"

    @pytest.mark.parametrize(
        "ref", ["hf.co/org/model:Q4_K_M", "org/model:q4_k_m", "huggingface.co/org/model"]
    )
    def test_a_reference_maps_to_the_local_copy(self, names, ref):
        assert names(ref) == "local-q4"

    def test_a_reference_not_on_disk_is_a_404_not_a_pull(self, names):
        from hfl.exceptions import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            names("hf.co/org/other:Q4_K_M")

    @pytest.mark.parametrize(
        "bad",
        [
            "hf.co/org/../../etc:Q4_K_M",  # traversal inside the repo part
            "org/model:latest",  # a tag that is not a quantization
            "hf.co/org/model:Q4_K_M@abc",  # revisions are pinned at pull time
        ],
    )
    def test_each_part_is_validated(self, names, bad):
        from hfl.exceptions import ValidationError as APIValidationError

        with pytest.raises(APIValidationError):
            names(bad)


def test_serve_preload_goes_through_the_same_resolution(monkeypatch, temp_config):
    """`hfl serve --model hf.co/org/model:Q4_K_M` pulls if needed, like run."""
    from typer.testing import CliRunner

    from hfl.cli import main

    seen = []
    monkeypatch.setattr(main, "_local_or_pulled", lambda model, cls: seen.append(model))
    monkeypatch.setattr("hfl.api.server.start_server", lambda **k: None)
    result = CliRunner().invoke(
        main.app, ["serve", "--model", "hf.co/org/model:Q4_K_M", "--host", "127.0.0.1"]
    )
    assert seen == ["hf.co/org/model:Q4_K_M"]
    assert result.exit_code == 1  # resolution returned nothing: said, not ignored


class TestQuantizationLabel:
    """A model is labelled with a quantization only if it IS quantized so."""

    @staticmethod
    def _pull(tmp_path, temp_config, monkeypatch, resolved, files):
        from unittest.mock import patch

        from typer.testing import CliRunner

        from hfl.cli.main import app
        from hfl.models.registry import ModelRegistry

        target = temp_config.models_dir / "repo"
        target.mkdir(parents=True)
        for name, content in files.items():
            (target / name).write_bytes(content)
        path = target / next(iter(files)) if len(files) == 1 else target
        monkeypatch.setattr("hfl.converter.formats.is_mlx_quantized_repo", lambda *a: True)
        monkeypatch.setattr("hfl.engine.selector._mlx_preferred", lambda: True)
        with patch("hfl.hub.resolver.resolve", return_value=resolved):
            with patch("hfl.hub.downloader.pull_model", return_value=path):
                result = CliRunner().invoke(app, ["pull", "x", "--skip-license"])
        assert result.exit_code == 0, result.stdout
        return ModelRegistry().list_all()[0]

    def test_an_mlx_build_kept_as_is_has_no_gguf_label(self, tmp_path, temp_config, monkeypatch):
        from hfl.hub.resolver import ResolvedModel

        # What the resolver returns for a safetensors repo: the REQUESTED level.
        resolved = ResolvedModel(
            repo_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            format="safetensors",
            quantization="Q4_K_M",
            pipeline_tag="text-generation",
        )
        manifest = self._pull(
            tmp_path,
            temp_config,
            monkeypatch,
            resolved,
            {"model.safetensors": b"x", "config.json": b"{}"},
        )
        assert manifest.name == "qwen2.5-0.5b-instruct-4bit"
        assert manifest.quantization is None

    def test_a_gguf_keeps_its_label(self, tmp_path, temp_config, monkeypatch):
        from hfl.hub.resolver import ResolvedModel

        resolved = ResolvedModel(
            repo_id="org/Model-GGUF",
            filename="model-Q5_K_M.gguf",
            format="gguf",
            quantization="Q5_K_M",
            pipeline_tag="text-generation",
        )
        manifest = self._pull(
            tmp_path, temp_config, monkeypatch, resolved, {"model-Q5_K_M.gguf": b"GGUF"}
        )
        assert manifest.name == "model-gguf-q5_k_m"
        assert manifest.quantization == "Q5_K_M"


def test_api_pull_labels_only_gguf(tmp_path, monkeypatch, temp_config):
    from types import SimpleNamespace

    from hfl.api.routes_pull import _record_server_pull
    from hfl.models.registry import ModelRegistry

    license_info = SimpleNamespace(
        license_id="apache-2.0", license_name="Apache", url="u", restrictions=[], gated=False
    )
    folder = tmp_path / "mlx"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x")
    (folder / "config.json").write_bytes(b"{}")
    resolved = SimpleNamespace(
        repo_id="mlx-community/M-4bit", quantization="Q4_K_M", revision="main", commit_sha="abc"
    )
    _record_server_pull(resolved, folder, license_info, "permissive")
    manifest = ModelRegistry().get("m-4bit")
    assert manifest is not None and manifest.quantization is None

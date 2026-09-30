# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The pull steps ``hfl pull`` and ``/api/pull`` share.

They had drifted: the server registered no model type and kept a model
HFL cannot serve; the CLI logged no provenance and could take an alias
another model used."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hfl.hub import pull_service
from tests.gguf_file import model_fields, write_gguf


class _Registry:
    def __init__(self, *entries):
        self.by_name = {e.name: e for e in entries}

    def get(self, name):
        if name in self.by_name:
            return self.by_name[name]
        return next((e for e in self.by_name.values() if e.alias == name), None)

    def add(self, manifest):
        self.by_name[manifest.name] = manifest


def _resolved(repo="org/Model-GGUF", tag="text-generation", quant="Q4_K_M"):
    return SimpleNamespace(
        repo_id=repo, pipeline_tag=tag, quantization=quant, revision="main", commit_sha="abc"
    )


def _gguf(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "m.gguf", model_fields("llama"), size_bytes=2048)


@pytest.fixture
def provenance(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "hfl.models.provenance.log_conversion", lambda **kw: seen.append(kw) or None
    )
    return seen


def test_registration_records_the_type_and_the_provenance(tmp_path, provenance):
    from hfl.converter.formats import ModelType

    registry = _Registry()
    finished = pull_service.Finished(path=_gguf(tmp_path), model_type=ModelType.LLM)
    manifest = pull_service.register_pulled(
        _resolved(), finished, registry=registry, license_info=None,
        accepted_at=None, alias="m", quantize="Q4_K_M", source="hfl pull",
    )  # fmt: skip
    assert manifest.name == "model-gguf-q4_k_m" and manifest.model_type == "llm"
    assert registry.get("m") is manifest
    assert provenance and provenance[0]["notes"] == "hfl pull"


def test_a_re_pull_keeps_its_alias_and_a_used_alias_is_not_taken(tmp_path, provenance):
    from hfl.converter.formats import ModelType

    other = SimpleNamespace(name="other-model", alias="taken")
    before = SimpleNamespace(name="model-gguf-q4_k_m", alias="mine")
    registry = _Registry(other, before)
    finished = pull_service.Finished(path=_gguf(tmp_path), model_type=ModelType.LLM)
    again = pull_service.register_pulled(
        _resolved(), finished, registry=registry, license_info=None,
        accepted_at=None, alias=None, quantize="Q4_K_M", source="x",
    )  # fmt: skip
    assert again.alias == "mine"
    stolen = pull_service.register_pulled(
        _resolved(), finished, registry=registry, license_info=None,
        accepted_at=None, alias="taken", quantize="Q4_K_M", source="x",
    )  # fmt: skip
    assert stolen.alias is None and registry.get("taken") is other


def test_a_type_hfl_cannot_serve_is_known_before_downloading():
    assert pull_service.unsupported_type(_resolved(tag="image-segmentation")) is not None
    assert pull_service.unsupported_type(_resolved(tag="text-generation")) is None


def test_one_found_only_once_downloaded_is_removed(tmp_path, monkeypatch):
    from hfl.converter.formats import ModelType

    folder = tmp_path / "repo"
    folder.mkdir()
    (folder / "weights.bin").write_bytes(b"x")
    monkeypatch.setattr("hfl.converter.formats.detect_model_type", lambda p: ModelType.UNKNOWN)
    monkeypatch.setattr("hfl.converter.formats.is_model_type_supported", lambda t: False)
    with pytest.raises(pull_service.PullStepError) as caught:
        pull_service.finish_download(_resolved(tag=None), folder)
    assert caught.value.key == "errors.unsupported_model_type" and not folder.exists()


def test_the_server_keeps_safetensors_as_they_are(tmp_path, monkeypatch):
    folder = tmp_path / "repo"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(pull_service, "_convert", lambda *a: pytest.fail("converted"))
    done = pull_service.finish_download(_resolved(), folder, convert=False)
    assert done.kept == pull_service.Kept.AS_IS and done.path == folder


def test_the_conversion_is_announced_before_it_runs(tmp_path, monkeypatch):
    folder = tmp_path / "repo"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x")
    order = []
    monkeypatch.setattr("hfl.engine.selector._mlx_preferred", lambda: False)
    monkeypatch.setattr("hfl.converter.formats.is_mlx_quantized_repo", lambda r, p: False)
    monkeypatch.setattr(
        pull_service, "_convert", lambda path, repo, q: order.append("convert") or path
    )
    done = pull_service.finish_download(
        _resolved(), folder, on_convert=lambda: order.append("announce")
    )
    assert order == ["announce", "convert"] and done.kept == pull_service.Kept.CONVERTED

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``/api/show`` and the model lists take the architecture and the trained
context from the model's own files, as Ollama does.

Measured on 2026-11-06: all eleven of the owner's models answered
``general.architecture: unknown`` with no ``<arch>.context_length``, because
no pull records them in the manifest. Copilot Chat's Ollama provider sizes
its prompts from ``model_info[general.architecture + ".context_length"]``
and falls back to 32768, so qwen3-coder (trained for 262144) was offered an
eighth of its window.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.models.file_facts import architecture, file_facts
from tests.gguf_file import model_fields, write_gguf

CODER = model_fields(
    "qwen3moe",
    context_length=262144,
    embedding_length=2048,
    block_count=48,
    head_count=32,
    head_count_kv=4,
)


def _copilot_context(model_info: dict) -> int:
    """How Copilot Chat's Ollama provider reads the window."""
    return model_info.get(f"{model_info['general.architecture']}.context_length", 32768)


@pytest.fixture
def client(temp_config):
    return TestClient(app)


@pytest.fixture
def coder(temp_config, tmp_path):
    """A model as a pull registers it: no architecture, no context."""
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import ModelRegistry

    path = write_gguf(tmp_path / "coder.gguf", CODER)
    manifest = ModelManifest(
        name="coder",
        repo_id="Qwen/Coder-GGUF",
        local_path=str(path),
        format="gguf",
        quantization="Q4_K_M",
    )
    ModelRegistry().add(manifest)
    return manifest


def test_show_reports_the_architecture_and_window_the_file_gives(client, coder):
    body = client.post("/api/show", json={"model": "coder"}).json()
    info = body["model_info"]
    assert info["general.architecture"] == "qwen3moe"
    assert _copilot_context(info) == 262144
    assert info["qwen3moe.embedding_length"] == 2048
    assert info["qwen3moe.block_count"] == 48
    assert info["qwen3moe.attention.head_count"] == 32
    assert info["qwen3moe.attention.head_count_kv"] == 4
    assert body["details"]["family"] == "qwen3moe"
    assert body["details"]["families"] == ["qwen3moe"]


def test_the_trained_window_is_reported_and_the_pinned_one_stays_a_parameter(client, coder):
    from hfl.models.registry import ModelRegistry

    coder.context_length = 8192
    ModelRegistry().add(coder)  # replaces the entry of the same name
    body = client.post("/api/show", json={"model": "coder"}).json()
    assert body["model_info"]["qwen3moe.context_length"] == 262144
    assert "num_ctx 8192" in body["parameters"].splitlines()


def test_the_lists_name_the_same_family(client, coder):
    from hfl.api.routes_ps import _manifest_details

    tags = client.get("/api/tags").json()["models"]
    assert [m["details"]["family"] for m in tags] == ["qwen3moe"]
    assert _manifest_details(coder)["family"] == "qwen3moe"
    assert _manifest_details(coder)["families"] == ["qwen3moe"]


def test_a_safetensors_checkpoint_reports_its_config(tmp_path):
    """A vision model nests its language model: the checkpoint's own type
    names it, the nested config gives the window."""
    from hfl.models.manifest import ModelManifest

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "max_position_embeddings": 262144,
                    "hidden_size": 5376,
                    "num_hidden_layers": 60,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 16,
                },
            }
        )
    )
    assert file_facts(tmp_path) == {
        "architecture": "gemma4",
        "context_length": 262144,
        "embedding_length": 5376,
        "block_count": 60,
        "head_count": 32,
        "head_count_kv": 16,
    }
    manifest = ModelManifest(name="g", repo_id="r", local_path=str(tmp_path), format="safetensors")
    assert architecture(manifest) == "gemma4"


def test_unreadable_files_fall_back_to_the_manifest(client, temp_config, tmp_path):
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import ModelRegistry

    broken = tmp_path / "broken.gguf"
    broken.write_bytes(b"not a gguf at all")
    manifest = ModelManifest(
        name="broken",
        repo_id="r",
        local_path=str(broken),
        format="gguf",
        architecture="llama",
        context_length=4096,
    )
    ModelRegistry().add(manifest)
    assert file_facts(broken) == {}
    assert architecture(manifest) == "llama"
    info = client.post("/api/show", json={"model": "broken"}).json()["model_info"]
    assert info["general.architecture"] == "llama"
    assert info["llama.context_length"] == 4096


def test_an_empty_path_is_not_the_working_directory(tmp_path, monkeypatch):
    from hfl.models.manifest import ModelManifest

    write_gguf(tmp_path / "other.gguf", model_fields("other", context_length=999))
    monkeypatch.chdir(tmp_path)
    manifest = ModelManifest(
        name="m", repo_id="r", local_path="", format="gguf", architecture="llama"
    )
    assert architecture(manifest) == "llama"
    assert file_facts("") == {}


@pytest.mark.parametrize("bad", [True, 0, -1, "262144", None])
def test_a_count_that_is_not_a_positive_integer_is_left_out(tmp_path, bad):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "max_position_embeddings": bad, "hidden_size": 64})
    )
    assert file_facts(tmp_path) == {"architecture": "llama", "embedding_length": 64}

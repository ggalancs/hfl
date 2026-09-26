# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The chat web UI offers chat models only (local audit F8): it listed
embedding and speech models too, defaulted to one, and a new user's first
message failed with "not a llm model"."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hfl.models.capabilities import detect_capabilities
from hfl.models.manifest import ModelManifest


def _manifest(name: str, model_type: str | None) -> ModelManifest:
    return ModelManifest(
        name=name,
        repo_id=f"org/{name}",
        local_path="/nowhere",
        format="safetensors",
        model_type=model_type,
    )


@pytest.mark.parametrize(
    ("model_type", "completes"),
    [
        ("llm", True),
        (None, True),  # a GGUF carries no type: a text model
        ("embedding", False),  # what pull stores (ModelType.EMBEDDING)
        ("embed", False),
        ("tts", False),
        ("stt", False),
        ("image-generation", False),
        ("video", False),
    ],
)
def test_only_text_models_complete(model_type, completes) -> None:
    caps = detect_capabilities(_manifest("m", model_type))
    assert ("completion" in caps) is completes


def test_an_embedding_type_is_an_embedding_model() -> None:
    assert detect_capabilities(_manifest("m", "embedding")) == ["embedding"]


def test_tags_carry_capabilities(temp_config) -> None:
    from fastapi.testclient import TestClient

    from hfl.api.server import app
    from hfl.models.registry import get_registry, reset_registry

    reset_registry()
    get_registry().add(_manifest("minilm", "embedding"))
    get_registry().add(_manifest("chatty", "llm"))
    models = {m["name"]: m for m in TestClient(app).get("/api/tags").json()["models"]}
    assert models["minilm"]["capabilities"] == ["embedding"]
    assert "completion" in models["chatty"]["capabilities"]


def test_the_page_filters_on_completion() -> None:
    page = (Path(__file__).resolve().parents[1] / "src/hfl/ui/chat.html").read_text()
    code = "\n".join(line for line in page.splitlines() if not line.strip().startswith("//"))
    assert re.search(r"capabilities\.includes\(\"completion\"\)", code)
    assert re.search(r"\(data\.models \|\| \[\]\)\.filter\(chatty\)", code)


@pytest.mark.parametrize(
    ("repo", "thinks"),
    [
        ("Qwen/Qwen3-0.6B-GGUF", True),
        ("Qwen/Qwen3-30B-A3B", True),
        ("Qwen/Qwen3-4B-Thinking-2507", True),
        ("Qwen/Qwen3-4B-Instruct-2507", False),
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", False),
        ("Qwen/Qwen2.5-0.5B-Instruct", False),
    ],
)
def test_qwen3_is_a_thinking_model(repo, thinks) -> None:
    """Hybrid Qwen3 reasons by default; only "qwen3-thinking" was matched."""
    manifest = ModelManifest(
        name=repo.split("/")[1].lower(),
        repo_id=repo,
        local_path="/nowhere",
        format="gguf",
        model_type="llm",
    )
    assert ("thinking" in detect_capabilities(manifest)) is thinks

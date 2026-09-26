# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``show`` reports the model's own chat template, read from its files — it
reported only a Modelfile TEMPLATE, so every pulled model's was empty (local
audit A34/B25)."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

from hfl.models.chat_template import model_template

TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


def _gguf(path, template: str | None):
    fields = [("general.architecture", "qwen2")]
    if template is not None:
        fields.append(("tokenizer.chat_template", template))
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields))
    for key, value in fields:
        for text in (key,):
            body += struct.pack("<Q", len(text.encode())) + text.encode()
        body += struct.pack("<I", 8) + struct.pack("<Q", len(value.encode())) + value.encode()
    path.write_bytes(body)
    return path


def _model(path, own: str | None = None):
    return SimpleNamespace(local_path=str(path), chat_template=own)


def test_a_gguf_template_from_its_header(tmp_path) -> None:
    assert model_template(_model(_gguf(tmp_path / "m.gguf", TEMPLATE))) == TEMPLATE


def test_a_gguf_without_one_is_empty(tmp_path) -> None:
    assert model_template(_model(_gguf(tmp_path / "m.gguf", None))) == ""


def test_a_folder_template_file_first_then_tokenizer_config(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "from config"}))
    assert model_template(_model(tmp_path)) == "from config"
    (tmp_path / "chat_template.jinja").write_text(TEMPLATE)
    assert model_template(_model(tmp_path)) == TEMPLATE


def test_named_templates_give_the_default(tmp_path) -> None:
    named = [{"name": "tool_use", "template": "t"}, {"name": "default", "template": "d"}]
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": named}))
    assert model_template(_model(tmp_path)) == "d"


def test_a_created_models_template_wins(tmp_path) -> None:
    assert model_template(_model(_gguf(tmp_path / "m.gguf", TEMPLATE), own="{{ .Prompt }}")) == (
        "{{ .Prompt }}"
    )


def test_an_unreadable_file_is_empty_not_an_error(tmp_path) -> None:
    (tmp_path / "broken.gguf").write_bytes(b"not a gguf")
    assert model_template(_model(tmp_path / "broken.gguf")) == ""
    assert model_template(_model(tmp_path / "missing")) == ""


def test_api_show_returns_it(temp_config, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from hfl.api.server import app
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import get_registry, reset_registry

    reset_registry()
    path = _gguf(temp_config.models_dir / "m.gguf", TEMPLATE)
    get_registry().add(
        ModelManifest(name="m", repo_id="org/m", local_path=str(path), format="gguf")
    )
    body = TestClient(app).post("/api/show", json={"model": "m"}).json()
    assert body["template"] == TEMPLATE

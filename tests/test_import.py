# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl import``: a GGUF you already have, registered where it is.

Checked for real with files outside HFL's home: a single GGUF, the second
part of a split model, a vision model's folder (its projector used for
images), and ``hfl rm`` leaving the file in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hfl.models.importer import ImportRefused, choose_gguf, default_name, manifest_for


def _gguf(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * size)
    return path


def _not_gguf(path: Path) -> Path:
    path.write_bytes(b"NOPE")
    return path


class TestChoose:
    def test_a_file(self, tmp_path):
        model = _gguf(tmp_path / "Qwen3-8B-Q4_K_M.gguf")
        assert choose_gguf(model) == model.resolve()

    def test_a_folder_with_a_model_and_its_projector(self, tmp_path):
        model = _gguf(tmp_path / "vl" / "Qwen2.5-VL-7B-Q4_K_M.gguf")
        _gguf(tmp_path / "vl" / "mmproj-Qwen2.5-VL-7B-f16.gguf")
        assert choose_gguf(tmp_path / "vl") == model.resolve()

    def test_a_split_model_is_its_first_part_whichever_is_given(self, tmp_path):
        first = _gguf(tmp_path / "m-q4_k_m-00001-of-00002.gguf")
        second = _gguf(tmp_path / "m-q4_k_m-00002-of-00002.gguf")
        assert choose_gguf(second) == first.resolve()
        assert choose_gguf(tmp_path) == first.resolve()

    @pytest.mark.parametrize(
        ("setup", "key"),
        [
            (lambda d: d / "missing.gguf", "import.not_found"),
            (lambda d: (d / "empty").mkdir() or d / "empty", "import.no_gguf"),
            (lambda d: _gguf(d / "mmproj-x-f16.gguf"), "import.projector"),
            (lambda d: _not_gguf(d / "x.gguf"), "import.not_gguf"),
            (lambda d: _gguf(d / "m-00002-of-00002.gguf"), "import.no_first_part"),
            (lambda d: _gguf(d / "a.gguf") and _gguf(d / "b.gguf") and d, "import.several"),
        ],
    )
    def test_what_is_refused_and_why(self, tmp_path, setup, key):
        with pytest.raises(ImportRefused) as refused:
            choose_gguf(setup(tmp_path))
        assert refused.value.key == key


def test_the_name_and_the_record(tmp_path):
    first = _gguf(tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00002.gguf", 100)
    _gguf(tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M-00002-of-00002.gguf", 200)
    assert default_name(first) == "qwen2.5-7b-instruct-q4_k_m"
    manifest = manifest_for(first, "q", alias="coder")
    assert manifest.local_path == str(first) and manifest.format == "gguf"
    assert manifest.size_bytes == 104 + 204  # both parts
    assert (manifest.quantization, manifest.alias) == ("Q4_K_M", "coder")


def test_the_command_registers_in_place_and_refuses_a_taken_name(temp_config, tmp_path):
    from hfl.cli.main import app
    from hfl.models.registry import ModelRegistry

    model = _gguf(tmp_path / "elsewhere" / "Phi-4-mini-Q4_K_M.gguf")
    result = CliRunner().invoke(app, ["import", str(model)])
    assert result.exit_code == 0, result.output
    entry = ModelRegistry().get("phi-4-mini-q4_k_m")
    assert entry is not None and entry.local_path == str(model.resolve())
    assert not list(temp_config.models_dir.glob("**/*.gguf"))  # nothing copied
    again = CliRunner().invoke(app, ["import", str(model)])
    assert again.exit_code == 1 and "already exists" in again.output


def test_a_refusal_is_a_message_not_a_traceback(temp_config, tmp_path):
    from hfl.cli.main import app

    result = CliRunner().invoke(app, ["import", str(tmp_path / "nope.gguf")])
    assert result.exit_code == 1 and "Nothing at" in result.output

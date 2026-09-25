# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Which files a GGUF pull needs: all parts of a split model, a vision
model's projector, and never a projector taken for the model.

The file lists are real Hub listings (bartowski/Llama-3.3-70B-Instruct-GGUF,
unsloth/gemma-3-27b-it-GGUF, ggml-org/SmolVLM-256M-Instruct-GGUF), trimmed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hfl.engine.projector import find_projector
from hfl.hub.resolver import _detect_quant, _select_gguf, resolve

LLAMA_70B = [
    "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    "Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf",
    "Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00002-of-00002.gguf",
    "Llama-3.3-70B-Instruct-f16/Llama-3.3-70B-Instruct-f16-00001-of-00004.gguf",
    "Llama-3.3-70B-Instruct-f16/Llama-3.3-70B-Instruct-f16-00002-of-00004.gguf",
    "Llama-3.3-70B-Instruct-f16/Llama-3.3-70B-Instruct-f16-00003-of-00004.gguf",
    "Llama-3.3-70B-Instruct-f16/Llama-3.3-70B-Instruct-f16-00004-of-00004.gguf",
]
GEMMA_27B = [
    "BF16/gemma-3-27b-it-BF16-00001-of-00002.gguf",
    "BF16/gemma-3-27b-it-BF16-00002-of-00002.gguf",
    "gemma-3-27b-it-Q4_K_M.gguf",
    "mmproj-BF16.gguf",
    "mmproj-F16.gguf",
    "mmproj-F32.gguf",
]
VL_Q = ["mmproj-Qwen2.5-VL-f16.gguf", "qwen2.5-vl-7b-f16.gguf", "qwen2.5-vl-7b-Q4_K_M.gguf"]


def _resolve(files: list[str], spec: str):
    info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=f) for f in files],
        pipeline_tag="text-generation",
        sha="abc",
    )
    api = MagicMock()
    api.model_info.return_value = info
    with patch("hfl.hub.resolver.HfApi", return_value=api):
        return resolve(spec)


def test_every_part_of_a_split_model_comes_with_it():
    """Q8_0 of Llama-3.3-70B is two files; one alone does not load."""
    r = _resolve(LLAMA_70B, "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q8_0")
    assert r.filename.endswith("Q8_0-00001-of-00002.gguf")
    assert r.parts == [LLAMA_70B[2]]
    r = _resolve(LLAMA_70B, "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:F16")
    assert r.filename.endswith("-00001-of-00004.gguf") and len(r.parts) == 3
    # Whatever order the Hub lists them in, llama.cpp is handed part 1.
    r = _resolve(LLAMA_70B[::-1], "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:F16")
    assert r.filename.endswith("-00001-of-00004.gguf") and len(r.parts) == 3


def test_a_single_file_has_no_parts_and_a_text_model_no_projector():
    r = _resolve(LLAMA_70B, "hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF")
    assert (r.filename, r.parts, r.projector) == (LLAMA_70B[0], [], None)


def test_a_vision_model_brings_its_projector():
    r = _resolve(GEMMA_27B, "hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M")
    assert (r.filename, r.projector) == ("gemma-3-27b-it-Q4_K_M.gguf", "mmproj-F16.gguf")


def test_a_projector_is_never_taken_for_the_model():
    """``mmproj-...-f16`` sorts before ``qwen...-f16``: asked for F16, the
    model used to be the projector."""
    assert _select_gguf([f for f in VL_Q if "mmproj" not in f], "F16") == VL_Q[1]
    r = _resolve(VL_Q, "hf.co/x/qwen2.5-vl-7b-GGUF:F16")
    assert (r.filename, r.projector) == (VL_Q[1], VL_Q[0])


@pytest.mark.parametrize(
    ("filename", "quant"),
    [("BF16/gemma-3-27b-it-BF16-00001-of-00002.gguf", "BF16"), ("m-f16.gguf", "F16")],
)
def test_bf16_is_not_f16(filename, quant):
    assert _detect_quant(filename) == quant


def test_asked_for_f16_an_f16_file_wins_over_bf16():
    files = ["m-BF16.gguf", "m-F16.gguf"]
    assert _select_gguf(files, "F16") == "m-F16.gguf"


class TestFindProjector:
    def test_beside_the_model_f16_first(self, tmp_path):
        for name in ("m-Q4_K_M.gguf", "mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"):
            (tmp_path / name).write_bytes(b"GGUF")
        assert find_projector(tmp_path / "m-Q4_K_M.gguf") == tmp_path / "mmproj-F16.gguf"

    def test_a_text_model_has_none(self, tmp_path):
        (tmp_path / "m.gguf").write_bytes(b"GGUF")
        assert find_projector(tmp_path / "m.gguf") is None

    def test_a_split_model_finds_the_one_at_its_repo_root(self, temp_config):
        repo = temp_config.models_dir / "unsloth--gemma-3-27b-it-GGUF"
        (repo / "BF16").mkdir(parents=True)
        model = repo / "BF16" / "gemma-3-27b-it-BF16-00001-of-00002.gguf"
        model.write_bytes(b"GGUF")
        (repo / "mmproj-F16.gguf").write_bytes(b"GGUF")
        assert find_projector(model) == repo / "mmproj-F16.gguf"

    def test_never_outside_the_models_folder(self, tmp_path, temp_config):
        """A GGUF registered from elsewhere must not pick up a stranger's
        projector a folder up."""
        (tmp_path / "downloads" / "sub").mkdir(parents=True)
        (tmp_path / "downloads" / "mmproj-F16.gguf").write_bytes(b"GGUF")
        model = tmp_path / "downloads" / "sub" / "m.gguf"
        model.write_bytes(b"GGUF")
        assert find_projector(model) is None


def test_the_pull_downloads_the_parts_and_the_projector(temp_config, monkeypatch):
    from hfl.hub import downloader
    from hfl.hub.resolver import ResolvedModel

    fetched: list[str] = []
    monkeypatch.setattr(downloader, "ensure_auth", lambda repo: None)
    monkeypatch.setattr(downloader, "_rate_limit", lambda: None)
    monkeypatch.setattr(
        downloader,
        "_download_file",
        lambda repo_id, filename, revision, local_dir, token: (
            fetched.append(filename) or local_dir / filename
        ),
    )
    resolved = ResolvedModel(
        repo_id="unsloth/gemma-3-27b-it-GGUF",
        filename=GEMMA_27B[0],
        format="gguf",
        parts=[GEMMA_27B[1]],
        projector="mmproj-F16.gguf",
    )
    path = downloader.pull_model(resolved)
    assert fetched == [GEMMA_27B[0], GEMMA_27B[1], "mmproj-F16.gguf"]
    assert path.name == "gemma-3-27b-it-BF16-00001-of-00002.gguf"

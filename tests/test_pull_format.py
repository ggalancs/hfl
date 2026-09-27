# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl pull --format gguf`` converts even where MLX would serve the
safetensors (it was ignored on Apple Silicon, and ``-q`` with it), and a
conversion that cannot run says why instead of a traceback (audit E10)."""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hfl.exceptions import ToolNotFoundError


def _pull(temp_config, monkeypatch, args: list[str], convert=None):
    from hfl.cli.main import app
    from hfl.hub.resolver import ResolvedModel

    target = temp_config.models_dir / "HuggingFaceTB--SmolLM2-135M-Instruct"
    target.mkdir(parents=True)
    (target / "model.safetensors").write_bytes(b"x")
    (target / "config.json").write_bytes(b"{}")
    resolved = ResolvedModel(
        repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        format="safetensors",
        quantization="Q4_K_M",
        pipeline_tag="text-generation",
    )
    calls: list[tuple] = []

    def fake_convert(self, source, output, quantize):
        calls.append((source, output, quantize))
        if convert is not None:
            return convert(output)
        output.with_suffix(".gguf").write_bytes(b"GGUF")
        return output.with_suffix(".gguf")

    monkeypatch.setattr("hfl.converter.formats.is_mlx_quantized_repo", lambda *a: False)
    monkeypatch.setattr("hfl.engine.selector._mlx_preferred", lambda: True)
    monkeypatch.setattr(
        "hfl.converter.gguf_converter.check_model_convertibility", lambda p: (True, "")
    )
    monkeypatch.setattr("hfl.converter.gguf_converter.GGUFConverter.convert", fake_convert)
    with patch("hfl.hub.resolver.resolve", return_value=resolved):
        with patch("hfl.hub.downloader.pull_model", return_value=target):
            result = CliRunner().invoke(app, ["pull", "x", "--skip-license", *args])
    return result, calls


def test_explicit_gguf_is_converted_even_where_mlx_serves_safetensors(
    temp_config, monkeypatch
) -> None:
    result, calls = _pull(temp_config, monkeypatch, ["--format", "gguf", "-q", "Q5_K_M"])
    assert result.exit_code == 0, result.stdout
    assert len(calls) == 1 and calls[0][2] == "Q5_K_M"


def test_auto_keeps_safetensors_for_mlx(temp_config, monkeypatch) -> None:
    result, calls = _pull(temp_config, monkeypatch, [])
    assert result.exit_code == 0, result.stdout
    assert calls == []


def test_a_conversion_that_cannot_run_says_why(temp_config, monkeypatch) -> None:
    def missing_cmake(output):
        raise ToolNotFoundError("cmake", "Install it with brew.")

    result, _ = _pull(temp_config, monkeypatch, ["--format", "gguf"], convert=missing_cmake)
    assert result.exit_code == 1
    assert "cmake is not installed" in result.stdout
    assert "Traceback" not in result.stdout
    assert not isinstance(result.exception, ToolNotFoundError)


def _converter(temp_config, monkeypatch, *, which, llama_cpp=False):
    from hfl.converter.gguf_converter import GGUFConverter

    monkeypatch.setattr("hfl.converter.gguf_converter.config", temp_config)
    monkeypatch.setattr("hfl.converter.gguf_converter.shutil.which", which)
    monkeypatch.setattr(
        "hfl.converter.gguf_converter.subprocess.run",
        lambda cmd, **k: types.SimpleNamespace(returncode=0 if llama_cpp else 1),
    )
    return GGUFConverter()


def test_a_quantizer_built_here_before_comes_first(temp_config, monkeypatch) -> None:
    converter = _converter(temp_config, monkeypatch, which=lambda t: f"/opt/{t}")
    converter.quantize_bin.parent.mkdir(parents=True)
    converter.quantize_bin.write_text("")
    assert converter._quantizer() == [str(converter.quantize_bin)]


def test_then_llama_quantize_on_the_path(temp_config, monkeypatch) -> None:
    """Homebrew's llama.cpp ships it: no build needed (it used to need cmake)."""
    converter = _converter(temp_config, monkeypatch, which=lambda t: f"/opt/{t}")
    assert converter._quantizer() == ["/opt/llama-quantize"]


def test_then_llama_cpp_pythons_own_quantizer(temp_config, monkeypatch) -> None:
    from hfl.converter.gguf_converter import _LLAMA_CPP_QUANTIZE

    converter = _converter(temp_config, monkeypatch, which=lambda t: None, llama_cpp=True)
    assert converter._quantizer()[-1] == _LLAMA_CPP_QUANTIZE


def test_with_none_and_no_cmake_it_says_how_to_get_one(temp_config, monkeypatch) -> None:
    converter = _converter(temp_config, monkeypatch, which=lambda t: None)
    with pytest.raises(ToolNotFoundError) as caught:
        converter._quantizer()
    assert "brew install llama.cpp" in str(caught.value.details)
    assert "hfl[llama]" in str(caught.value.details)


def test_the_source_archive_is_extracted_safely(tmp_path, monkeypatch) -> None:
    """Without git the converter comes from llama.cpp's archive: the whole
    tree (the script imports its ``conversion`` package), but no member that
    would land outside the target, and no links."""
    import io
    import tarfile
    from contextlib import contextmanager

    from hfl.converter import gguf_converter

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:

        def add(name: str, data: bytes = b"x", kind: bytes = tarfile.REGTYPE) -> None:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = "/etc/passwd"
            tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)

        add("llama.cpp-master/convert_hf_to_gguf.py", b"print(1)")
        add("llama.cpp-master/conversion/__init__.py")
        add("llama.cpp-master/gguf-py/gguf/__init__.py")
        add("llama.cpp-master/../../escape.py")
        add("llama.cpp-master/link", kind=tarfile.SYMTYPE)

    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_bytes(self):
            yield raw.getvalue()

    @contextmanager
    def stream(*a, **k):
        yield Response()

    monkeypatch.setattr("httpx.stream", stream)
    target = tmp_path / "tools" / "llama.cpp"
    target.parent.mkdir()
    gguf_converter._download_converter(target)
    assert (target / "convert_hf_to_gguf.py").read_text() == "print(1)"
    assert (target / "conversion" / "__init__.py").exists()
    assert (target / "gguf-py" / "gguf" / "__init__.py").exists()
    assert not (target / "link").exists()
    assert list(tmp_path.rglob("escape.py")) == []  # nowhere, not beside the target either

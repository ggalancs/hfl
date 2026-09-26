# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl pull --format gguf`` converts even where MLX would serve the
safetensors (it was ignored on Apple Silicon, and ``-q`` with it), and a
conversion that cannot run says why instead of a traceback (audit E10)."""

from __future__ import annotations

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


def test_ensure_tools_checks_for_cmake_before_building(temp_config, monkeypatch) -> None:
    from hfl.converter.gguf_converter import GGUFConverter

    converter = GGUFConverter()
    monkeypatch.setattr(
        "hfl.converter.gguf_converter.shutil.which",
        lambda tool: None if tool == "cmake" else f"/usr/bin/{tool}",
    )
    ran: list = []
    monkeypatch.setattr(
        "hfl.converter.gguf_converter.subprocess.run", lambda *a, **k: ran.append(a)
    )
    with pytest.raises(ToolNotFoundError) as caught:
        converter.ensure_tools()
    assert caught.value.tool_name == "cmake"
    assert "brew install cmake" in str(caught.value.details)
    assert ran == []  # nothing cloned or built before the check

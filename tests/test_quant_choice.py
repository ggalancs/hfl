# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The GGUF level of a conversion: the most precise one that fits.

``hfl pull`` of a safetensors LLM converted it at a fixed Q4_K_M: a model
that fits in F16 lost precision for nothing, and one that fits nowhere was
downloaded only to fail at load."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from hfl.hub.hw_profile import HardwareProfile
from hfl.hub.quant_choice import choose

MAC = HardwareProfile(
    os="darwin", arch="arm64", system_ram_gb=128.0, gpu_kind="metal", gpu_vram_gb=89.6,
    has_mlx=True, has_cuda=False, has_rocm=False,
)  # fmt: skip
L4 = replace(MAC, os="linux", arch="x86_64", system_ram_gb=64.0, gpu_kind="cuda", gpu_vram_gb=24.0)
CPU = replace(MAC, os="linux", arch="x86_64", system_ram_gb=16.0, gpu_kind="none", gpu_vram_gb=None)


class TestChoose:
    def test_a_small_model_keeps_full_precision(self) -> None:
        assert choose(3, L4).recommended == "F16"

    def test_the_most_precise_level_that_fits(self) -> None:
        choice = choose(14, L4)
        assert choice.recommended == "Q8_0" and not choice.split
        f16 = next(level for level in choice.levels if level.name == "F16")
        assert not f16.fits  # it would not have fitted

    def test_a_gpu_too_small_splits_with_ram(self) -> None:
        choice = choose(70, L4)
        assert choice.recommended == "Q4_K_M" and choice.split

    def test_nothing_fits_says_how_much_is_needed(self) -> None:
        choice = choose(235, L4)
        assert choice.recommended is None and choice.needed_gb > choice.total_gb

    def test_q2_is_never_recommended(self) -> None:
        # 14B on 16 GB of RAM (11.2 GB for the model): Q2_K (~10.8 GB)
        # fits, Q3_K_M (~12.6 GB) does not.
        choice = choose(14, CPU)
        q2 = next(level for level in choice.levels if level.name == "Q2_K")
        assert q2.fits and choice.recommended is None

    def test_apple_silicon_s_share_is_not_cut_twice(self) -> None:
        # The profile's gpu_vram_gb is already the GPU's share of memory.
        assert choose(32, MAC).fast_gb == 89.6 and choose(32, MAC).recommended == "F16"


def _pull(temp_config, monkeypatch, args, *, params_b, profile, mlx=False, answer=None):
    from hfl.cli.main import app
    from hfl.hub.resolver import ResolvedModel

    target = temp_config.models_dir / "org--model"
    target.mkdir(parents=True)
    (target / "model.safetensors").write_bytes(b"x")
    (target / "config.json").write_bytes(b"{}")
    resolved = ResolvedModel(  # the resolver names a level even for safetensors
        repo_id="org/model",
        format="safetensors",
        quantization="Q4_K_M",
        pipeline_tag="text-generation",
    )
    levels: list[str] = []
    downloads: list[str] = []

    def fake_convert(self, source, output, quantize):
        levels.append(quantize)
        output.with_suffix(".gguf").write_bytes(b"GGUF")
        return output.with_suffix(".gguf")

    def fake_download(resolved_model, *a, **k):
        downloads.append(resolved_model.repo_id)
        return target

    monkeypatch.setattr("hfl.engine.selector._mlx_preferred", lambda: mlx)
    monkeypatch.setattr("hfl.converter.formats.is_mlx_quantized_repo", lambda *a: False)
    monkeypatch.setattr(
        "hfl.converter.gguf_converter.check_model_convertibility", lambda p: (True, "")
    )
    monkeypatch.setattr("hfl.converter.gguf_converter.GGUFConverter.convert", fake_convert)
    monkeypatch.setattr(
        "hfl.hub.params.estimate_params",
        lambda repo, api=None: SimpleNamespace(total_b=params_b, active_b=None),
    )
    monkeypatch.setattr("hfl.hub.hw_profile.get_hw_profile", lambda: profile)
    monkeypatch.setattr("hfl.cli.main.stdin_is_terminal", lambda: answer is not None)
    with (
        patch("hfl.hub.resolver.resolve", return_value=resolved),
        patch("hfl.hub.downloader.pull_model", side_effect=fake_download),
    ):
        result = CliRunner().invoke(
            app, ["pull", "org/model", "--skip-license", *args], input=answer
        )
    return result, levels, downloads


class TestPull:
    def test_without_q_it_converts_at_the_level_that_fits(self, temp_config, monkeypatch):
        result, levels, _ = _pull(temp_config, monkeypatch, ["--yes"], params_b=14, profile=L4)
        assert result.exit_code == 0, result.output
        assert levels == ["Q8_0"] and "Q8_0" in result.output
        # Registered at the level it was converted to, not the resolver's
        # (an F16 conversion was named and labelled Q4_K_M, measured).
        from hfl.models.registry import ModelRegistry

        manifest = ModelRegistry().get("model-q8_0")
        assert manifest is not None and manifest.quantization == "Q8_0"

    def test_a_typed_level_is_taken(self, temp_config, monkeypatch):
        result, levels, _ = _pull(
            temp_config, monkeypatch, [], params_b=14, profile=L4, answer="q6_k\n"
        )
        assert result.exit_code == 0, result.output
        assert levels == ["Q6_K"]

    def test_enter_takes_the_recommended_one(self, temp_config, monkeypatch):
        result, levels, _ = _pull(temp_config, monkeypatch, [], params_b=3, profile=L4, answer="\n")
        assert result.exit_code == 0, result.output
        assert levels == ["F16"]

    def test_an_explicit_q_is_respected(self, temp_config, monkeypatch):
        result, levels, _ = _pull(
            temp_config, monkeypatch, ["-q", "Q4_K_M"], params_b=3, profile=L4
        )
        assert result.exit_code == 0, result.output
        assert levels == ["Q4_K_M"]

    def test_nothing_fits_is_refused_before_downloading(self, temp_config, monkeypatch):
        result, levels, downloads = _pull(
            temp_config, monkeypatch, ["--yes"], params_b=235, profile=L4
        )
        assert result.exit_code == 1 and downloads == [] and levels == []
        assert "GB" in result.output

    def test_mlx_keeps_a_model_that_fits_in_its_own_precision(self, temp_config, monkeypatch):
        result, levels, _ = _pull(
            temp_config, monkeypatch, ["--yes"], params_b=7, profile=MAC, mlx=True
        )
        assert result.exit_code == 0, result.output
        assert levels == []  # kept for MLX, as before

    def test_mlx_converts_a_model_that_does_not_fit(self, temp_config, monkeypatch):
        result, levels, _ = _pull(
            temp_config, monkeypatch, ["--yes"], params_b=70, profile=MAC, mlx=True
        )
        assert result.exit_code == 0, result.output
        assert levels == ["Q6_K"]


def _safetensors(directory, gigabytes: float, config: str = "{}"):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "model.safetensors", "wb") as f:
        f.truncate(int(gigabytes * 1e9))  # sparse: the size, not the bytes
    (directory / "config.json").write_text(config)
    return directory


class TestTheMemoryError:
    """A safetensors model too big for this machine: the error names the
    command that makes it fit (it said "a smaller quantization", which a
    safetensors model cannot take without converting)."""

    def test_names_the_conversion_that_fits(self, tmp_path, monkeypatch) -> None:
        from hfl.hub.quant_choice import conversion_hint

        monkeypatch.setattr("hfl.hub.hw_profile.get_hw_profile", lambda: L4)
        folder = _safetensors(tmp_path / "m", 28.0)  # 14B in 16 bits
        manifest = SimpleNamespace(format="safetensors", local_path=str(folder), repo_id="org/m")
        hint = conversion_hint(manifest)
        assert "hfl pull org/m --format gguf -q Q8_0" in hint
        assert "HFL_TRANSFORMERS_QUANT=8bit" in hint  # on NVIDIA only

    def test_off_nvidia_no_bitsandbytes(self, tmp_path, monkeypatch) -> None:
        from hfl.hub.quant_choice import conversion_hint

        monkeypatch.setattr("hfl.hub.hw_profile.get_hw_profile", lambda: MAC)
        folder = _safetensors(tmp_path / "m", 140.0)  # 70B
        hint = conversion_hint(
            SimpleNamespace(format="safetensors", local_path=str(folder), repo_id="org/m")
        )
        assert "-q Q6_K" in hint and "bitsandbytes" not in hint

    def test_a_gguf_gets_none(self, tmp_path) -> None:
        from hfl.hub.quant_choice import conversion_hint

        assert conversion_hint(SimpleNamespace(format="gguf", local_path=str(tmp_path))) is None

    def test_the_error_carries_it(self) -> None:
        from hfl.exceptions import MemoryBudgetExceededError

        plan = SimpleNamespace(reason="too_big", constraint="ram", floor=0, limit=0)
        err = MemoryBudgetExceededError(
            "m", needed=2**40, plan=plan, total=0, budget=0.85, remedy="hfl pull x -q Q8_0."
        )
        assert "hfl pull x -q Q8_0." in str(err.details)

    def test_working_it_out_never_breaks_the_error(self, monkeypatch) -> None:
        from hfl.api import state

        def boom():
            raise RuntimeError("registry gone")

        monkeypatch.setattr("hfl.core.get_registry", boom)
        assert state._fitting_remedy("m") is None


def _with_bitsandbytes(monkeypatch, present: bool = True) -> None:
    import importlib.util

    real = importlib.util.find_spec

    def find_spec(name, *args):
        if name == "bitsandbytes":
            return object() if present else None
        return real(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


class TestBitsandbytesSetting:
    """HFL_TRANSFORMERS_QUANT: the bitsandbytes loader existed but nothing
    reached it. Explicit, NVIDIA only, and planned at its real size."""

    def test_without_bitsandbytes_it_is_ignored(self, tmp_path, monkeypatch) -> None:
        from hfl.config import config
        from hfl.engine.footprint import estimate_footprint
        from hfl.engine.transformers_engine import _configured_quant

        _with_bitsandbytes(monkeypatch, present=False)
        folder = _safetensors(tmp_path / "m", 10.0)
        full = estimate_footprint(folder).weights_bytes
        monkeypatch.setattr(config, "transformers_quant", "4bit")
        assert _configured_quant() is None and estimate_footprint(folder).weights_bytes == full

    def test_the_footprint_is_planned_quantized(self, tmp_path, monkeypatch) -> None:
        from hfl.config import config
        from hfl.engine.footprint import estimate_footprint

        _with_bitsandbytes(monkeypatch)
        folder = _safetensors(tmp_path / "m", 10.0)
        full = estimate_footprint(folder).weights_bytes
        monkeypatch.setattr(config, "transformers_quant", "4bit")
        assert estimate_footprint(folder).weights_bytes == int(full * 0.35)

    def test_an_mlx_folder_is_not_scaled(self, tmp_path, monkeypatch) -> None:
        from hfl.config import config
        from hfl.engine.footprint import estimate_footprint

        _with_bitsandbytes(monkeypatch)
        mlx = '{"quantization": {"group_size": 64, "bits": 4}}'
        folder = _safetensors(tmp_path / "m", 4.0, config=mlx)
        full = estimate_footprint(folder).weights_bytes
        monkeypatch.setattr(config, "transformers_quant", "4bit")
        assert estimate_footprint(folder).weights_bytes == full

    def test_only_8bit_and_4bit_are_taken(self, monkeypatch) -> None:
        from hfl.config import config
        from hfl.engine.transformers_engine import _configured_quant

        _with_bitsandbytes(monkeypatch)
        for value, expected in (("8bit", "8bit"), ("4BIT", "4bit"), ("none", None), ("2bit", None)):
            monkeypatch.setattr(config, "transformers_quant", value)
            assert _configured_quant() == expected, value


def test_unmeasured_memory_converts_as_before(temp_config, monkeypatch) -> None:
    """No memory reading (psutil absent: 0 GB) must not refuse every pull:
    it converts at Q4_K_M, as before the choice existed."""
    zero = replace(CPU, system_ram_gb=0.0)
    result, levels, downloads = _pull(temp_config, monkeypatch, ["--yes"], params_b=7, profile=zero)
    assert result.exit_code == 0, result.output
    assert levels == ["Q4_K_M"] and downloads

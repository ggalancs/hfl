# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Conversion of HuggingFace models (safetensors/pytorch) to GGUF format.

This is the most critical step of the pipeline. It uses llama.cpp tools
for conversion and quantization.

Flow:
  safetensors -> convert_hf_to_gguf.py (FP16) -> quantize -> final GGUF

Requires: llama.cpp cloned and compiled in ~/.hfl/tools/llama.cpp

LEGAL NOTE (R3 - Legal Audit):
Format conversion preserves the weights of the original model.
The license and restrictions of the original model remain in effect
on the converted file. hfl records the provenance of each conversion
for legal compliance.

Models supported for GGUF conversion:
- Text models (LLMs) with architectures supported by llama.cpp
- Require config.json with a valid model_type

Unsupported models:
- LoRA adapters (adapter_*.safetensors files without base model)
- Image models (Stable Diffusion, FLUX, etc.)
- Models without config.json
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from hfl.config import config

console = Console()


class UnsupportedModelError(Exception):
    """Raised when a model cannot be converted to GGUF format."""

    pass


# Model types that CANNOT be converted to GGUF
UNSUPPORTED_MODEL_TYPES = {
    # Image models
    "stable-diffusion",
    "sdxl",
    "flux",
    "vae",
    "unet",
    "controlnet",
    # LoRA adapters
    "lora",
    "adapter",
    # Others
    "clip",
    "vit",
    "audio",
    "whisper",
}

# File patterns that indicate non-convertible models
UNSUPPORTED_FILE_PATTERNS = {
    "adapter_model.safetensors",  # LoRA adapter
    "adapter_config.json",  # LoRA config
    "diffusion_pytorch_model.safetensors",  # Diffusion
    "unet/",  # Stable Diffusion UNet
    "vae/",  # VAE
}


def check_model_convertibility(model_path: Path) -> tuple[bool, str]:
    """
    Checks if a model can be converted to GGUF format.

    Args:
        model_path: Path to the downloaded model directory

    Returns:
        Tuple (is_convertible, reason)
        - (True, "") if convertible
        - (False, reason) if not convertible
    """
    # 1. Check that config.json exists
    config_path = model_path / "config.json"
    if not config_path.exists():
        # Check if it's a LoRA adapter
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            return (
                False,
                "This is a LoRA adapter, not a complete model. "
                "LoRA adapters require a base model to function.",
            )

        # Check if there are diffusion files
        for pattern in UNSUPPORTED_FILE_PATTERNS:
            if (model_path / pattern).exists() or list(model_path.glob(f"*{pattern}*")):
                return (
                    False,
                    "This appears to be an image diffusion model. "
                    "GGUF only supports text models (LLMs).",
                )

        return (
            False,
            "config.json not found. This model does not have the standard "
            "HuggingFace format for text models.",
        )

    # 2. Read config.json and verify model_type
    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return (False, f"Could not read config.json: {e}")

    model_type = config_data.get("model_type", "").lower()

    if not model_type:
        # Without model_type, check other indicators
        if "adapter_config" in config_data or "_name_or_path" in str(config_data.get("base_model")):
            return (
                False,
                "This is a LoRA adapter. LoRA adapters require "
                "a base model to function.",
            )
        return (
            False,
            "config.json does not contain 'model_type'. "
            "The model cannot be identified.",
        )

    # 3. Check if the model_type is in the unsupported list
    for unsupported in UNSUPPORTED_MODEL_TYPES:
        if unsupported in model_type:
            return (
                False,
                f"The model type '{model_type}' is not supported for GGUF conversion. "
                "GGUF only supports text models (LLMs).",
            )

    # 4. The model appears to be convertible
    return (True, "")


def _get_llama_cpp_version(llama_cpp_dir: Path) -> str:
    """Gets the version/commit of the installed llama.cpp."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=llama_cpp_dir,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class GGUFConverter:
    """Manages the conversion of models to GGUF format."""

    def __init__(self):
        self.llama_cpp_dir = config.llama_cpp_dir
        self.convert_script = self.llama_cpp_dir / "convert_hf_to_gguf.py"
        self.quantize_bin = self.llama_cpp_dir / "build" / "bin" / "llama-quantize"

    def ensure_tools(self):
        """
        Verifies that llama.cpp is available.
        If not, clones and compiles it automatically.
        """
        if self.convert_script.exists() and self.quantize_bin.exists():
            return

        console.print("[yellow]Installing conversion tools (llama.cpp)...[/]")

        if not self.llama_cpp_dir.exists():
            self.llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "https://github.com/ggml-org/llama.cpp.git",
                    str(self.llama_cpp_dir),
                ],
                check=True,
            )

        # Compile llama.cpp
        build_dir = self.llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)

        # Detect if CUDA is available
        cuda_flag = "-DGGML_CUDA=ON" if shutil.which("nvcc") else ""

        cmake_cmd = ["cmake", ".."]
        if cuda_flag:
            cmake_cmd.append(cuda_flag)

        subprocess.run(cmake_cmd, cwd=build_dir, check=True)
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j"],
            cwd=build_dir,
            check=True,
        )

        # Install Python dependencies for the conversion script
        requirements_file = self.llama_cpp_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
            )

        console.print("[green]Conversion tools ready.[/]")

    def convert(
        self,
        model_path: Path,
        output_path: Path,
        quantization: str = "Q4_K_M",
        source_repo: str = "",
        original_license: str = "",
        license_accepted: bool = False,
    ) -> Path:
        """
        Converts an HF model to quantized GGUF.

        Args:
            model_path: Path to the HF model directory (with config.json)
            output_path: Base path for the output file
            quantization: Quantization level (Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16)
            source_repo: Original HuggingFace repository (for provenance)
            original_license: License of the original model
            license_accepted: Whether the user accepted the license

        Returns:
            Path to the final GGUF file.
        """
        self.ensure_tools()

        # R3 - Legal warning about license preservation
        console.print(
            "\n[yellow]Note:[/] Format conversion preserves the weights of the original "
            "model. The license and restrictions of the original model remain "
            "in effect on the converted file.\n"
        )

        # Step 1: Convert to GGUF FP16 (intermediate format)
        fp16_path = output_path.with_suffix(".fp16.gguf")

        console.print("[cyan]Step 1/2:[/] Converting to GGUF FP16...")

        subprocess.run(
            [
                sys.executable,
                str(self.convert_script),
                str(model_path),
                "--outtype",
                "f16",
                "--outfile",
                str(fp16_path),
            ],
            check=True,
        )

        if quantization.upper() == "F16":
            # If FP16 is requested, we're done
            final_path = output_path.with_suffix(".f16.gguf")
            fp16_path.rename(final_path)
            return final_path

        # Step 2: Quantize to the requested level
        quant = quantization.upper()
        final_path = output_path.with_suffix(f".{quant}.gguf")

        console.print(f"[cyan]Step 2/2:[/] Quantizing to {quant}...")

        subprocess.run(
            [
                str(self.quantize_bin),
                str(fp16_path),
                str(final_path),
                quant,
            ],
            check=True,
        )

        # Clean up intermediate FP16
        fp16_path.unlink(missing_ok=True)

        # R3 - Record conversion provenance
        if source_repo:
            try:
                from hfl.converter.formats import detect_format
                from hfl.models.provenance import log_conversion

                source_format = detect_format(model_path).value
                tool_version = _get_llama_cpp_version(self.llama_cpp_dir)

                log_conversion(
                    source_repo=source_repo,
                    source_format=source_format,
                    target_path=str(final_path),
                    quantization=quant,
                    original_license=original_license,
                    license_accepted=license_accepted,
                    tool_version=tool_version,
                    notes=f"Converted using hfl from {model_path}",
                )
            except Exception as e:
                console.print(f"[dim]Warning: Could not record provenance: {e}[/]")

        console.print(f"[green]Conversion completed:[/] {final_path}")
        return final_path


# Quick reference for quantization levels:
#
# Level       | Bits/weight | % Quality | Use case
# ------------|-------------|-----------|----------------------------------
# Q2_K        | ~2.5        | ~80%      | Extreme compression, low quality
# Q3_K_M      | ~3.5        | ~87%      | Low RAM, acceptable quality
# Q4_K_M      | ~4.5        | ~92%      | * DEFAULT - best balance
# Q5_K_M      | ~5.0        | ~96%      | High quality, more RAM
# Q6_K        | ~6.5        | ~97%      | Premium, almost no loss
# Q8_0        | ~8.0        | ~98%+     | Maximum quantized quality
# F16         | 16.0        | 100%      | No quantization

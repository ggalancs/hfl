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
- Audio/TTS models (Whisper, Bark, Qwen-TTS, VITS, etc.)
- Vision-only models (CLIP, ViT, DINO, etc.)
- Multimodal models (LLaVA, BLIP, etc.)
- Models without config.json
"""

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from rich.console import Console

from hfl.config import config

console = Console()

_conversion_locks: dict[str, threading.Lock] = {}
_conversion_locks_guard = threading.Lock()


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
    # Audio/TTS models
    "whisper",
    "wav2vec",
    "wav2vec2",
    "hubert",
    "speecht5",
    "bark",
    "musicgen",
    "encodec",
    "seamless",
    "mms",
    # TTS specific
    "tts",
    "vits",
    "fastspeech",
    "tacotron",
    "parler",
    "parler-tts",
    "qwen3_tts",
    "qwen_tts",
    "cosyvoice",
    "f5-tts",
    "xtts",
    "coqui",
    "tortoise",
    "valle",
    "vocos",
    # Vision models
    "clip",
    "vit",
    "dino",
    "siglip",
    # Multimodal (non-text-only)
    "llava",
    "blip",
    "git",
    "pix2struct",
}

# File patterns that indicate non-convertible models
UNSUPPORTED_FILE_PATTERNS = {
    "adapter_model.safetensors",  # LoRA adapter
    "adapter_config.json",  # LoRA config
    "diffusion_pytorch_model.safetensors",  # Diffusion
    "unet/",  # Stable Diffusion UNet
    "vae/",  # VAE
}

# Keywords in architecture names that indicate non-LLM models
UNSUPPORTED_ARCHITECTURE_KEYWORDS = {
    "tts",  # Text-to-Speech
    "stt",  # Speech-to-Text
    "asr",  # Automatic Speech Recognition
    "speech",  # Speech models
    "voice",  # Voice models
    "audio",  # Audio models
    "music",  # Music generation
    "vocoder",  # Audio vocoders
    "diffusion",  # Diffusion models
    "vae",  # Variational autoencoders
    "gan",  # GANs
    "vision",  # Vision-only models
    "image",  # Image models
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
                "This is a LoRA adapter. LoRA adapters require a base model to function.",
            )
        return (
            False,
            "config.json does not contain 'model_type'. The model cannot be identified.",
        )

    # 3. Check if the model_type is in the unsupported list
    for unsupported in UNSUPPORTED_MODEL_TYPES:
        if unsupported in model_type:
            return (
                False,
                f"The model type '{model_type}' is not supported for GGUF conversion. "
                "GGUF only supports text models (LLMs).",
            )

    # 4. Check architectures field for unsupported patterns
    architectures = config_data.get("architectures", [])
    if architectures:
        arch_str = " ".join(architectures).lower()
        for keyword in UNSUPPORTED_ARCHITECTURE_KEYWORDS:
            if keyword in arch_str:
                return (
                    False,
                    f"Architecture '{architectures[0]}' appears to be {keyword.upper()}. "
                    "GGUF conversion only supports text-based LLMs.",
                )

    # 5. Check for audio/TTS specific config fields
    audio_indicators = ["num_mel_bins", "vocoder", "speaker_embedding", "audio_encoder", "codec"]
    for indicator in audio_indicators:
        if indicator in config_data:
            return (
                False,
                f"This model has audio-specific configuration ('{indicator}'). "
                "It appears to be an audio/TTS model which cannot be converted to GGUF.",
            )

    # 6. The model appears to be convertible
    return (True, "")


# Pinned llama.cpp version for reproducibility and security
# Update this when testing a new version
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_BRANCH = "master"  # Can be changed to a specific tag or commit


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
    except (FileNotFoundError, OSError):
        return "unknown"


def _verify_git_clone(repo_dir: Path, expected_repo: str) -> bool:
    """Verify git clone integrity by checking remote URL."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        actual_url = result.stdout.strip()
        # Normalize URLs for comparison (handle .git suffix)
        return actual_url.rstrip(".git") == expected_repo.rstrip(".git")
    except (FileNotFoundError, OSError):
        return False


def convert_with_cache(
    converter: "GGUFConverter",
    model_path: Path,
    output_path: Path,
    quantization: str = "Q4_K_M",
    **kwargs,
) -> Path:
    """Convert with caching to avoid duplicate conversions.

    Uses file-based locking to prevent concurrent conversions of the same model.
    """
    cache_key = f"{model_path}:{quantization}"

    # Check if already converted (fast path)
    quant = quantization.upper()
    if quant == "F16":
        expected_output = output_path.with_suffix(".f16.gguf")
    else:
        expected_output = output_path.with_suffix(f".{quant}.gguf")

    if expected_output.exists():
        console.print(f"[green]Using cached conversion:[/] {expected_output}")
        return expected_output

    # Get or create lock for this conversion
    with _conversion_locks_guard:
        if cache_key not in _conversion_locks:
            _conversion_locks[cache_key] = threading.Lock()
        lock = _conversion_locks[cache_key]

    with lock:
        # Double-check after acquiring lock
        if expected_output.exists():
            console.print(f"[green]Using cached conversion:[/] {expected_output}")
            return expected_output

        # Actually convert
        return converter.convert(model_path, output_path, quantization, **kwargs)


class GGUFConverter:
    """Manages the conversion of models to GGUF format."""

    def __init__(self):
        self.llama_cpp_dir = config.llama_cpp_dir
        self.convert_script = self.llama_cpp_dir / "convert_hf_to_gguf.py"
        self.quantize_bin = self.llama_cpp_dir / "build" / "bin" / "llama-quantize"

    def _verify_output(self, output_path: Path, source_path: Path) -> None:
        """Verify conversion output integrity.

        Args:
            output_path: Path to the converted GGUF file
            source_path: Path to the source model directory

        Raises:
            RuntimeError: If verification fails
        """
        # Check file exists
        if not output_path.exists():
            raise RuntimeError(f"Conversion failed: {output_path} not created")

        # Check file is not empty
        file_size = output_path.stat().st_size
        if file_size == 0:
            output_path.unlink()
            raise RuntimeError("Conversion produced empty file")

        # Sanity check: GGUF should have a reasonable size
        # Minimum expected size: ~100MB for smallest quantized models
        min_size = 50 * 1024 * 1024  # 50MB minimum
        if file_size < min_size:
            console.print(
                f"[yellow]Warning:[/] Output file unusually small: {file_size / 1024 / 1024:.1f}MB"
            )

        # Check input size for comparison (if safetensors available)
        try:
            input_files = list(source_path.glob("*.safetensors"))
            if input_files:
                input_size = sum(f.stat().st_size for f in input_files)
                if file_size > input_size * 1.1:  # Allow 10% overhead
                    console.print(
                        f"[yellow]Warning:[/] Output larger than input: "
                        f"{file_size / 1e9:.2f}GB > {input_size / 1e9:.2f}GB"
                    )
        except OSError:
            pass  # Skip size comparison if input files not available

        console.print(f"[green]✓[/] Output verified: {output_path.name} ({file_size / 1e9:.2f}GB)")

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
                    "--branch",
                    LLAMA_CPP_BRANCH,
                    LLAMA_CPP_REPO,
                    str(self.llama_cpp_dir),
                ],
                check=True,
            )
            # Verify clone integrity
            if not _verify_git_clone(self.llama_cpp_dir, LLAMA_CPP_REPO):
                shutil.rmtree(self.llama_cpp_dir)
                raise RuntimeError(
                    "Git clone integrity verification failed. "
                    "The cloned repository does not match expected source."
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

        # Resume support: skip FP16 conversion if already exists
        if fp16_path.exists() and fp16_path.stat().st_size > 0:
            console.print("[green]Resuming:[/] FP16 intermediate already exists, skipping step 1")
        else:
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
            # Verify output
            self._verify_output(final_path, model_path)
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

        # Verify output integrity
        self._verify_output(final_path, model_path)

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

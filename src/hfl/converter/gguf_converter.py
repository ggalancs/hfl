# SPDX-License-Identifier: Apache-2.0
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
from hfl.exceptions import ConversionError, ToolNotFoundError

console = Console()

_conversion_locks: dict[str, threading.Lock] = {}
_conversion_locks_guard = threading.Lock()


class UnsupportedModelError(Exception):
    """Raised when a model cannot be converted to GGUF format."""


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


# Quantizes argv[1] -> argv[2] at type argv[3] with llama-cpp-python's own
# llama_model_quantize: the [llama] extra ships the quantizer, compiled.
_LLAMA_CPP_QUANTIZE = """
import ctypes, sys
import llama_cpp
source, target, kind = sys.argv[1], sys.argv[2], sys.argv[3].upper()
ftype = getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_" + kind, None)
if ftype is None:
    sys.exit(f"llama-cpp-python cannot quantize to {kind}")
params = llama_cpp.llama_model_quantize_default_params()
params.ftype = ftype
code = llama_cpp.llama_model_quantize(source.encode(), target.encode(), ctypes.byref(params))
sys.exit(1 if code != 0 else 0)
"""

LLAMA_CPP_ARCHIVE = (
    "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/" + LLAMA_CPP_BRANCH
)


def _download_converter(target: Path) -> None:
    """llama.cpp's source from its archive, for a machine without git.

    The whole tree, not only ``convert_hf_to_gguf.py`` and ``gguf-py``: the
    script now imports its own ``conversion`` package, and a list of the
    files it needs would break on its next split. Regular files and
    directories only (no links, no devices), none outside ``target``."""
    import tarfile
    import tempfile

    import httpx

    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        archive = Path(staging) / "llama.cpp.tar.gz"
        with httpx.stream("GET", LLAMA_CPP_ARCHIVE, timeout=60.0, follow_redirects=True) as r:
            r.raise_for_status()
            with open(archive, "wb") as sink:
                for chunk in r.iter_bytes():
                    sink.write(chunk)
        unpacked = Path(staging) / "src"
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                inner = Path(*parts[1:]) if len(parts) > 1 else None
                if inner is None or not (member.isfile() or member.isdir()):
                    continue
                if inner.is_absolute() or ".." in inner.parts:
                    continue
                if member.isdir():
                    (unpacked / inner).mkdir(parents=True, exist_ok=True)
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                (unpacked / inner).parent.mkdir(parents=True, exist_ok=True)
                (unpacked / inner).write_bytes(source.read())
        shutil.move(str(unpacked), str(target))


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


def _gguf_variant_path(base: Path, label: str) -> Path:
    """Return ``base`` with ``.{label}.gguf`` appended.

    Deliberately avoids :meth:`pathlib.Path.with_suffix`: model names
    routinely contain version dots (e.g. ``Qwen--Qwen2.5-7B-Instruct``)
    which ``with_suffix`` mistakes for a file extension and truncates,
    yielding wrong — and, across models sharing a prefix, colliding —
    output filenames.
    """
    return base.with_name(f"{base.name}.{label}.gguf")


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
    # Normalise the quantization BEFORE building the lock key so the key and the
    # output path derive from the same canonical token. Otherwise two callers
    # passing 'q4_k_m' and 'Q4_K_M' acquire DIFFERENT locks but target the SAME
    # output file, defeating the serialization the lock exists to provide (both
    # can pass the exists() check and convert concurrently, corrupting output).
    quant = quantization.upper()
    cache_key = f"{model_path}:{quant}"

    # Check if already converted (fast path)
    if quant == "F16":
        expected_output = _gguf_variant_path(output_path, "f16")
    else:
        expected_output = _gguf_variant_path(output_path, quant)

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

    def _check_conversion_environment(self) -> None:
        """Probe the same Python that will run ``convert_hf_to_gguf.py``.

        ``convert_hf_to_gguf.py`` imports ``transformers`` at module
        load. If the host Python has an incompatible
        ``transformers`` / ``huggingface_hub`` combo (e.g. transformers
        5.x with huggingface_hub 0.x), the import crashes with a
        cryptic ``ImportError: cannot import name 'is_offline_mode'``.

        We probe with a short subprocess that imports the same surface,
        and turn any failure into an actionable error message *before*
        we waste time launching the real conversion.
        """
        probe = (
            "import sys\n"
            "try:\n"
            "    import numpy, torch  # noqa: F401\n"
            "    import huggingface_hub  # noqa: F401\n"
            "    import transformers  # noqa: F401\n"
            "    from transformers import AutoTokenizer  # noqa: F401\n"
            "except Exception as e:\n"
            "    sys.stderr.write(f'{type(e).__name__}: {e}')\n"
            "    sys.exit(2)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or "<no stderr>"
            # A ConversionError, which `hfl pull` shows as a message; the
            # advice to pin transformers 4.x predated HFL's move to 5.x.
            raise ConversionError(
                "safetensors",
                "GGUF",
                "llama.cpp's converter runs in this Python and cannot import what it "
                f"needs ({stderr}). Install HFL's Transformers extra here: "
                f"pip install 'hfl[transformers]'  (interpreter: {sys.executable})",
            )

    def ensure_tools(self) -> None:
        """Make sure ``convert_hf_to_gguf.py`` (and its ``gguf-py``) is here.

        Only the Python converter: nothing is compiled. It used to build
        all of llama.cpp with cmake on the first conversion, so every install
        without cmake and a C++ toolchain — a new user, Docker, a clean audit —
        failed at HFL's main feature, while an install that had built it once
        kept working. Quantizing finds a ``llama-quantize`` of its own
        (:meth:`_quantizer`).
        """
        if self.convert_script.exists():
            return
        console.print("[yellow]Fetching llama.cpp's converter (Python only)...[/]")
        self.llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("git") is not None:
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
            if not _verify_git_clone(self.llama_cpp_dir, LLAMA_CPP_REPO):
                shutil.rmtree(self.llama_cpp_dir)
                raise RuntimeError(
                    "Git clone integrity verification failed. "
                    "The cloned repository does not match expected source."
                )
        else:
            _download_converter(self.llama_cpp_dir)
        if not self.convert_script.exists():
            raise ToolNotFoundError(
                "convert_hf_to_gguf.py",
                f"llama.cpp's source was fetched into {self.llama_cpp_dir} without it.",
            )
        console.print("[green]Converter ready.[/]")

    def _quantizer(self) -> list[str]:
        """The command that quantizes ``<in> <out> <TYPE>``, from what is
        already on this machine, in order: a ``llama-quantize`` built here
        before, one on the PATH (Homebrew's llama.cpp, a distro package),
        llama-cpp-python's own quantizer (the [llama] extra). Only when none
        exists is llama.cpp built, which needs git, cmake and a C++ compiler.
        """
        if self.quantize_bin.exists():
            return [str(self.quantize_bin)]
        on_path = shutil.which("llama-quantize")
        if on_path is not None:
            return [on_path]
        # The one `hfl install llama-server` brings (or an executable bundles).
        from hfl.engine.llama_server_dist import bundled_binary, managed_binary

        installed = bundled_binary("llama-quantize") or managed_binary("llama-quantize")
        if installed is not None:
            return [installed]
        probe = subprocess.run(
            [sys.executable, "-c", "import llama_cpp; llama_cpp.llama_model_quantize"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return [sys.executable, "-c", _LLAMA_CPP_QUANTIZE]
        self._build_quantizer()
        return [str(self.quantize_bin)]

    def _build_quantizer(self) -> None:
        """Build llama.cpp's ``llama-quantize`` — the last resort."""
        hint = (
            "Quantizing needs llama-quantize: `hfl install llama-server` (brings it), "
            "`pip install 'hfl[llama]'`, or git + cmake + a C++ compiler to build it. "
            "Or pull a GGUF build of the model: `hfl search <name> --gguf`."
        )
        for tool in ("git", "cmake"):
            if shutil.which(tool) is None:
                raise ToolNotFoundError(tool, hint)
        if not (self.llama_cpp_dir / "CMakeLists.txt").exists():
            raise ToolNotFoundError(
                "llama.cpp's build files", "The converter was fetched without them. " + hint
            )
        console.print("[yellow]Building llama-quantize (llama.cpp)...[/]")
        build_dir = self.llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)
        cmake_cmd = ["cmake", ".."]
        if shutil.which("nvcc"):
            cmake_cmd.append("-DGGML_CUDA=ON")
        subprocess.run(cmake_cmd, cwd=build_dir, check=True)
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j", "--target", "llama-quantize"],
            cwd=build_dir,
            check=True,
        )

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
        fp16_path = _gguf_variant_path(output_path, "fp16")

        # Resume support: skip FP16 conversion if already exists
        if fp16_path.exists() and fp16_path.stat().st_size > 0:
            console.print("[green]Resuming:[/] FP16 intermediate already exists, skipping step 1")
        else:
            console.print("[cyan]Step 1/2:[/] Converting to GGUF FP16...")
            # Fail fast on incompatible host Python before invoking
            # convert_hf_to_gguf.py — otherwise the user sees a cryptic
            # ``ImportError`` from inside transformers/huggingface_hub.
            self._check_conversion_environment()

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
            final_path = _gguf_variant_path(output_path, "f16")
            fp16_path.rename(final_path)
            # Verify output
            self._verify_output(final_path, model_path)
            return final_path

        # Step 2: Quantize to the requested level
        quant = quantization.upper()
        final_path = _gguf_variant_path(output_path, quant)

        console.print(f"[cyan]Step 2/2:[/] Quantizing to {quant}...")

        subprocess.run([*self._quantizer(), str(fp16_path), str(final_path), quant], check=True)

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

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Check engine dependencies at startup.

Provides functions to check which inference backends are available
and log their status at startup.
"""

import logging

logger = logging.getLogger(__name__)


def check_engine_availability() -> dict[str, bool | str]:
    """Check which inference backends are available.

    Returns:
        Dict mapping backend names to availability status or error message.
    """
    results: dict[str, bool | str] = {}

    # llama-cpp-python
    try:
        import llama_cpp  # noqa: F401

        results["llama-cpp"] = True
    except ImportError as e:
        results["llama-cpp"] = f"Not installed: {e}"

    # transformers + torch
    try:
        import transformers  # noqa: F401

        results["transformers"] = True
    except ImportError as e:
        results["transformers"] = f"Not installed: {e}"

    # torch and CUDA
    try:
        import torch

        results["torch"] = True
        results["torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["cuda_device"] = torch.cuda.get_device_name(0)
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            results["torch_mps"] = True
    except ImportError as e:
        results["torch"] = f"Not installed: {e}"

    # vllm
    try:
        import vllm  # noqa: F401

        results["vllm"] = True
    except ImportError as e:
        results["vllm"] = f"Not installed: {e}"

    # TTS dependencies
    try:
        import soundfile  # noqa: F401

        results["soundfile"] = True
    except ImportError:
        results["soundfile"] = False

    try:
        import torchaudio  # noqa: F401

        results["torchaudio"] = True
    except ImportError:
        results["torchaudio"] = False

    return results


def log_available_backends() -> None:
    """Log available backends at startup."""
    availability = check_engine_availability()

    logger.info("Backend availability check:")

    # Core backends
    for backend in ["llama-cpp", "transformers", "vllm"]:
        status = availability.get(backend, "unknown")
        if status is True:
            logger.info(f"  ✓ {backend}: available")
        else:
            logger.info(f"  ✗ {backend}: {status}")

    # GPU support
    if availability.get("torch") is True:
        if availability.get("torch_cuda"):
            device = availability.get("cuda_device", "unknown")
            logger.info(f"  ✓ CUDA: available ({device})")
        elif availability.get("torch_mps"):
            logger.info("  ✓ MPS (Apple Silicon): available")
        else:
            logger.info("  ✗ GPU: not available (CPU only)")

    # TTS support
    tts_available = availability.get("transformers") is True
    if tts_available:
        logger.info("  ✓ TTS (Bark): available")
    else:
        logger.info("  ✗ TTS: requires transformers")


def get_recommended_backend(model_format: str) -> str | None:
    """Get recommended backend for a model format.

    Args:
        model_format: Model format ("gguf", "safetensors", etc.)

    Returns:
        Recommended backend name or None if no suitable backend.
    """
    availability = check_engine_availability()

    if model_format == "gguf":
        if availability.get("llama-cpp") is True:
            return "llama-cpp"
        return None

    if model_format in ("safetensors", "pytorch"):
        # Prefer vllm if CUDA available
        if availability.get("vllm") is True and availability.get("torch_cuda"):
            return "vllm"
        # Fall back to transformers
        if availability.get("transformers") is True:
            return "transformers"
        # Last resort: llama-cpp can load some models
        if availability.get("llama-cpp") is True:
            return "llama-cpp"

    return None

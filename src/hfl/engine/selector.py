# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Automatic inference backend selection.

Decision logic for LLM:
  1. If model is GGUF -> LlamaCppEngine
  2. If NVIDIA GPU + safetensors model -> TransformersEngine (4bit)
  3. If vLLM installed + GPU -> vLLM for production
  4. Fallback -> Convert to GGUF + LlamaCppEngine

Decision logic for TTS:
  1. If Bark model -> BarkEngine (transformers)
  2. If Coqui model -> CoquiEngine
  3. Auto-detect based on config.json
"""

from pathlib import Path

from hfl.converter.formats import ModelFormat, ModelType, detect_format, detect_model_type
from hfl.engine.base import AudioEngine, InferenceEngine


class MissingDependencyError(Exception):
    """Error when an optional dependency is missing."""

    pass


def _get_llama_cpp_engine():
    """Lazy import of LlamaCppEngine."""
    try:
        from hfl.engine.llama_cpp import LlamaCppEngine

        return LlamaCppEngine()
    except ImportError as e:
        raise MissingDependencyError(
            "The llama-cpp backend requires the 'llama-cpp-python' library.\n\n"
            "Install it with:\n"
            "  pip install llama-cpp-python\n\n"
            "For GPU support (CUDA):\n"
            '  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python\n\n'
            "For macOS with Metal:\n"
            '  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python'
        ) from e


def select_engine(
    model_path: Path,
    backend: str = "auto",
    **kwargs,
) -> InferenceEngine:
    """
    Selects and instantiates the appropriate inference engine.

    Args:
        model_path: Path to the model
        backend: "auto", "llama-cpp", "transformers", "vllm"
        **kwargs: Additional parameters for the engine
    """
    fmt = detect_format(model_path)

    if backend != "auto":
        return _create_engine(backend)

    # Auto-selection
    if fmt == ModelFormat.GGUF:
        return _get_llama_cpp_engine()

    # For safetensors, check GPU
    if _has_cuda():
        try:
            return _get_transformers_engine()
        except MissingDependencyError:
            pass  # Fallback to llama.cpp

    # Fallback: llama.cpp (will need prior conversion)
    return _get_llama_cpp_engine()


def _get_transformers_engine():
    """Lazy import of TransformersEngine."""
    try:
        from hfl.engine.transformers_engine import TransformersEngine

        return TransformersEngine()
    except ImportError as e:
        raise MissingDependencyError(
            "The transformers backend requires additional dependencies.\n\n"
            "Install them with:\n"
            "  pip install hfl[transformers]\n\n"
            "Or directly:\n"
            "  pip install transformers torch accelerate"
        ) from e


def _get_vllm_engine():
    """Lazy import of VLLMEngine."""
    try:
        from hfl.engine.vllm_engine import VLLMEngine

        return VLLMEngine()
    except ImportError as e:
        raise MissingDependencyError(
            "The vLLM backend requires additional dependencies.\n\n"
            "Install them with:\n"
            "  pip install hfl[vllm]\n\n"
            "Or directly:\n"
            "  pip install vllm\n\n"
            "Note: vLLM requires NVIDIA GPU with CUDA."
        ) from e


def _create_engine(name: str) -> InferenceEngine:
    if name == "llama-cpp":
        return _get_llama_cpp_engine()
    elif name == "transformers":
        return _get_transformers_engine()
    elif name == "vllm":
        return _get_vllm_engine()
    raise ValueError(f"Unknown backend: {name}")


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# =============================================================================
# TTS Engine Selection
# =============================================================================


def _get_bark_engine():
    """Lazy import of BarkEngine."""
    try:
        from hfl.engine.bark_engine import BarkEngine

        return BarkEngine()
    except ImportError as e:
        raise MissingDependencyError(
            "The Bark TTS engine requires additional dependencies.\n\n"
            "Install them with:\n"
            "  pip install hfl[tts]\n\n"
            "Or directly:\n"
            "  pip install transformers torch torchaudio soundfile"
        ) from e


def _get_coqui_engine():
    """Lazy import of CoquiEngine."""
    try:
        from hfl.engine.coqui_engine import CoquiEngine

        return CoquiEngine()
    except ImportError as e:
        raise MissingDependencyError(
            "The Coqui TTS engine requires additional dependencies.\n\n"
            "Install them with:\n"
            "  pip install hfl[coqui]\n\n"
            "Or directly:\n"
            "  pip install coqui-tts"
        ) from e


def _is_bark_model(model_path: Path) -> bool:
    """Check if the model is a Bark model."""
    import json

    # Check model name
    model_name = model_path.name.lower()
    if "bark" in model_name:
        return True

    # Check config.json
    if model_path.is_dir():
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                architectures = config.get("architectures", [])
                return any("Bark" in arch for arch in architectures)
            except (json.JSONDecodeError, OSError):
                pass

    return False


def _is_coqui_model(model_path: Path) -> bool:
    """Check if the model is a Coqui TTS model."""
    # Coqui models typically have a specific structure
    # or are specified by model name pattern
    model_name = str(model_path).lower()

    # Check for Coqui model naming patterns
    coqui_patterns = [
        "tts_models/",
        "xtts",
        "vits",
        "tacotron",
        "glow-tts",
        "speedy-speech",
    ]

    return any(pattern in model_name for pattern in coqui_patterns)


def select_tts_engine(
    model_path: Path,
    backend: str = "auto",
    **kwargs,
) -> AudioEngine:
    """
    Selects and instantiates the appropriate TTS engine.

    Args:
        model_path: Path to the model
        backend: "auto", "bark", or "coqui"
        **kwargs: Additional parameters for the engine

    Returns:
        AudioEngine instance

    Raises:
        MissingDependencyError: If required dependencies are not installed
        ValueError: If no suitable backend is found
    """
    # Explicit backend selection
    if backend == "bark":
        return _get_bark_engine()
    elif backend == "coqui":
        return _get_coqui_engine()

    # Auto-detection
    if backend == "auto":
        # Check model type
        model_type = detect_model_type(model_path)

        if model_type != ModelType.TTS:
            raise ValueError(
                f"Model at {model_path} does not appear to be a TTS model. "
                f"Detected type: {model_type.value}"
            )

        # Try to identify the specific TTS framework
        if _is_bark_model(model_path):
            return _get_bark_engine()

        if _is_coqui_model(model_path):
            return _get_coqui_engine()

        # Default to Bark for transformers-based TTS
        # (SpeechT5, MMS, etc. are also supported via transformers pipeline)
        try:
            return _get_bark_engine()
        except MissingDependencyError:
            pass

        # Try Coqui as fallback
        try:
            return _get_coqui_engine()
        except MissingDependencyError:
            pass

    raise ValueError(
        f"Could not find a suitable TTS backend for {model_path}.\n"
        "Install one of:\n"
        "  pip install hfl[tts]     # For Bark/transformers\n"
        "  pip install hfl[coqui]   # For Coqui TTS"
    )

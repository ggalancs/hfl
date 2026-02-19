# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Automatic inference backend selection.

Decision logic:
  1. If model is GGUF -> LlamaCppEngine
  2. If NVIDIA GPU + safetensors model -> TransformersEngine (4bit)
  3. If vLLM installed + GPU -> vLLM for production
  4. Fallback -> Convert to GGUF + LlamaCppEngine
"""

from pathlib import Path

from hfl.converter.formats import ModelFormat, detect_format
from hfl.engine.base import InferenceEngine


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

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Selección automática del backend de inferencia.

Lógica de decisión:
  1. Si el modelo es GGUF → LlamaCppEngine
  2. Si hay GPU NVIDIA + modelo safetensors → TransformersEngine (4bit)
  3. Si hay vLLM instalado + GPU → vLLM para producción
  4. Fallback → Convertir a GGUF + LlamaCppEngine
"""

from pathlib import Path

from hfl.engine.base import InferenceEngine
from hfl.converter.formats import ModelFormat, detect_format


class MissingDependencyError(Exception):
    """Error cuando falta una dependencia opcional."""
    pass


def _get_llama_cpp_engine():
    """Import lazy de LlamaCppEngine."""
    try:
        from hfl.engine.llama_cpp import LlamaCppEngine
        return LlamaCppEngine()
    except ImportError as e:
        raise MissingDependencyError(
            "El backend llama-cpp requiere la librería 'llama-cpp-python'.\n\n"
            "Instálala con:\n"
            "  pip install llama-cpp-python\n\n"
            "Para soporte GPU (CUDA):\n"
            "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python\n\n"
            "Para macOS con Metal:\n"
            "  CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python"
        ) from e


def select_engine(
    model_path: Path,
    backend: str = "auto",
    **kwargs,
) -> InferenceEngine:
    """
    Selecciona e instancia el motor de inferencia adecuado.

    Args:
        model_path: Ruta al modelo
        backend: "auto", "llama-cpp", "transformers", "vllm"
        **kwargs: Parámetros adicionales para el engine
    """
    fmt = detect_format(model_path)

    if backend != "auto":
        return _create_engine(backend)

    # Auto-selección
    if fmt == ModelFormat.GGUF:
        return _get_llama_cpp_engine()

    # Para safetensors, comprobar GPU
    if _has_cuda():
        try:
            return _get_transformers_engine()
        except MissingDependencyError:
            pass  # Fallback a llama.cpp

    # Fallback: llama.cpp (necesitará conversión previa)
    return _get_llama_cpp_engine()


def _get_transformers_engine():
    """Import lazy de TransformersEngine."""
    try:
        from hfl.engine.transformers_engine import TransformersEngine
        return TransformersEngine()
    except ImportError as e:
        raise MissingDependencyError(
            "El backend transformers requiere dependencias adicionales.\n\n"
            "Instálalas con:\n"
            "  pip install hfl[transformers]\n\n"
            "O directamente:\n"
            "  pip install transformers torch accelerate"
        ) from e


def _get_vllm_engine():
    """Import lazy de VLLMEngine."""
    try:
        from hfl.engine.vllm_engine import VLLMEngine
        return VLLMEngine()
    except ImportError as e:
        raise MissingDependencyError(
            "El backend vLLM requiere dependencias adicionales.\n\n"
            "Instálalas con:\n"
            "  pip install hfl[vllm]\n\n"
            "O directamente:\n"
            "  pip install vllm\n\n"
            "Nota: vLLM requiere GPU NVIDIA con CUDA."
        ) from e


def _create_engine(name: str) -> InferenceEngine:
    if name == "llama-cpp":
        return _get_llama_cpp_engine()
    elif name == "transformers":
        return _get_transformers_engine()
    elif name == "vllm":
        return _get_vllm_engine()
    raise ValueError(f"Backend desconocido: {name}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

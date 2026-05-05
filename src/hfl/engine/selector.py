# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Automatic inference backend selection.

Decision logic for LLM:
  1. If model is GGUF -> LlamaCppEngine (Metal on macOS)
  2. On Darwin-arm64 with mlx-lm installed + safetensors/pytorch -> MLXEngine
  3. If NVIDIA GPU + safetensors model -> TransformersEngine (4bit)
  4. If vLLM installed + GPU -> vLLM for production
  5. Fallback -> Convert to GGUF + LlamaCppEngine

The MLX path hits raw Metal directly and outperforms llama-cpp's
Metal path on M-series silicon for Llama-family architectures. It is
opt-out via ``HFL_DISABLE_MLX=1`` for users who want the llama-cpp
behaviour on Apple Silicon regardless.

Decision logic for TTS:
  1. If Bark model -> BarkEngine (transformers)
  2. If Coqui model -> CoquiEngine
  3. Auto-detect based on config.json
"""

import os
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


def _get_mlx_engine():
    """Lazy import + availability gate for MLXEngine.

    Raises MissingDependencyError either when the host isn't
    Darwin-arm64 (MLX is Apple Silicon only) or when ``mlx_lm`` is
    not installed. Callers in auto-mode should catch this and fall
    through to the next candidate; callers requesting MLX explicitly
    get the error surfaced.
    """
    from hfl.engine import mlx_engine

    if not mlx_engine.is_available():
        raise MissingDependencyError(
            "The MLX backend requires Apple Silicon (Darwin-arm64) with "
            "the 'mlx-lm' library installed.\n\n"
            "Install it with:\n"
            "  pip install 'hfl[mlx]'\n\n"
            "Or directly:\n"
            "  pip install mlx-lm"
        )
    return mlx_engine.MLXEngine()


def _mlx_preferred() -> bool:
    """True when MLX should be the default backend for safetensors.

    Off-switch: ``HFL_DISABLE_MLX=1`` forces the legacy path on
    Apple Silicon (useful for benchmarking parity with llama-cpp).
    """
    if os.environ.get("HFL_DISABLE_MLX", "").strip() in ("1", "true", "True", "yes"):
        return False
    from hfl.engine import mlx_engine

    return mlx_engine.is_available()


def _resolve_forced_backend() -> str | None:
    """Read the server-level backend override from the environment.

    Resolution: ``HFL_LLM_LIBRARY`` first, then ``OLLAMA_LLM_LIBRARY``
    (the Ollama-equivalent name). Returns ``None`` when neither is
    set, so the per-call ``backend=`` argument keeps working.

    Accepted values: ``"llama-cpp"``, ``"transformers"``, ``"vllm"``,
    ``"mlx"``. An unrecognised value logs a warning and returns
    ``None`` so the auto path is used — operator typos must not crash
    the server.
    """
    import logging as _logging

    raw = os.environ.get("HFL_LLM_LIBRARY") or os.environ.get("OLLAMA_LLM_LIBRARY")
    if not raw:
        return None
    name = raw.strip().lower()
    if name in {"llama-cpp", "transformers", "vllm", "mlx"}:
        return name
    _logging.getLogger(__name__).warning(
        "HFL_LLM_LIBRARY=%r is not a recognised backend, ignoring", raw
    )
    return None


def select_engine(
    model_path: Path,
    backend: str = "auto",
    **kwargs,
) -> InferenceEngine:
    """
    Selects and instantiates the appropriate inference engine.

    Args:
        model_path: Path to the model
        backend: "auto", "llama-cpp", "transformers", "vllm", "mlx".
            Overridden by ``HFL_LLM_LIBRARY`` /
            ``OLLAMA_LLM_LIBRARY`` when the caller passed ``"auto"``.
            An explicit non-auto request from the caller (e.g.
            Modelfile-driven or ``--backend`` flag) always wins so
            per-model decisions are not silently overwritten by a
            server default.
        **kwargs: Additional parameters for the engine
    """
    fmt = detect_format(model_path)

    if backend == "auto":
        forced = _resolve_forced_backend()
        if forced is not None:
            return _create_engine(forced)
    elif backend != "auto":
        return _create_engine(backend)

    # Auto-selection
    if fmt == ModelFormat.GGUF:
        # GGUF stays on llama-cpp (Metal on macOS). MLX does not
        # ingest GGUF, so no opportunity to route it there.
        return _get_llama_cpp_engine()

    # Safetensors / pytorch weights. On Apple Silicon with mlx-lm
    # installed, MLX is the fastest path for Llama-family models and
    # avoids the detour through GGUF conversion.
    if _mlx_preferred():
        try:
            return _get_mlx_engine()
        except MissingDependencyError:
            pass  # Fall through to the legacy decision tree.

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
    elif name == "mlx":
        return _get_mlx_engine()
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

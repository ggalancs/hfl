# SPDX-License-Identifier: Apache-2.0
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

import importlib.util
import logging
import os
import platform
import re
from pathlib import Path
from typing import cast

from hfl.converter.formats import ModelFormat, ModelType, detect_format, detect_model_type
from hfl.engine.base import AudioEngine, InferenceEngine
from hfl.exceptions import EngineError

logger = logging.getLogger(__name__)


class MissingDependencyError(EngineError):
    """Error when an optional dependency is missing.

    An :class:`HFLError`, so the API answers it with its message (501: this
    install does not implement what the request needs) rather than an opaque
    500."""

    status_code = 501


def _get_llama_cpp_engine() -> InferenceEngine:
    """Lazy import of LlamaCppEngine."""
    try:
        from hfl.engine.llama_cpp import LlamaCppEngine

        return cast(InferenceEngine, LlamaCppEngine())
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


def _llama_server_if_no_llama_cpp(model_path: Path) -> InferenceEngine | None:
    """llama-server for a GGUF when llama-cpp-python is not installed.

    An install without the ``[llama]`` extra (Homebrew's, for one, which
    depends on llama.cpp instead of compiling the Python binding) still
    serves GGUF models when ``llama-server`` is on the PATH.
    """
    import importlib.util
    import sys

    if "llama_cpp" in sys.modules:  # already imported (or stubbed): present
        return None
    try:
        if importlib.util.find_spec("llama_cpp") is not None:
            return None
    except (ImportError, ValueError):
        pass
    from hfl.engine.llama_server import LlamaServerEngine, binary

    if binary() is None:
        return None
    logging.getLogger(__name__).info(
        "llama-cpp-python is not installed; serving %s with llama-server", model_path.name
    )
    return LlamaServerEngine()


def _get_mlx_engine() -> InferenceEngine:
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
    return cast(InferenceEngine, mlx_engine.MLXEngine())


def _mlx_preferred() -> bool:
    """True when MLX should be the default backend for safetensors.

    Off-switch: ``HFL_DISABLE_MLX=1`` forces the legacy path on
    Apple Silicon (useful for benchmarking parity with llama-cpp).
    """
    if os.environ.get("HFL_DISABLE_MLX", "").strip() in ("1", "true", "True", "yes"):
        return False
    from hfl.engine import mlx_engine

    return mlx_engine.is_available()


# Models already advised about, so a chatty server does not repeat itself
# on every load. Keyed by path: a different quantisation of the same model
# is a different decision and deserves its own line.
_MLX_ADVISED: set[str] = set()


def _advise_mlx_alternative(model_path: str | Path) -> None:
    """Point out that an MLX build of this model would likely be faster.

    Only on Apple Silicon, only when ``mlx-lm`` is already installed, and
    only for a GGUF — the three conditions under which the operator can
    act on the advice today.

    Deliberately **offline**. Confirming that ``mlx-community/<model>``
    exists would mean a Hub round-trip on every model load, and loading a
    local model must not require the network. So the wording promises a
    build *may* exist and names the command that checks; it never claims
    one does. An advisory that overstates what it knows is how a user
    learns to ignore advisories.
    """
    if os.environ.get("HFL_NO_MLX_HINT", "").strip() in ("1", "true", "True", "yes"):
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return
    if not _mlx_preferred():
        return

    key = str(model_path)
    if key in _MLX_ADVISED:
        return
    _MLX_ADVISED.add(key)

    stem = Path(model_path).stem
    # Community GGUF names carry the quantisation; the MLX fork will not.
    base = re.split(r"[.-](?:[IiQq]\d|f16|bf16|f32)", stem)[0]

    logger.info(
        "Serving %s through llama.cpp (Metal). On Apple Silicon an MLX build of "
        "the same model is typically faster, most of all on prompt processing. "
        "Look for one with: hfl search mlx-community/%s    (silence this with "
        "HFL_NO_MLX_HINT=1)",
        stem,
        base,
    )


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
    if name in {"llama-cpp", "llama-server", "transformers", "vllm", "mlx"}:
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
        backend: "auto", "llama-cpp", "llama-server", "transformers", "vllm", "mlx".
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
        if forced == "llama-server" and fmt != ModelFormat.GGUF:
            # Chosen per model: llama-server takes the GGUF models — vision
            # ones too, with their projector (``--mmproj``) — and anything
            # that is not GGUF keeps its own backend.
            forced = None
        if forced is not None:
            return _create_engine(forced)
    elif backend != "auto":
        return _create_engine(backend)

    # Auto-selection
    if fmt == ModelFormat.GGUF:
        # GGUF stays on llama-cpp (Metal on macOS). MLX does not ingest
        # GGUF, so this file cannot be routed there — but on Apple
        # Silicon an MLX *build* of the same model usually can, and is
        # markedly faster. Say so once instead of silently serving the
        # slower path.
        _advise_mlx_alternative(model_path)
        fallback = _llama_server_if_no_llama_cpp(model_path)
        if fallback is not None:
            return fallback
        return _get_llama_cpp_engine()

    # Safetensors / pytorch weights. On Apple Silicon with mlx-lm
    # installed, MLX is the fastest path for Llama-family models and
    # avoids the detour through GGUF conversion.
    if _mlx_preferred():
        try:
            return _get_mlx_engine()
        except MissingDependencyError:
            pass  # Fall through to the legacy decision tree.

    if fmt in (ModelFormat.SAFETENSORS, ModelFormat.PYTORCH):
        # llama.cpp reads GGUF files only: sending these weights there failed
        # at load with "Model path is not a file". Transformers serves them
        # on CUDA, MPS or CPU; without it, say what would serve the model.
        try:
            return _get_transformers_engine()
        except MissingDependencyError:
            raise MissingDependencyError(
                f"{model_path.name} is in {fmt.value} format, which this install "
                "cannot serve: llama.cpp reads GGUF only.\n\n"
                "Either install the Transformers backend:\n"
                "  pip install 'hfl[transformers]'\n"
                "or pull a GGUF build of the model:\n"
                "  hfl pull <repo> --format gguf"
            ) from None

    # Unknown layout: llama.cpp, whose loader says what it cannot read.
    return _get_llama_cpp_engine()


def _get_transformers_engine() -> InferenceEngine:
    """Lazy import of TransformersEngine."""
    try:
        # The engine module imports its libraries lazily, at load: without
        # this check an install lacking them got an engine that failed later,
        # opaquely, instead of this message now.
        for package in ("transformers", "torch"):
            if importlib.util.find_spec(package) is None:
                raise ImportError(package)
        from hfl.engine.transformers_engine import TransformersEngine

        return cast(InferenceEngine, TransformersEngine())
    except ImportError as e:
        raise MissingDependencyError(
            "The transformers backend requires additional dependencies.\n\n"
            "Install them with:\n"
            "  pip install hfl[transformers]\n\n"
            "Or directly:\n"
            "  pip install transformers torch accelerate"
        ) from e


def _get_vllm_engine() -> InferenceEngine:
    """Lazy import of VLLMEngine."""
    try:
        from hfl.engine.vllm_engine import VLLMEngine

        return cast(InferenceEngine, VLLMEngine())
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
    if name == "llama-server":
        from hfl.engine.llama_server import LlamaServerEngine

        return LlamaServerEngine()
    if name == "transformers":
        return _get_transformers_engine()
    if name == "vllm":
        return _get_vllm_engine()
    if name == "mlx":
        return _get_mlx_engine()
    raise ValueError(f"Unknown backend: {name}")


def _has_cuda() -> bool:
    try:
        import torch

        available: bool = torch.cuda.is_available()
        return available
    except ImportError:
        return False


# =============================================================================
# TTS Engine Selection
# =============================================================================


def _get_bark_engine() -> AudioEngine:
    """Lazy import of BarkEngine."""
    try:
        from hfl.engine.bark_engine import BarkEngine

        return cast(AudioEngine, BarkEngine())
    except ImportError as e:
        raise MissingDependencyError(
            "The Bark TTS engine requires additional dependencies.\n\n"
            "Install them with:\n"
            "  pip install hfl[tts]\n\n"
            "Or directly:\n"
            "  pip install transformers torch torchaudio soundfile"
        ) from e


def _get_coqui_engine() -> AudioEngine:
    """Lazy import of CoquiEngine."""
    try:
        from hfl.engine.coqui_engine import CoquiEngine

        return cast(AudioEngine, CoquiEngine())
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
    if backend == "coqui":
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

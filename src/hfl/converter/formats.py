# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Detection and validation of model formats."""

from enum import Enum
from pathlib import Path


class ModelFormat(Enum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    UNKNOWN = "unknown"


def detect_format(model_path: Path) -> ModelFormat:
    """Detects the format of a model given its directory or file."""
    if model_path.is_file():
        if model_path.suffix == ".gguf":
            return ModelFormat.GGUF
        elif model_path.suffix == ".safetensors":
            return ModelFormat.SAFETENSORS
        elif model_path.suffix in (".pt", ".pth", ".bin"):
            return ModelFormat.PYTORCH

    if model_path.is_dir():
        files = list(model_path.rglob("*"))
        extensions = {f.suffix for f in files}

        if ".gguf" in extensions:
            return ModelFormat.GGUF
        elif ".safetensors" in extensions:
            return ModelFormat.SAFETENSORS
        elif ".bin" in extensions or ".pt" in extensions:
            return ModelFormat.PYTORCH

    return ModelFormat.UNKNOWN


def find_model_file(model_path: Path, fmt: ModelFormat) -> Path | None:
    """Finds the main model file."""
    if model_path.is_file():
        return model_path

    if fmt == ModelFormat.GGUF:
        gguf_files = list(model_path.rglob("*.gguf"))
        return gguf_files[0] if gguf_files else None

    return model_path  # For safetensors/pytorch, return the directory

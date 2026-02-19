# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Smart resolution of model names.

Supports multiple input formats:
  - "meta-llama/Llama-3.3-70B-Instruct"       -> exact repo
  - "llama3.3:70b"                              -> search by alias
  - "TheBloke/Llama-3.3-70B-Instruct-GGUF"     -> pre-quantized GGUF
"""

from dataclasses import dataclass

from huggingface_hub import HfApi


@dataclass
class ResolvedModel:
    repo_id: str  # org/model in HF
    revision: str = "main"  # branch/tag/commit
    filename: str | None = None  # Specific file (for GGUF)
    format: str = "auto"  # auto, gguf, safetensors, pytorch
    quantization: str | None = None  # Q4_K_M, Q5_K_M, etc.


def resolve(model_spec: str, quantization: str | None = None) -> ResolvedModel:
    """
    Resolve a model specification to a concrete ResolvedModel.

    Supported formats:
    - "org/model"                    -> direct repo
    - "org/model:Q4_K_M"             -> repo with quantization (Ollama style)
    - "model-name"                   -> search by name

    Resolution strategy:
    1. Extract quantization if using repo:quant format
    2. If contains '/', treat as direct repo_id
    3. If the repo has GGUF files, use them directly
    4. Otherwise, use safetensors and mark for conversion
    """
    api = HfApi()

    # Extract quantization from "repo:Q4_K_M" format (Ollama style)
    if ":" in model_spec:
        parts = model_spec.rsplit(":", 1)
        # Verify that the part after : looks like a quantization
        if _is_quantization(parts[1]):
            model_spec = parts[0]
            # Explicit quantization in the spec takes priority
            quantization = parts[1]

    # Case 1: direct repo_id (org/model)
    if "/" in model_spec:
        repo_id = model_spec
    else:
        # Case 2: search by name
        results = api.list_models(
            search=model_spec,
            sort="downloads",
            direction=-1,
            limit=5,
        )
        results = list(results)
        if not results:
            raise ValueError(f"Model not found: {model_spec}")
        repo_id = results[0].id

    # Detect available format
    siblings = api.model_info(repo_id).siblings or []
    filenames = [s.rfilename for s in siblings]

    gguf_files = [f for f in filenames if f.endswith(".gguf")]
    safetensor_files = [f for f in filenames if f.endswith(".safetensors")]

    if gguf_files:
        # Select the GGUF that best matches the requested quantization
        target_file = _select_gguf(gguf_files, quantization)
        return ResolvedModel(
            repo_id=repo_id,
            filename=target_file,
            format="gguf",
            quantization=_detect_quant(target_file),
        )
    elif safetensor_files:
        return ResolvedModel(
            repo_id=repo_id,
            format="safetensors",
            quantization=quantization,
        )
    else:
        return ResolvedModel(
            repo_id=repo_id,
            format="pytorch",
            quantization=quantization,
        )


def _select_gguf(files: list[str], quant: str | None) -> str:
    """Select the most appropriate GGUF file."""
    if quant:
        quant_upper = quant.upper()
        for f in files:
            if quant_upper in f.upper():
                return f

    # Default priority: Q4_K_M > Q5_K_M > Q4_K_S > first file
    priority = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S", "Q6_K", "Q8_0"]
    for q in priority:
        for f in files:
            if q in f.upper():
                return f

    return files[0]


def _detect_quant(filename: str) -> str | None:
    """Detect the quantization level from the filename."""
    quant_levels = _get_quant_levels()
    upper = filename.upper()
    for q in quant_levels:
        if q in upper:
            return q
    return None


def _get_quant_levels() -> list[str]:
    """List of known quantization levels."""
    return [
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_0",
        "Q4_1",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_0",
        "Q5_1",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
        "F16",
        "F32",
        "IQ1_S",
        "IQ1_M",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ2_M",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ3_M",
        "IQ4_NL",
        "IQ4_XS",
    ]


def _is_quantization(s: str) -> bool:
    """Check if a string looks like a quantization level."""
    if not s:
        return False
    upper = s.upper()
    # Check for exact or partial match with known levels
    for q in _get_quant_levels():
        if upper == q or upper == q.replace("_", ""):
            return True
    # Generic pattern: Q followed by number, or F16/F32, or IQ
    if upper.startswith(("Q", "F", "IQ")) and any(c.isdigit() for c in upper):
        return True
    return False

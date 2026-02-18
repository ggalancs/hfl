# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Resolución inteligente de nombres de modelo.

Soporta múltiples formatos de entrada:
  - "meta-llama/Llama-3.3-70B-Instruct"       → repo exacto
  - "llama3.3:70b"                              → búsqueda por alias
  - "TheBloke/Llama-3.3-70B-Instruct-GGUF"     → GGUF pre-cuantizado
"""

from dataclasses import dataclass
from huggingface_hub import HfApi


@dataclass
class ResolvedModel:
    repo_id: str                    # org/model en HF
    revision: str = "main"          # branch/tag/commit
    filename: str | None = None     # Archivo específico (para GGUF)
    format: str = "auto"            # auto, gguf, safetensors, pytorch
    quantization: str | None = None # Q4_K_M, Q5_K_M, etc.


def resolve(model_spec: str, quantization: str | None = None) -> ResolvedModel:
    """
    Resuelve una especificación de modelo a un ResolvedModel concreto.

    Estrategia de resolución:
    1. Si contiene '/', tratar como repo_id directo
    2. Si el repo tiene archivos GGUF, usarlos directamente
    3. Si no, usar safetensors y marcar para conversión
    """
    api = HfApi()

    # Caso 1: repo_id directo (org/model)
    if "/" in model_spec:
        repo_id = model_spec
    else:
        # Caso 2: búsqueda por nombre
        results = api.list_models(
            search=model_spec,
            sort="downloads",
            direction=-1,
            limit=5,
        )
        results = list(results)
        if not results:
            raise ValueError(f"No se encontró modelo: {model_spec}")
        repo_id = results[0].id

    # Detectar formato disponible
    siblings = api.model_info(repo_id).siblings or []
    filenames = [s.rfilename for s in siblings]

    gguf_files = [f for f in filenames if f.endswith(".gguf")]
    safetensor_files = [f for f in filenames if f.endswith(".safetensors")]

    if gguf_files:
        # Seleccionar el GGUF que mejor coincida con la cuantización pedida
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
    """Selecciona el archivo GGUF más apropiado."""
    if quant:
        quant_upper = quant.upper()
        for f in files:
            if quant_upper in f.upper():
                return f

    # Prioridad por defecto: Q4_K_M > Q5_K_M > Q4_K_S > primer archivo
    priority = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S", "Q6_K", "Q8_0"]
    for q in priority:
        for f in files:
            if q in f.upper():
                return f

    return files[0]


def _detect_quant(filename: str) -> str | None:
    """Detecta el nivel de cuantización del nombre del archivo."""
    quant_levels = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0", "F16", "F32",
    ]
    upper = filename.upper()
    for q in quant_levels:
        if q in upper:
            return q
    return None

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

    Formatos soportados:
    - "org/model"                    → repo directo
    - "org/model:Q4_K_M"             → repo con cuantización (estilo Ollama)
    - "nombre-modelo"                → búsqueda por nombre

    Estrategia de resolución:
    1. Extraer cuantización si usa formato repo:quant
    2. Si contiene '/', tratar como repo_id directo
    3. Si el repo tiene archivos GGUF, usarlos directamente
    4. Si no, usar safetensors y marcar para conversión
    """
    api = HfApi()

    # Extraer cuantización del formato "repo:Q4_K_M" (estilo Ollama)
    if ":" in model_spec:
        parts = model_spec.rsplit(":", 1)
        # Verificar que la parte después de : parece una cuantización
        if _is_quantization(parts[1]):
            model_spec = parts[0]
            # La cuantización explícita en el spec tiene prioridad
            quantization = parts[1]

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
    quant_levels = _get_quant_levels()
    upper = filename.upper()
    for q in quant_levels:
        if q in upper:
            return q
    return None


def _get_quant_levels() -> list[str]:
    """Lista de niveles de cuantización conocidos."""
    return [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0", "F16", "F32",
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
        "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
        "IQ4_NL", "IQ4_XS",
    ]


def _is_quantization(s: str) -> bool:
    """Verifica si una cadena parece un nivel de cuantización."""
    if not s:
        return False
    upper = s.upper()
    # Verificar coincidencia exacta o parcial con niveles conocidos
    for q in _get_quant_levels():
        if upper == q or upper == q.replace("_", ""):
            return True
    # Patrón genérico: Q seguido de número, o F16/F32, o IQ
    if upper.startswith(("Q", "F", "IQ")) and any(c.isdigit() for c in upper):
        return True
    return False

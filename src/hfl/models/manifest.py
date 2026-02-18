# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Modelo de datos para metadata de modelos locales."""

from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ModelManifest:
    """Metadata completa de un modelo descargado."""

    # Identificación (campos requeridos primero)
    name: str                        # Nombre corto (ej: "llama3.3-70b-q4")
    repo_id: str                     # HuggingFace repo (ej: "meta-llama/Llama-3.3-70B")

    # Almacenamiento
    local_path: str                  # Ruta absoluta al modelo
    format: str                      # "gguf", "safetensors", "pytorch"

    # Campos opcionales (con default)
    alias: str | None = None         # Alias definido por usuario (ej: "coder")
    size_bytes: int = 0              # Tamaño en disco

    # Cuantización
    quantization: str | None = None  # Q4_K_M, Q5_K_M, etc.
    original_format: str | None = None  # Formato original si se convirtió

    # Modelo
    architecture: str | None = None  # llama, mistral, gemma, etc.
    parameters: str | None = None    # "7B", "70B", etc.
    context_length: int = 4096       # Longitud de contexto

    # Chat template
    chat_template: str | None = None # Jinja2 template para chat

    # Licencia (R1 - Auditoría Legal)
    license: str | None = None           # Identificador SPDX de la licencia
    license_name: str | None = None      # Nombre completo legible
    license_url: str | None = None       # URL a los términos completos
    license_restrictions: list[str] = field(default_factory=list)
    # ej: ["non-commercial", "no-derivative", "attribution-required"]
    gated: bool = False                  # Requirió aceptación de términos
    license_accepted_at: str | None = None  # Timestamp de aceptación

    # EU AI Act (R4 - Auditoría Legal)
    gpai_classification: str | None = None  # "gpai", "gpai-systemic", "exempt"
    training_flops: str | None = None       # Si está disponible en model card

    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    last_used: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def display_size(self) -> str:
        """Tamaño legible (ej: '4.2 GB')."""
        gb = self.size_bytes / (1024**3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        mb = self.size_bytes / (1024**2)
        return f"{mb:.0f} MB"

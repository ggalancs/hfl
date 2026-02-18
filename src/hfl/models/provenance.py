# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 hfl Contributors
"""
Registro de procedencia para modelos convertidos.

Documenta la cadena completa: origen -> conversión -> resultado.
Esto es importante para cumplimiento legal (R3 - Auditoría Legal)
y para preservar atribución en obras derivadas.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from hfl.config import config


@dataclass
class ConversionRecord:
    """Registro inmutable de una operación de conversión."""

    # Timestamp de la operación
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Origen
    source_repo: str = ""  # Repositorio HuggingFace original
    source_format: str = ""  # "safetensors", "pytorch", "gguf"
    source_revision: str = ""  # Commit hash del repo

    # Destino
    target_format: str = "gguf"
    target_path: str = ""  # Ruta del archivo resultante
    quantization: str = ""  # Q4_K_M, Q5_K_M, etc.

    # Herramienta usada
    tool_used: str = "llama.cpp/convert_hf_to_gguf.py"
    tool_version: str = ""
    hfl_version: str = "0.1.0"

    # Licencia
    original_license: str = ""
    license_accepted: bool = False
    license_accepted_at: str = ""

    # Tipo de conversión
    conversion_type: str = "format"  # "format", "quantize", "both"

    # Notas adicionales
    notes: str = ""


class ProvenanceLog:
    """
    Log persistente de todas las conversiones realizadas.

    Almacena un registro de cada operación de conversión para:
    1. Trazabilidad legal (demostrar procedencia)
    2. Atribución correcta a autores originales
    3. Auditoría de cumplimiento de licencias
    """

    def __init__(self, log_path: Path | None = None):
        """
        Inicializa el log de procedencia.

        Args:
            log_path: Ruta al archivo de log. Si no se especifica,
                      usa ~/.hfl/provenance.json
        """
        self.path = log_path or (config.home_dir / "provenance.json")
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Carga registros existentes del archivo."""
        if self.path.exists():
            try:
                self._records = json.loads(self.path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                self._records = []

    def _save(self) -> None:
        """Guarda registros al archivo."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._records, indent=2))

    def record(self, conversion: ConversionRecord) -> None:
        """
        Registra una nueva conversión.

        Args:
            conversion: Registro de la conversión realizada.
        """
        self._records.append(asdict(conversion))
        self._save()

    def get_history(self, repo_id: str) -> list[dict]:
        """
        Obtiene el historial de conversiones para un repositorio.

        Args:
            repo_id: ID del repositorio (ej: "meta-llama/Llama-3.1-8B")

        Returns:
            Lista de registros de conversión ordenados por fecha.
        """
        return sorted(
            [r for r in self._records if r.get("source_repo") == repo_id],
            key=lambda r: r.get("timestamp", ""),
        )

    def get_all(self) -> list[dict]:
        """
        Obtiene todos los registros de conversión.

        Returns:
            Lista completa de registros ordenados por fecha.
        """
        return sorted(self._records, key=lambda r: r.get("timestamp", ""))

    def find_by_target(self, target_path: str) -> dict | None:
        """
        Busca el registro de conversión para un archivo destino.

        Args:
            target_path: Ruta del archivo convertido.

        Returns:
            Registro de conversión o None si no existe.
        """
        for record in self._records:
            if record.get("target_path") == target_path:
                return record
        return None


# Instancia global del log
_provenance_log: ProvenanceLog | None = None


def get_provenance_log() -> ProvenanceLog:
    """Obtiene la instancia global del log de procedencia."""
    global _provenance_log
    if _provenance_log is None:
        _provenance_log = ProvenanceLog()
    return _provenance_log


def log_conversion(
    source_repo: str,
    source_format: str,
    target_path: str,
    quantization: str = "",
    original_license: str = "",
    license_accepted: bool = False,
    tool_version: str = "",
    notes: str = "",
) -> ConversionRecord:
    """
    Función de conveniencia para registrar una conversión.

    Args:
        source_repo: Repositorio HuggingFace original
        source_format: Formato original (safetensors, pytorch)
        target_path: Ruta del archivo GGUF resultante
        quantization: Nivel de cuantización aplicado
        original_license: Licencia del modelo original
        license_accepted: Si el usuario aceptó la licencia
        tool_version: Versión de la herramienta de conversión
        notes: Notas adicionales

    Returns:
        El registro de conversión creado.
    """
    record = ConversionRecord(
        source_repo=source_repo,
        source_format=source_format,
        target_path=target_path,
        quantization=quantization,
        original_license=original_license,
        license_accepted=license_accepted,
        license_accepted_at=datetime.now().isoformat() if license_accepted else "",
        tool_version=tool_version,
        conversion_type="both" if quantization else "format",
        notes=notes,
    )

    get_provenance_log().record(record)
    return record

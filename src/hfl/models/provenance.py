# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Provenance registry for converted models.

Documents the complete chain: origin -> conversion -> result.
This is important for legal compliance (R3 - Legal Audit)
and for preserving attribution in derivative works.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from hfl.config import config


@dataclass
class ConversionRecord:
    """Immutable record of a conversion operation."""

    # Operation timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Source
    source_repo: str = ""  # Original HuggingFace repository
    source_format: str = ""  # "safetensors", "pytorch", "gguf"
    source_revision: str = ""  # Repo commit hash

    # Destination
    target_format: str = "gguf"
    target_path: str = ""  # Path of the resulting file
    quantization: str = ""  # Q4_K_M, Q5_K_M, etc.

    # Tool used
    tool_used: str = "llama.cpp/convert_hf_to_gguf.py"
    tool_version: str = ""
    hfl_version: str = "0.1.0"

    # License
    original_license: str = ""
    license_accepted: bool = False
    license_accepted_at: str = ""

    # Conversion type
    conversion_type: str = "format"  # "format", "quantize", "both"

    # Additional notes
    notes: str = ""


class ProvenanceLog:
    """
    Persistent log of all conversions performed.

    Stores a record of each conversion operation for:
    1. Legal traceability (prove provenance)
    2. Correct attribution to original authors
    3. License compliance audit
    """

    def __init__(self, log_path: Path | None = None):
        """
        Initializes the provenance log.

        Args:
            log_path: Path to the log file. If not specified,
                      uses ~/.hfl/provenance.json
        """
        self.path = log_path or (config.home_dir / "provenance.json")
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Loads existing records from the file."""
        if self.path.exists():
            try:
                self._records = json.loads(self.path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                self._records = []

    def _save(self) -> None:
        """Saves records to the file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._records, indent=2))

    def record(self, conversion: ConversionRecord) -> None:
        """
        Records a new conversion.

        Args:
            conversion: Record of the performed conversion.
        """
        self._records.append(asdict(conversion))
        self._save()

    def get_history(self, repo_id: str) -> list[dict]:
        """
        Gets the conversion history for a repository.

        Args:
            repo_id: Repository ID (e.g., "meta-llama/Llama-3.1-8B")

        Returns:
            List of conversion records sorted by date.
        """
        return sorted(
            [r for r in self._records if r.get("source_repo") == repo_id],
            key=lambda r: r.get("timestamp", ""),
        )

    def get_all(self) -> list[dict]:
        """
        Gets all conversion records.

        Returns:
            Complete list of records sorted by date.
        """
        return sorted(self._records, key=lambda r: r.get("timestamp", ""))

    def find_by_target(self, target_path: str) -> dict | None:
        """
        Finds the conversion record for a target file.

        Args:
            target_path: Path of the converted file.

        Returns:
            Conversion record or None if it doesn't exist.
        """
        for record in self._records:
            if record.get("target_path") == target_path:
                return record
        return None


# Global log instance
_provenance_log: ProvenanceLog | None = None


def get_provenance_log() -> ProvenanceLog:
    """Gets the global provenance log instance."""
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
    Convenience function to record a conversion.

    Args:
        source_repo: Original HuggingFace repository
        source_format: Original format (safetensors, pytorch)
        target_path: Path of the resulting GGUF file
        quantization: Applied quantization level
        original_license: Original model license
        license_accepted: Whether the user accepted the license
        tool_version: Conversion tool version
        notes: Additional notes

    Returns:
        The created conversion record.
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

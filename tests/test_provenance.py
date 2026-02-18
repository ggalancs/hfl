# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests para el módulo models/provenance."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch


class TestConversionRecord:
    """Tests para ConversionRecord dataclass."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        from hfl.models.provenance import ConversionRecord

        record = ConversionRecord()

        assert record.source_repo == ""
        assert record.source_format == ""
        assert record.target_format == "gguf"
        assert record.tool_used == "llama.cpp/convert_hf_to_gguf.py"
        assert record.hfl_version == "0.1.0"
        assert record.conversion_type == "format"
        assert record.timestamp  # Debe tener un timestamp

    def test_custom_values(self):
        """Verifica valores personalizados."""
        from hfl.models.provenance import ConversionRecord

        record = ConversionRecord(
            source_repo="meta-llama/Llama-3.1-8B",
            source_format="safetensors",
            target_path="/path/to/model.gguf",
            quantization="Q4_K_M",
            original_license="llama3.1",
            license_accepted=True,
        )

        assert record.source_repo == "meta-llama/Llama-3.1-8B"
        assert record.source_format == "safetensors"
        assert record.target_path == "/path/to/model.gguf"
        assert record.quantization == "Q4_K_M"
        assert record.original_license == "llama3.1"
        assert record.license_accepted is True


class TestProvenanceLog:
    """Tests para ProvenanceLog."""

    def test_initialization_creates_empty_log(self, temp_dir):
        """Inicializa con log vacío si no existe archivo."""
        from hfl.models.provenance import ProvenanceLog

        log_path = temp_dir / "provenance.json"
        log = ProvenanceLog(log_path)

        assert log._records == []
        assert log.path == log_path

    def test_load_existing_log(self, temp_dir):
        """Carga log existente."""
        from hfl.models.provenance import ProvenanceLog

        log_path = temp_dir / "provenance.json"
        existing_data = [{"source_repo": "test/model", "timestamp": "2026-01-01"}]
        log_path.write_text(json.dumps(existing_data))

        log = ProvenanceLog(log_path)

        assert len(log._records) == 1
        assert log._records[0]["source_repo"] == "test/model"

    def test_load_corrupted_json(self, temp_dir):
        """Maneja JSON corrupto."""
        from hfl.models.provenance import ProvenanceLog

        log_path = temp_dir / "provenance.json"
        log_path.write_text("not valid json {{{")

        log = ProvenanceLog(log_path)

        assert log._records == []

    def test_record_conversion(self, temp_dir):
        """Registra una conversión."""
        from hfl.models.provenance import ProvenanceLog, ConversionRecord

        log_path = temp_dir / "provenance.json"
        log = ProvenanceLog(log_path)

        record = ConversionRecord(
            source_repo="test/model",
            source_format="safetensors",
            target_path="/path/to/output.gguf",
            quantization="Q4_K_M",
        )
        log.record(record)

        # Verificar que se guardó
        assert len(log._records) == 1
        assert log_path.exists()

        # Verificar contenido del archivo
        saved_data = json.loads(log_path.read_text())
        assert len(saved_data) == 1
        assert saved_data[0]["source_repo"] == "test/model"

    def test_get_history(self, temp_dir):
        """Obtiene historial de un repo específico."""
        from hfl.models.provenance import ProvenanceLog, ConversionRecord

        log_path = temp_dir / "provenance.json"
        log = ProvenanceLog(log_path)

        # Registrar múltiples conversiones
        log.record(ConversionRecord(source_repo="test/model-a", timestamp="2026-01-01"))
        log.record(ConversionRecord(source_repo="test/model-b", timestamp="2026-01-02"))
        log.record(ConversionRecord(source_repo="test/model-a", timestamp="2026-01-03"))

        history = log.get_history("test/model-a")

        assert len(history) == 2
        assert all(r["source_repo"] == "test/model-a" for r in history)

    def test_get_all(self, temp_dir):
        """Obtiene todos los registros ordenados."""
        from hfl.models.provenance import ProvenanceLog, ConversionRecord

        log_path = temp_dir / "provenance.json"
        log = ProvenanceLog(log_path)

        log.record(ConversionRecord(source_repo="c", timestamp="2026-01-03"))
        log.record(ConversionRecord(source_repo="a", timestamp="2026-01-01"))
        log.record(ConversionRecord(source_repo="b", timestamp="2026-01-02"))

        all_records = log.get_all()

        assert len(all_records) == 3
        # Verificar orden por timestamp
        assert all_records[0]["source_repo"] == "a"
        assert all_records[1]["source_repo"] == "b"
        assert all_records[2]["source_repo"] == "c"

    def test_find_by_target(self, temp_dir):
        """Encuentra registro por ruta de destino."""
        from hfl.models.provenance import ProvenanceLog, ConversionRecord

        log_path = temp_dir / "provenance.json"
        log = ProvenanceLog(log_path)

        log.record(ConversionRecord(
            source_repo="test/model",
            target_path="/path/to/model.gguf",
        ))

        result = log.find_by_target("/path/to/model.gguf")
        assert result is not None
        assert result["source_repo"] == "test/model"

        # No encontrado
        result = log.find_by_target("/nonexistent/path.gguf")
        assert result is None


class TestLogConversionHelper:
    """Tests para la función log_conversion."""

    def test_log_conversion_basic(self, temp_config):
        """Registra conversión básica."""
        from hfl.models.provenance import log_conversion, get_provenance_log
        import hfl.models.provenance as provenance_module

        # Reset global log
        provenance_module._provenance_log = None

        record = log_conversion(
            source_repo="test/model",
            source_format="safetensors",
            target_path="/path/to/output.gguf",
        )

        assert record.source_repo == "test/model"
        assert record.source_format == "safetensors"
        assert record.conversion_type == "format"

    def test_log_conversion_with_quantization(self, temp_config):
        """Registra conversión con cuantización."""
        from hfl.models.provenance import log_conversion
        import hfl.models.provenance as provenance_module

        provenance_module._provenance_log = None

        record = log_conversion(
            source_repo="test/model",
            source_format="safetensors",
            target_path="/path/to/output.gguf",
            quantization="Q4_K_M",
        )

        assert record.quantization == "Q4_K_M"
        assert record.conversion_type == "both"

    def test_log_conversion_with_license(self, temp_config):
        """Registra conversión con info de licencia."""
        from hfl.models.provenance import log_conversion
        import hfl.models.provenance as provenance_module

        provenance_module._provenance_log = None

        record = log_conversion(
            source_repo="test/model",
            source_format="safetensors",
            target_path="/path/to/output.gguf",
            original_license="apache-2.0",
            license_accepted=True,
        )

        assert record.original_license == "apache-2.0"
        assert record.license_accepted is True
        assert record.license_accepted_at  # Debe tener timestamp


class TestGetProvenanceLog:
    """Tests para get_provenance_log singleton."""

    def test_returns_same_instance(self, temp_config):
        """Devuelve la misma instancia."""
        from hfl.models.provenance import get_provenance_log
        import hfl.models.provenance as provenance_module

        provenance_module._provenance_log = None

        log1 = get_provenance_log()
        log2 = get_provenance_log()

        assert log1 is log2

    def test_creates_instance_if_none(self, temp_config):
        """Crea instancia si no existe."""
        from hfl.models.provenance import get_provenance_log
        import hfl.models.provenance as provenance_module

        provenance_module._provenance_log = None

        log = get_provenance_log()

        assert log is not None
        assert provenance_module._provenance_log is log

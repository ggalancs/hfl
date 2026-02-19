# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Conversión de modelos HuggingFace (safetensors/pytorch) a formato GGUF.

Este es el paso más crítico del pipeline. Utiliza las herramientas de
llama.cpp para la conversión y cuantización.

Flujo:
  safetensors → convert_hf_to_gguf.py (FP16) → quantize → GGUF final

Requiere: llama.cpp clonado y compilado en ~/.hfl/tools/llama.cpp

NOTA LEGAL (R3 - Auditoría Legal):
La conversión de formato preserva los pesos del modelo original.
La licencia y restricciones del modelo original siguen vigentes
sobre el archivo convertido. hfl registra la procedencia de
cada conversión para cumplimiento legal.

Modelos soportados para conversión GGUF:
- Modelos de texto (LLMs) con arquitecturas soportadas por llama.cpp
- Requieren config.json con model_type válido

Modelos NO soportados:
- LoRA adapters (archivos adapter_*.safetensors sin modelo base)
- Modelos de imagen (Stable Diffusion, FLUX, etc.)
- Modelos sin config.json
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from hfl.config import config

console = Console()


class UnsupportedModelError(Exception):
    """Raised when a model cannot be converted to GGUF format."""

    pass


# Tipos de modelo que NO pueden convertirse a GGUF
UNSUPPORTED_MODEL_TYPES = {
    # Modelos de imagen
    "stable-diffusion",
    "sdxl",
    "flux",
    "vae",
    "unet",
    "controlnet",
    # LoRA adapters
    "lora",
    "adapter",
    # Otros
    "clip",
    "vit",
    "audio",
    "whisper",
}

# Patrones de archivos que indican modelos no convertibles
UNSUPPORTED_FILE_PATTERNS = {
    "adapter_model.safetensors",  # LoRA adapter
    "adapter_config.json",  # LoRA config
    "diffusion_pytorch_model.safetensors",  # Difusión
    "unet/",  # Stable Diffusion UNet
    "vae/",  # VAE
}


def check_model_convertibility(model_path: Path) -> tuple[bool, str]:
    """
    Verifica si un modelo puede convertirse a formato GGUF.

    Args:
        model_path: Ruta al directorio del modelo descargado

    Returns:
        Tuple (es_convertible, razón)
        - (True, "") si es convertible
        - (False, razón) si no es convertible
    """
    # 1. Verificar que existe config.json
    config_path = model_path / "config.json"
    if not config_path.exists():
        # Verificar si es un LoRA adapter
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            return (
                False,
                "Este es un LoRA adapter, no un modelo completo. "
                "Los LoRA adapters requieren un modelo base para funcionar.",
            )

        # Verificar si hay archivos de difusión
        for pattern in UNSUPPORTED_FILE_PATTERNS:
            if (model_path / pattern).exists() or list(model_path.glob(f"*{pattern}*")):
                return (
                    False,
                    "Este parece ser un modelo de difusión de imágenes. "
                    "GGUF solo soporta modelos de texto (LLMs).",
                )

        return (
            False,
            "No se encontró config.json. Este modelo no tiene el formato "
            "estándar de HuggingFace para modelos de texto.",
        )

    # 2. Leer config.json y verificar model_type
    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return (False, f"No se pudo leer config.json: {e}")

    model_type = config_data.get("model_type", "").lower()

    if not model_type:
        # Sin model_type, verificar otros indicadores
        if "adapter_config" in config_data or "_name_or_path" in str(config_data.get("base_model")):
            return (
                False,
                "Este es un LoRA adapter. Los LoRA adapters requieren "
                "un modelo base para funcionar.",
            )
        return (
            False,
            "config.json no contiene 'model_type'. "
            "El modelo no puede ser identificado.",
        )

    # 3. Verificar si el model_type está en la lista de no soportados
    for unsupported in UNSUPPORTED_MODEL_TYPES:
        if unsupported in model_type:
            return (
                False,
                f"El tipo de modelo '{model_type}' no es soportado para conversión GGUF. "
                "GGUF solo soporta modelos de texto (LLMs).",
            )

    # 4. El modelo parece ser convertible
    return (True, "")


def _get_llama_cpp_version(llama_cpp_dir: Path) -> str:
    """Obtiene la versión/commit de llama.cpp instalado."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=llama_cpp_dir,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class GGUFConverter:
    """Gestiona la conversión de modelos a formato GGUF."""

    def __init__(self):
        self.llama_cpp_dir = config.llama_cpp_dir
        self.convert_script = self.llama_cpp_dir / "convert_hf_to_gguf.py"
        self.quantize_bin = self.llama_cpp_dir / "build" / "bin" / "llama-quantize"

    def ensure_tools(self):
        """
        Verifica que llama.cpp esté disponible.
        Si no lo está, lo clona y compila automáticamente.
        """
        if self.convert_script.exists() and self.quantize_bin.exists():
            return

        console.print("[yellow]Instalando herramientas de conversión (llama.cpp)...[/]")

        if not self.llama_cpp_dir.exists():
            self.llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "https://github.com/ggml-org/llama.cpp.git",
                    str(self.llama_cpp_dir),
                ],
                check=True,
            )

        # Compilar llama.cpp
        build_dir = self.llama_cpp_dir / "build"
        build_dir.mkdir(exist_ok=True)

        # Detectar si hay CUDA disponible
        cuda_flag = "-DGGML_CUDA=ON" if shutil.which("nvcc") else ""

        cmake_cmd = ["cmake", ".."]
        if cuda_flag:
            cmake_cmd.append(cuda_flag)

        subprocess.run(cmake_cmd, cwd=build_dir, check=True)
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j"],
            cwd=build_dir,
            check=True,
        )

        # Instalar dependencias Python del script de conversión
        requirements_file = self.llama_cpp_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
            )

        console.print("[green]Herramientas de conversión listas.[/]")

    def convert(
        self,
        model_path: Path,
        output_path: Path,
        quantization: str = "Q4_K_M",
        source_repo: str = "",
        original_license: str = "",
        license_accepted: bool = False,
    ) -> Path:
        """
        Convierte un modelo HF a GGUF cuantizado.

        Args:
            model_path: Ruta al directorio del modelo HF (con config.json)
            output_path: Ruta base para el archivo de salida
            quantization: Nivel de cuantización (Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16)
            source_repo: Repositorio HuggingFace original (para provenance)
            original_license: Licencia del modelo original
            license_accepted: Si el usuario aceptó la licencia

        Returns:
            Path al archivo GGUF final.
        """
        self.ensure_tools()

        # R3 - Advertencia legal sobre preservación de licencia
        console.print(
            "\n[yellow]Nota:[/] La conversión de formato preserva los pesos del modelo "
            "original. La licencia y restricciones del modelo original siguen "
            "vigentes sobre el archivo convertido.\n"
        )

        # Paso 1: Convertir a GGUF FP16 (formato intermedio)
        fp16_path = output_path.with_suffix(".fp16.gguf")

        console.print("[cyan]Paso 1/2:[/] Convirtiendo a GGUF FP16...")

        subprocess.run(
            [
                sys.executable,
                str(self.convert_script),
                str(model_path),
                "--outtype",
                "f16",
                "--outfile",
                str(fp16_path),
            ],
            check=True,
        )

        if quantization.upper() == "F16":
            # Si piden FP16, ya terminamos
            final_path = output_path.with_suffix(".f16.gguf")
            fp16_path.rename(final_path)
            return final_path

        # Paso 2: Cuantizar al nivel solicitado
        quant = quantization.upper()
        final_path = output_path.with_suffix(f".{quant}.gguf")

        console.print(f"[cyan]Paso 2/2:[/] Cuantizando a {quant}...")

        subprocess.run(
            [
                str(self.quantize_bin),
                str(fp16_path),
                str(final_path),
                quant,
            ],
            check=True,
        )

        # Limpiar el FP16 intermedio
        fp16_path.unlink(missing_ok=True)

        # R3 - Registrar provenance de la conversión
        if source_repo:
            try:
                from hfl.converter.formats import detect_format
                from hfl.models.provenance import log_conversion

                source_format = detect_format(model_path).value
                tool_version = _get_llama_cpp_version(self.llama_cpp_dir)

                log_conversion(
                    source_repo=source_repo,
                    source_format=source_format,
                    target_path=str(final_path),
                    quantization=quant,
                    original_license=original_license,
                    license_accepted=license_accepted,
                    tool_version=tool_version,
                    notes=f"Converted using hfl from {model_path}",
                )
            except Exception as e:
                console.print(f"[dim]Advertencia: No se pudo registrar provenance: {e}[/]")

        console.print(f"[green]Conversión completada:[/] {final_path}")
        return final_path


# Referencia rápida de niveles de cuantización:
#
# Nivel       | Bits/peso | % Calidad | Caso de uso
# ------------|-----------|-----------|----------------------------------
# Q2_K        | ~2.5      | ~80%      | Extrema compresión, baja calidad
# Q3_K_M      | ~3.5      | ~87%      | Poca RAM, calidad aceptable
# Q4_K_M      | ~4.5      | ~92%      | ★ DEFAULT — mejor balance
# Q5_K_M      | ~5.0      | ~96%      | Alta calidad, más RAM
# Q6_K        | ~6.5      | ~97%      | Premium, casi sin pérdida
# Q8_0        | ~8.0      | ~98%+     | Máxima calidad cuantizada
# F16         | 16.0      | 100%      | Sin cuantización

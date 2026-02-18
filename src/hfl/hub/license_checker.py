# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 hfl Contributors
"""
Verificación y clasificación de licencias de modelos.

Cada modelo descargado DEBE pasar por este checker antes
de proceder con la descarga y/o conversión.

Este módulo implementa las recomendaciones de la auditoría legal R1
para mitigar el riesgo de violación de licencias de modelos.
"""

from dataclasses import dataclass
from enum import Enum

from huggingface_hub import HfApi


class LicenseRisk(Enum):
    """Clasificación de riesgo de licencias."""

    PERMISSIVE = "permissive"  # Apache 2.0, MIT — uso libre
    CONDITIONAL = "conditional"  # Llama, Gemma — restricciones específicas
    NON_COMMERCIAL = "non_commercial"  # CC-BY-NC, MRL — NO uso comercial
    RESTRICTED = "restricted"  # MNPL, research-only
    UNKNOWN = "unknown"  # No se pudo determinar


# Licencias conocidas y su clasificación
LICENSE_CLASSIFICATION: dict[str, LicenseRisk] = {
    # Permisivas
    "apache-2.0": LicenseRisk.PERMISSIVE,
    "mit": LicenseRisk.PERMISSIVE,
    "bsd-2-clause": LicenseRisk.PERMISSIVE,
    "bsd-3-clause": LicenseRisk.PERMISSIVE,
    "cc0-1.0": LicenseRisk.PERMISSIVE,
    "unlicense": LicenseRisk.PERMISSIVE,
    "wtfpl": LicenseRisk.PERMISSIVE,
    "cc-by-4.0": LicenseRisk.PERMISSIVE,
    # Condicionales (requieren atención)
    "openrail": LicenseRisk.CONDITIONAL,
    "openrail++": LicenseRisk.CONDITIONAL,
    "bigscience-openrail-m": LicenseRisk.CONDITIONAL,
    "creativeml-openrail-m": LicenseRisk.CONDITIONAL,
    "llama2": LicenseRisk.CONDITIONAL,
    "llama3": LicenseRisk.CONDITIONAL,
    "llama3.1": LicenseRisk.CONDITIONAL,
    "llama3.2": LicenseRisk.CONDITIONAL,
    "llama3.3": LicenseRisk.CONDITIONAL,
    "gemma": LicenseRisk.CONDITIONAL,
    "qwen": LicenseRisk.CONDITIONAL,
    "deepseek": LicenseRisk.CONDITIONAL,
    "cc-by-sa-4.0": LicenseRisk.CONDITIONAL,
    # No comerciales
    "cc-by-nc-4.0": LicenseRisk.NON_COMMERCIAL,
    "cc-by-nc-sa-4.0": LicenseRisk.NON_COMMERCIAL,
    "cc-by-nc-nd-4.0": LicenseRisk.NON_COMMERCIAL,
    # Restringidas
    "other": LicenseRisk.UNKNOWN,
}

# Restricciones específicas por familia de licencia
LICENSE_RESTRICTIONS: dict[str, list[str]] = {
    "llama2": [
        "commercial-use-up-to-700M-MAU",
        "attribution-required: 'Built with Llama'",
        "no-use-to-train-other-models",
        "litigation-termination-clause",
    ],
    "llama3": [
        "commercial-use-up-to-700M-MAU",
        "attribution-required: 'Built with Llama'",
        "no-use-to-train-other-models",
        "litigation-termination-clause",
    ],
    "llama3.1": [
        "commercial-use-up-to-700M-MAU",
        "attribution-required: 'Built with Llama'",
        "no-use-to-train-other-models",
        "litigation-termination-clause",
    ],
    "llama3.2": [
        "commercial-use-up-to-700M-MAU",
        "attribution-required: 'Built with Llama'",
        "no-use-to-train-other-models",
        "litigation-termination-clause",
    ],
    "llama3.3": [
        "commercial-use-up-to-700M-MAU",
        "attribution-required: 'Built with Llama'",
        "no-use-to-train-other-models",
        "litigation-termination-clause",
    ],
    "gemma": [
        "acceptable-use-policy-required",
        "redistribution-must-include-terms",
    ],
    "qwen": [
        "attribution-required",
        "no-use-for-illegal-activities",
    ],
    "deepseek": [
        "attribution-required",
        "no-use-for-illegal-activities",
    ],
    "cc-by-nc-4.0": [
        "non-commercial-only",
        "attribution-required",
    ],
    "cc-by-nc-sa-4.0": [
        "non-commercial-only",
        "attribution-required",
        "share-alike-required",
    ],
    "cc-by-nc-nd-4.0": [
        "non-commercial-only",
        "attribution-required",
        "no-derivatives",
    ],
    "cc-by-sa-4.0": [
        "attribution-required",
        "share-alike-required",
    ],
    "openrail": [
        "use-restrictions-apply",
        "no-harmful-use",
    ],
    "openrail++": [
        "use-restrictions-apply",
        "no-harmful-use",
        "attribution-required",
    ],
    "bigscience-openrail-m": [
        "use-restrictions-apply",
        "no-harmful-use",
        "attribution-required",
    ],
}


@dataclass
class LicenseInfo:
    """Información de licencia de un modelo."""

    license_id: str
    license_name: str
    risk: LicenseRisk
    restrictions: list[str]
    url: str | None
    gated: bool


def check_model_license(repo_id: str, token: str | None = None) -> LicenseInfo:
    """
    Consulta y clasifica la licencia de un modelo de HuggingFace.

    Args:
        repo_id: ID del repositorio (ej: "meta-llama/Llama-3.1-8B")
        token: Token de autenticación opcional

    Returns:
        LicenseInfo con todos los detalles de la licencia.
    """
    api = HfApi()
    info = api.model_info(repo_id, token=token)

    # Intentar obtener licencia de card_data
    license_id = None
    card_data = getattr(info, "card_data", None)
    if card_data and hasattr(card_data, "license"):
        license_id = card_data.license

    # Fallback: buscar en tags
    if not license_id:
        tags = getattr(info, "tags", []) or []
        license_tags = [t.replace("license:", "") for t in tags if t.startswith("license:")]
        license_id = license_tags[0] if license_tags else "other"

    # Normalizar license_id
    license_id = license_id.lower().strip() if license_id else "other"

    # Buscar clasificación
    risk = LICENSE_CLASSIFICATION.get(license_id, LicenseRisk.UNKNOWN)

    # Buscar restricciones (intentar coincidencia parcial para variantes)
    restrictions = []
    for key, restr in LICENSE_RESTRICTIONS.items():
        if key in license_id or license_id in key:
            restrictions = restr
            break

    # Detectar si es gated
    gated = getattr(info, "gated", False) or False

    return LicenseInfo(
        license_id=license_id,
        license_name=license_id.replace("-", " ").title(),
        risk=risk,
        restrictions=restrictions,
        url=f"https://huggingface.co/{repo_id}#license",
        gated=bool(gated),
    )


def require_user_acceptance(license_info: LicenseInfo, repo_id: str) -> bool:
    """
    Presenta la licencia al usuario y requiere aceptación explícita.

    Obligatorio para licencias CONDITIONAL, NON_COMMERCIAL, RESTRICTED y UNKNOWN.

    Args:
        license_info: Información de la licencia
        repo_id: ID del repositorio

    Returns:
        True si el usuario acepta, False si rechaza.
    """
    from rich.console import Console
    from rich.panel import Panel
    import typer

    console = Console()

    # Licencias permisivas no requieren confirmación
    if license_info.risk == LicenseRisk.PERMISSIVE:
        console.print(f"  [green]Licencia:[/] {license_info.license_id} (permisiva)")
        return True

    # Construir mensaje según nivel de riesgo
    if license_info.risk == LicenseRisk.NON_COMMERCIAL:
        color = "red"
        title = "LICENCIA NO COMERCIAL"
        warning = "Este modelo NO puede usarse con fines comerciales."
    elif license_info.risk == LicenseRisk.RESTRICTED:
        color = "red"
        title = "LICENCIA RESTRINGIDA"
        warning = "Este modelo tiene restricciones severas de uso."
    elif license_info.risk == LicenseRisk.UNKNOWN:
        color = "yellow"
        title = "LICENCIA DESCONOCIDA"
        warning = "No se pudo determinar la licencia. Revise los términos manualmente."
    else:  # CONDITIONAL
        color = "yellow"
        title = "LICENCIA CON RESTRICCIONES"
        warning = "Este modelo tiene condiciones de uso específicas."

    # Construir panel de información
    restrictions_text = ""
    if license_info.restrictions:
        restrictions_text = "\n\nRestricciones:\n" + "\n".join(
            f"  - {r}" for r in license_info.restrictions
        )

    gated_text = ""
    if license_info.gated:
        gated_text = "\n\n[bold]Modelo Gated:[/] Requiere aceptación previa en huggingface.co"

    console.print(
        Panel(
            f"[bold {color}]{title}[/]\n\n"
            f"Modelo: {repo_id}\n"
            f"Licencia: {license_info.license_id}\n\n"
            f"{warning}"
            f"{restrictions_text}"
            f"{gated_text}"
            f"\n\nDetalles: {license_info.url}",
            title="Términos de Uso",
            border_style=color,
        )
    )

    return typer.confirm(
        "¿Aceptas los términos de esta licencia y confirmas que tu uso es conforme?",
        default=False,
    )


def get_license_summary(license_info: LicenseInfo) -> str:
    """
    Genera un resumen corto de la licencia para mostrar en tablas.

    Returns:
        String corto con el ID y emoji indicador de riesgo.
    """
    risk_emoji = {
        LicenseRisk.PERMISSIVE: "[green]OK[/]",
        LicenseRisk.CONDITIONAL: "[yellow]![/]",
        LicenseRisk.NON_COMMERCIAL: "[red]NC[/]",
        LicenseRisk.RESTRICTED: "[red]R[/]",
        LicenseRisk.UNKNOWN: "[yellow]?[/]",
    }
    emoji = risk_emoji.get(license_info.risk, "?")
    return f"{license_info.license_id} {emoji}"

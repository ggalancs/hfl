# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Model license verification and classification.

Each downloaded model MUST pass through this checker before
proceeding with the download and/or conversion.

This module implements the recommendations from legal audit R1
to mitigate the risk of model license violations.
"""

from dataclasses import dataclass
from enum import Enum

from huggingface_hub import HfApi


class LicenseRisk(Enum):
    """License risk classification."""

    PERMISSIVE = "permissive"  # Apache 2.0, MIT - free use
    CONDITIONAL = "conditional"  # Llama, Gemma - specific restrictions
    NON_COMMERCIAL = "non_commercial"  # CC-BY-NC, MRL - NO commercial use
    RESTRICTED = "restricted"  # MNPL, research-only
    UNKNOWN = "unknown"  # Could not be determined


# Known licenses and their classification
LICENSE_CLASSIFICATION: dict[str, LicenseRisk] = {
    # Permissive
    "apache-2.0": LicenseRisk.PERMISSIVE,
    "mit": LicenseRisk.PERMISSIVE,
    "bsd-2-clause": LicenseRisk.PERMISSIVE,
    "bsd-3-clause": LicenseRisk.PERMISSIVE,
    "cc0-1.0": LicenseRisk.PERMISSIVE,
    "unlicense": LicenseRisk.PERMISSIVE,
    "wtfpl": LicenseRisk.PERMISSIVE,
    "cc-by-4.0": LicenseRisk.PERMISSIVE,
    # Conditional (require attention)
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
    # Non-commercial
    "cc-by-nc-4.0": LicenseRisk.NON_COMMERCIAL,
    "cc-by-nc-sa-4.0": LicenseRisk.NON_COMMERCIAL,
    "cc-by-nc-nd-4.0": LicenseRisk.NON_COMMERCIAL,
    # Restricted
    "other": LicenseRisk.UNKNOWN,
}

# Specific restrictions by license family
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
    """License information for a model."""

    license_id: str
    license_name: str
    risk: LicenseRisk
    restrictions: list[str]
    url: str | None
    gated: bool


def check_model_license(repo_id: str, token: str | None = None) -> LicenseInfo:
    """
    Query and classify the license of a HuggingFace model.

    Args:
        repo_id: Repository ID (e.g.: "meta-llama/Llama-3.1-8B")
        token: Optional authentication token

    Returns:
        LicenseInfo with all license details.
    """
    api = HfApi()
    info = api.model_info(repo_id, token=token)

    # Try to get license from card_data
    license_id = None
    license_name = None
    license_url = None
    card_data = getattr(info, "card_data", None)

    if card_data:
        # Get basic license
        if hasattr(card_data, "license"):
            license_id = card_data.license

        # Get license_name (more specific, e.g.: "qwen2" when license is "other")
        if hasattr(card_data, "license_name"):
            license_name = card_data.license_name

        # Get license_link if it exists
        if hasattr(card_data, "license_link"):
            license_url = card_data.license_link

    # Fallback: search in tags
    if not license_id:
        tags = getattr(info, "tags", []) or []
        license_tags = [t.replace("license:", "") for t in tags if t.startswith("license:")]
        license_id = license_tags[0] if license_tags else "other"

    # Normalize license_id and license_name
    license_id = license_id.lower().strip() if license_id else "other"
    license_name_normalized = license_name.lower().strip() if license_name else None

    # If license_id is "other" but we have license_name, use license_name for classification
    classification_key = license_id
    if license_id == "other" and license_name_normalized:
        classification_key = license_name_normalized

    # Search for classification
    risk = LICENSE_CLASSIFICATION.get(classification_key, LicenseRisk.UNKNOWN)

    # If still UNKNOWN but we have license_name, search for partial match
    if risk == LicenseRisk.UNKNOWN and license_name_normalized:
        for known_license, known_risk in LICENSE_CLASSIFICATION.items():
            if known_license in license_name_normalized or license_name_normalized in known_license:
                risk = known_risk
                classification_key = known_license
                break

    # Search for restrictions (try partial match for variants)
    restrictions = []
    for key, restr in LICENSE_RESTRICTIONS.items():
        if key in classification_key or classification_key in key:
            restrictions = restr
            break

    # Detect if gated
    gated = getattr(info, "gated", False) or False

    # Determine license name for display
    display_name = license_name if license_name else license_id.replace("-", " ").title()

    # Determine license URL
    if not license_url:
        license_url = f"https://huggingface.co/{repo_id}#license"

    return LicenseInfo(
        license_id=license_name if license_name else license_id,
        license_name=display_name,
        risk=risk,
        restrictions=restrictions,
        url=license_url,
        gated=bool(gated),
    )


def require_user_acceptance(license_info: LicenseInfo, repo_id: str) -> bool:
    """
    Present the license to the user and require explicit acceptance.

    Mandatory for CONDITIONAL, NON_COMMERCIAL, RESTRICTED and UNKNOWN licenses.

    Args:
        license_info: License information
        repo_id: Repository ID

    Returns:
        True if user accepts, False if user rejects.
    """
    import typer
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    # Permissive licenses do not require confirmation
    if license_info.risk == LicenseRisk.PERMISSIVE:
        console.print(f"  [green]License:[/] {license_info.license_id} (permissive)")
        return True

    # Build message according to risk level
    if license_info.risk == LicenseRisk.NON_COMMERCIAL:
        color = "red"
        title = "NON-COMMERCIAL LICENSE"
        warning = "This model CANNOT be used for commercial purposes."
    elif license_info.risk == LicenseRisk.RESTRICTED:
        color = "red"
        title = "RESTRICTED LICENSE"
        warning = "This model has severe use restrictions."
    elif license_info.risk == LicenseRisk.UNKNOWN:
        color = "yellow"
        title = "UNKNOWN LICENSE"
        warning = "Could not determine the license. Please review the terms manually."
    else:  # CONDITIONAL
        color = "yellow"
        title = "LICENSE WITH RESTRICTIONS"
        warning = "This model has specific terms of use."

    # Build information panel
    restrictions_text = ""
    if license_info.restrictions:
        restrictions_text = "\n\nRestrictions:\n" + "\n".join(
            f"  - {r}" for r in license_info.restrictions
        )

    gated_text = ""
    if license_info.gated:
        gated_text = "\n\n[bold]Gated Model:[/] Requires prior acceptance at huggingface.co"

    console.print(
        Panel(
            f"[bold {color}]{title}[/]\n\n"
            f"Model: {repo_id}\n"
            f"License: {license_info.license_id}\n\n"
            f"{warning}"
            f"{restrictions_text}"
            f"{gated_text}"
            f"\n\nDetails: {license_info.url}",
            title="Terms of Use",
            border_style=color,
        )
    )

    return typer.confirm(
        "Do you accept the terms of this license and confirm your use is compliant?",
        default=False,
    )


def get_license_summary(license_info: LicenseInfo) -> str:
    """
    Generate a short license summary for display in tables.

    Returns:
        Short string with the ID and risk indicator emoji.
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

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Gestión de autenticación con HuggingFace Hub.

IMPORTANTE - Cumplimiento con ToS de HuggingFace (R8 - Auditoría Legal):

hfl respeta el sistema de gating de HuggingFace. Si un modelo requiere
aceptación de licencia ("gated model"), el usuario DEBE haberla aceptado
previamente en huggingface.co.

hfl NO bypasea ni automatiza la aceptación de licencias gated.
El flujo correcto es:
1. Usuario visita huggingface.co/<repo_id>
2. Usuario lee y acepta los términos de la licencia
3. Usuario genera un token con permisos de lectura
4. hfl usa el token para descargar el modelo

Este diseño garantiza que:
- Los usuarios leen los términos de licencia
- Los autores de modelos pueden rastrear quién acepta sus términos
- hfl cumple con los Terms of Service de HuggingFace
"""

from huggingface_hub import HfApi, get_token

from hfl.config import config


def get_hf_token() -> str | None:
    """
    Obtiene el token de HuggingFace de las fuentes disponibles.

    Orden de prioridad:
    1. Variable de entorno HF_TOKEN (desde config)
    2. Token guardado por huggingface_hub (via 'hfl login' o 'huggingface-cli login')
    """
    # Primero intentar la variable de entorno
    if config.hf_token:
        return config.hf_token

    # Luego el token guardado por huggingface_hub
    try:
        return get_token()
    except Exception:
        return None


def ensure_auth(repo_id: str) -> str | None:
    """
    Verifica si el modelo requiere autenticación y gestiona el token.

    Modelos "gated" (ej: meta-llama/*) requieren:
    1. Aceptar la licencia en huggingface.co
    2. Token de acceso con permisos de lectura

    NOTA: hfl NO bypasea el sistema de gating. Si el usuario no ha
    aceptado la licencia en huggingface.co, la descarga fallará.

    Returns: token válido o None si no se necesita.
    """
    api = HfApi()
    token = get_hf_token()

    # Intentar acceder con el token disponible (si hay)
    try:
        api.model_info(repo_id, token=token)
        return token
    except Exception:
        pass

    # Si no hay token, pedir uno interactivamente
    if not token:
        from rich.console import Console
        from rich.prompt import Prompt

        console = Console()

        console.print("\n[yellow]Este modelo requiere autenticación HuggingFace.[/]")
        console.print("Puedes configurar tu token de forma permanente con: [cyan]hfl login[/]")
        console.print("O introduce tu token ahora (https://huggingface.co/settings/tokens):\n")

        token = Prompt.ask("Token HF")

    try:
        api.model_info(repo_id, token=token)
        return token
    except Exception as e:
        raise RuntimeError(
            f"No se puede acceder a {repo_id}. "
            f"Verifica que has aceptado la licencia en huggingface.co/{repo_id} "
            f"y que tu token tiene permisos de lectura.\nError: {e}"
        )

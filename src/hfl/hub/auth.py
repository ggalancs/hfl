# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
HuggingFace Hub authentication management.

IMPORTANT - Compliance with HuggingFace ToS (R8 - Legal Audit):

hfl respects the HuggingFace gating system. If a model requires
license acceptance ("gated model"), the user MUST have accepted it
previously at huggingface.co.

hfl does NOT bypass or automate gated license acceptance.
The correct flow is:
1. User visits huggingface.co/<repo_id>
2. User reads and accepts the license terms
3. User generates a token with read permissions
4. hfl uses the token to download the model

This design ensures that:
- Users read the license terms
- Model authors can track who accepts their terms
- hfl complies with HuggingFace Terms of Service
"""

from huggingface_hub import HfApi, get_token

from hfl.config import config


def get_hf_token() -> str | None:
    """
    Get the HuggingFace token from available sources.

    Priority order:
    1. HF_TOKEN environment variable (from config)
    2. Token saved by huggingface_hub (via 'hfl login' or 'huggingface-cli login')
    """
    # First try the environment variable
    if config.hf_token:
        return config.hf_token

    # Then the token saved by huggingface_hub
    try:
        return get_token()
    except Exception:
        return None


def ensure_auth(repo_id: str) -> str | None:
    """
    Check if the model requires authentication and manage the token.

    "Gated" models (e.g.: meta-llama/*) require:
    1. Accepting the license at huggingface.co
    2. Access token with read permissions

    NOTE: hfl does NOT bypass the gating system. If the user has not
    accepted the license at huggingface.co, the download will fail.

    Returns: valid token or None if not needed.
    """
    api = HfApi()
    token = get_hf_token()

    # Try to access with the available token (if any)
    try:
        api.model_info(repo_id, token=token)
        return token
    except Exception:
        pass

    # If there is no token, request one interactively
    if not token:
        from rich.console import Console
        from rich.prompt import Prompt

        console = Console()

        console.print("\n[yellow]This model requires HuggingFace authentication.[/]")
        console.print("You can configure your token permanently with: [cyan]hfl login[/]")
        console.print("Or enter your token now (https://huggingface.co/settings/tokens):\n")

        token = Prompt.ask("HF Token")

    try:
        api.model_info(repo_id, token=token)
        return token
    except Exception as e:
        raise RuntimeError(
            f"Cannot access {repo_id}. "
            f"Verify that you have accepted the license at huggingface.co/{repo_id} "
            f"and that your token has read permissions.\nError: {e}"
        )

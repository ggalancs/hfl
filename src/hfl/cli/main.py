# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Main CLI for hfl.

Usage:
  hfl pull <model> [--quantize Q4_K_M]
  hfl run <model> [--backend auto]
  hfl serve [--port 11434]
  hfl list
  hfl search <text>
  hfl rm <model>
  hfl inspect <model>
"""

import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="hfl",
    help="Run HuggingFace models locally.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def pull(
    model: str = typer.Argument(help="HF model (e.g.: meta-llama/Llama-3.3-70B-Instruct)"),
    quantize: str = typer.Option("Q4_K_M", "--quantize", "-q", help="Quantization level"),
    format: str = typer.Option("auto", "--format", "-f", help="Format: auto, gguf, safetensors"),
    alias: str = typer.Option(
        None, "--alias", "-a", help="Short alias to refer to the model (e.g.: 'coder')"
    ),
    skip_license: bool = typer.Option(
        False, "--skip-license", help="Skip license verification"
    ),
):
    """Download a model from HuggingFace Hub."""
    from datetime import datetime

    from hfl.converter.formats import ModelFormat, detect_format
    from hfl.hub.downloader import pull_model
    from hfl.hub.license_checker import check_model_license, require_user_acceptance
    from hfl.hub.resolver import resolve
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import ModelRegistry

    # 1. Resolve model
    console.print(f"[bold]Resolving[/] {model}...")
    try:
        resolved = resolve(model, quantization=quantize)
    except (ValueError, Exception) as e:
        error_msg = str(e)
        if "Repo id must" in error_msg or "repo_name" in error_msg:
            console.print(f"[red]Format error:[/] The model name is invalid.")
            console.print(f"\n[yellow]Supported formats:[/]")
            console.print("  - org/model                  -> direct HuggingFace repo")
            console.print("  - org/model:Q4_K_M           -> repo with quantization")
            console.print("  - model-name                 -> search by name")
            console.print(f"\n[dim]Input received:[/] {model}")
            console.print(f"[dim]Detail:[/] {e}")
        elif "No se encontró modelo" in error_msg:
            console.print(f"[red]Error:[/] {e}")
            console.print("[dim]Check the name or use 'hfl search' to search.[/]")
        else:
            console.print(f"[red]Error resolving model:[/] {e}")
        raise typer.Exit(1)

    console.print(f"  Repo: {resolved.repo_id}")
    console.print(f"  Format: {resolved.format}")
    if resolved.filename:
        console.print(f"  File: {resolved.filename}")

    # 2. Verify license (R1 - Legal Audit)
    license_info = None
    license_accepted_at = None
    if not skip_license:
        try:
            license_info = check_model_license(resolved.repo_id)
            if not require_user_acceptance(license_info, resolved.repo_id):
                console.print("[yellow]Download cancelled.[/]")
                raise typer.Exit(0)
            license_accepted_at = datetime.now().isoformat()
        except Exception as e:
            console.print(f"[yellow]Warning:[/] Could not verify license: {e}")
            if not typer.confirm("Continue without verifying license?", default=False):
                raise typer.Exit(0)

    # 3. Download
    local_path = pull_model(resolved)
    console.print(f"[green]Downloaded to:[/] {local_path}")

    # 3. Convert if necessary
    fmt = detect_format(local_path)
    final_path = local_path

    if fmt != ModelFormat.GGUF and format != "safetensors":
        # Check if the model can be converted to GGUF
        from hfl.converter.gguf_converter import GGUFConverter, check_model_convertibility

        is_convertible, reason = check_model_convertibility(local_path)

        if not is_convertible:
            console.print(f"\n[yellow]Cannot convert to GGUF:[/] {reason}")
            console.print(
                "\n[dim]The model has been downloaded but cannot be run with llama.cpp.[/]"
            )
            console.print("[dim]Consider searching for a GGUF version of the model:[/]")
            console.print(f"  hfl search {resolved.repo_id.split('/')[-1]} --gguf\n")
            raise typer.Exit(1)

        console.print(f"[yellow]Converting to GGUF ({quantize})...[/]")
        converter = GGUFConverter()
        output_name = resolved.repo_id.replace("/", "--")
        output_path = local_path.parent / output_name
        final_path = converter.convert(local_path, output_path, quantize)

    # 4. Register
    size = sum(
        f.stat().st_size
        for f in (final_path.rglob("*") if final_path.is_dir() else [final_path])
        if f.is_file()
    )

    short_name = resolved.repo_id.split("/")[-1].lower()
    if resolved.quantization:
        short_name += f"-{resolved.quantization.lower()}"
    elif quantize:
        short_name += f"-{quantize.lower()}"

    manifest = ModelManifest(
        name=short_name,
        repo_id=resolved.repo_id,
        alias=alias,
        local_path=str(final_path),
        format=detect_format(final_path).value,
        size_bytes=size,
        quantization=resolved.quantization or quantize,
        # R1 - License information
        license=license_info.license_id if license_info else None,
        license_name=license_info.license_name if license_info else None,
        license_url=license_info.url if license_info else None,
        license_restrictions=license_info.restrictions if license_info else [],
        gated=license_info.gated if license_info else False,
        license_accepted_at=license_accepted_at,
    )

    registry = ModelRegistry()
    registry.add(manifest)

    # Show result with alias if defined
    if alias:
        console.print(f"\n[bold green]Model ready:[/] {manifest.name} ({manifest.display_size})")
        console.print(f"[cyan]Alias:[/] {alias}")
        console.print(f"[dim]Use:[/] hfl run {alias}")
    else:
        console.print(f"\n[bold green]Model ready:[/] {manifest.name} ({manifest.display_size})")


@app.command()
def run(
    model: str = typer.Argument(help="Local model name"),
    backend: str = typer.Option("auto", "--backend", "-b"),
    ctx: int = typer.Option(4096, "--ctx", "-c", help="Context size"),
    system: str = typer.Option(None, "--system", "-s", help="System prompt"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show backend logs (Metal, CUDA)"
    ),
):
    """Start an interactive chat with a model."""
    from pathlib import Path

    from hfl.engine.base import ChatMessage
    from hfl.engine.selector import MissingDependencyError, select_engine
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)
    if not manifest:
        console.print(f"[red]Model not found:[/] {model}")
        console.print("Use 'hfl list' to see available models.")
        raise typer.Exit(1)

    console.print(f"[cyan]Loading[/] {manifest.name}...")
    try:
        engine = select_engine(Path(manifest.local_path), backend=backend)
        engine.load(manifest.local_path, n_ctx=ctx, verbose=verbose)
    except MissingDependencyError as e:
        console.print(f"[red]Missing dependency:[/]\n\n{e}")
        raise typer.Exit(1)
    console.print(f"[green]Model loaded.[/] Type '/exit' to quit.\n")

    # R9 - Legal disclaimer before starting chat
    console.print(
        "[dim]AI models may generate incorrect, biased, or inappropriate "
        "information. The user is responsible for the use they make "
        "of the generated responses.[/]\n"
    )

    messages: list[ChatMessage] = []
    if system:
        messages.append(ChatMessage(role="system", content=system))

    while True:
        try:
            user_input = console.input("[bold blue]>>> [/]")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ("/exit", "/quit", "/bye"):
            break
        if not user_input.strip():
            continue

        messages.append(ChatMessage(role="user", content=user_input))

        # Response streaming with style
        # markup=False prevents Rich from interpreting [] as format tags
        from rich.style import Style

        green_style = Style(color="green")

        full_response = []
        for token in engine.chat_stream(messages):
            console.print(token, end="", highlight=False, markup=False, style=green_style)
            full_response.append(token)
        console.print()  # New line at the end

        messages.append(ChatMessage(role="assistant", content="".join(full_response)))

    engine.unload()
    console.print("\n[dim]Session ended.[/]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(11434, "--port", "-p"),
    model: str = typer.Option(None, "--model", "-m", help="Pre-load model"),
):
    """Start the API server (OpenAI + Ollama compatible)."""
    from hfl.api.server import start_server, state

    # R6 - Privacy warning when exposing to the network
    if host == "0.0.0.0":
        console.print(
            "[yellow]Warning:[/] Exposing the server to the network. "
            "Prompts sent to the API will be accessible from "
            "any device on the network."
        )
        if not typer.confirm("Continue?", default=True):
            raise typer.Exit(0)

    if model:
        from pathlib import Path

        from hfl.engine.selector import MissingDependencyError, select_engine
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        manifest = registry.get(model)
        if manifest:
            console.print(f"[cyan]Pre-loading[/] {manifest.name}...")
            try:
                state.engine = select_engine(Path(manifest.local_path))
                state.engine.load(manifest.local_path)
                state.current_model = manifest
            except MissingDependencyError as e:
                console.print(f"[red]Missing dependency:[/]\n\n{e}")
                raise typer.Exit(1)

    console.print(f"[bold green]hfl server[/] at http://{host}:{port}")
    console.print("  OpenAI: POST /v1/chat/completions")
    console.print("  Ollama: POST /api/chat")
    start_server(host=host, port=port)


@app.command(name="list")
def list_models():
    """List all downloaded models."""
    from rich.table import Table

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    models = registry.list_all()

    if not models:
        console.print("[dim]No downloaded models. Use 'hfl pull' to download.[/]")
        return

    table = Table(title="Local Models")
    table.add_column("Name", style="cyan")
    table.add_column("Alias", style="green")
    table.add_column("Format")
    table.add_column("Quantization")
    table.add_column("License")
    table.add_column("Size", justify="right")

    for m in models:
        # R1 - Show license with risk indicator
        license_str = m.license or "?"
        if m.license:
            # Risk indicator based on license type
            nc_licenses = ["cc-by-nc", "mrl", "mnpl"]
            if any(nc in m.license.lower() for nc in nc_licenses):
                license_str = f"[red]{m.license}[/]"
            elif m.license.lower() in ["apache-2.0", "mit", "bsd"]:
                license_str = f"[green]{m.license}[/]"
            else:
                license_str = f"[yellow]{m.license}[/]"

        table.add_row(
            m.name,
            m.alias or "-",
            m.format,
            m.quantization or "-",
            license_str,
            m.display_size,
        )

    console.print(table)


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    if size_bytes == 0:
        return "N/A"
    gb = size_bytes / (1024**3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024**2)
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{size_bytes} B"


def _get_key() -> str:
    """Read a key without requiring Enter to be pressed."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def _extract_params_from_name(model_id: str) -> str | None:
    """Extract the number of parameters from the model name (e.g.: '70B', '7B')."""
    import re

    name = model_id.lower()
    # Patterns: 70b, 7b, 1.5b, 0.5b, 405b, etc.
    patterns = [
        r"(\d+\.?\d*)b(?:[-_]|$)",  # 70b, 7b, 1.5b
        r"(\d+)b-",  # 70b-instruct
        r"-(\d+\.?\d*)b",  # model-7b
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1) + "B"
    return None


def _estimate_model_size(params_str: str | None, quantization: str = "Q4") -> str:
    """Estimate model size based on parameters and quantization."""
    if not params_str:
        return "?"

    try:
        # Parse params (e.g., "70B" -> 70, "1.5B" -> 1.5)
        params = float(params_str.replace("B", "").replace("b", ""))

        # Bytes per parameter depends on quantization
        # Q4_K_M ~ 4.5 bits/param, Q8_0 ~ 8 bits/param, F16 ~ 16 bits
        bits_per_param = {
            "Q2": 2.5,
            "Q3": 3.5,
            "Q4": 4.5,
            "Q5": 5.5,
            "Q6": 6.5,
            "Q8": 8.0,
            "F1": 16.0,  # F16
        }
        bits = bits_per_param.get(quantization[:2].upper(), 4.5)
        size_gb = (params * 1e9 * bits / 8) / (1024**3)

        if size_gb >= 100:
            return f"{size_gb:.0f}GB"
        elif size_gb >= 10:
            return f"{size_gb:.0f}GB"
        else:
            return f"{size_gb:.1f}GB"
    except Exception:
        return "?"


def _display_model_row(model, index: int, show_index: bool = True) -> None:
    """Display a formatted model row."""
    # Get model information
    model_id = model.id
    downloads = getattr(model, "downloads", 0) or 0
    likes = getattr(model, "likes", 0) or 0

    # Detect if it has GGUF
    has_gguf = False
    siblings = getattr(model, "siblings", None)
    if siblings:
        has_gguf = any(s.rfilename.endswith(".gguf") for s in siblings)

    # Format tags
    tags = getattr(model, "tags", []) or []
    pipeline_tag = getattr(model, "pipeline_tag", None)

    # Format icon
    format_icon = "[green]●[/] GGUF" if has_gguf else "[dim]○[/] HF"

    # Format downloads
    if downloads >= 1_000_000:
        dl_str = f"{downloads / 1_000_000:.1f}M"
    elif downloads >= 1_000:
        dl_str = f"{downloads / 1_000:.1f}K"
    else:
        dl_str = str(downloads)

    # Extract parameters and estimate size
    params = _extract_params_from_name(model_id)
    size_q4 = _estimate_model_size(params, "Q4")
    size_str = f"[magenta]~{size_q4}[/]" if params else ""

    # Index number
    idx_str = f"[dim]{index:3}.[/] " if show_index else ""

    console.print(
        f"{idx_str}[bold cyan]{model_id}[/]  "
        f"{format_icon}  "
        f"[yellow]↓{dl_str}[/]  "
        f"[red]♥{likes}[/]  "
        f"{size_str}  "
        f"[dim]{pipeline_tag or ''}[/]"
    )


def _get_params_value(model_id: str) -> float | None:
    """Extract the numeric value of parameters from the name (e.g.: 70 for '70B')."""
    params = _extract_params_from_name(model_id)
    if not params:
        return None
    try:
        return float(params.replace("B", "").replace("b", ""))
    except ValueError:
        return None


@app.command()
def search(
    query: str = typer.Argument(help="Text to search (minimum 3 characters)"),
    limit: int = typer.Option(100, "--limit", "-l", help="Maximum number of results"),
    page_size: int = typer.Option(10, "--page-size", "-n", help="Results per page"),
    gguf_only: bool = typer.Option(False, "--gguf", "-g", help="Show only models with GGUF"),
    max_params: float = typer.Option(
        None, "--max-params", "-p", help="Maximum parameters in B (e.g.: 70 for <70B)"
    ),
    min_params: float = typer.Option(
        None, "--min-params", help="Minimum parameters in B (e.g.: 7 for >7B)"
    ),
    sort: str = typer.Option(
        "downloads", "--sort", "-s", help="Sort by: downloads, likes, created"
    ),
):
    """
    Search models on HuggingFace Hub with interactive pagination.

    Controls:
      SPACE/ENTER    Next page
      q/Q/ESC        Exit
      p              Previous page (if available)

    Examples:
      hfl search llama
      hfl search mistral --gguf
      hfl search phi --limit 50 --page-size 5
      hfl search qwen --max-params 14        # Models less than 14B
      hfl search llama --min-params 70       # Models 70B or more
    """
    from huggingface_hub import HfApi

    # Validate minimum length
    if len(query.strip()) < 3:
        console.print("[red]Error:[/] Search must have at least 3 characters.")
        raise typer.Exit(1)

    console.print(f"[bold]Searching[/] '{query}' on HuggingFace Hub...\n")

    api = HfApi()

    try:
        # Search models
        models = list(
            api.list_models(
                search=query,
                sort=sort,
                direction=-1,
                limit=limit,
                fetch_config=False,
                full=True,  # To get siblings and detect GGUF
            )
        )
    except Exception as e:
        console.print(f"[red]Error searching:[/] {e}")
        raise typer.Exit(1)

    if not models:
        console.print(f"[yellow]No models found for:[/] '{query}'")
        return

    # Filter by GGUF if requested
    if gguf_only:
        models = [
            m
            for m in models
            if hasattr(m, "siblings")
            and m.siblings
            and any(s.rfilename.endswith(".gguf") for s in m.siblings)
        ]
        if not models:
            console.print(f"[yellow]No GGUF models found for:[/] '{query}'")
            return

    # Filter by number of parameters
    if max_params is not None or min_params is not None:
        filtered = []
        for m in models:
            params = _get_params_value(m.id)
            if params is None:
                continue  # Exclude models without detectable parameters
            if max_params is not None and params > max_params:
                continue
            if min_params is not None and params < min_params:
                continue
            filtered.append(m)
        models = filtered

        if not models:
            filter_desc = []
            if max_params is not None:
                filter_desc.append(f"<{max_params}B")
            if min_params is not None:
                filter_desc.append(f">{min_params}B")
            console.print(
                f"[yellow]No models {' and '.join(filter_desc)} found for:[/] '{query}'"
            )
            return

    total = len(models)
    total_pages = (total + page_size - 1) // page_size
    current_page = 0

    # Show header
    console.print(
        Panel(
            f"[bold]{total}[/] models found  |  "
            f"[dim]SPACE/ENTER[/] continue  |  "
            f"[dim]q[/] quit  |  "
            f"[dim]p[/] previous",
            title=f"[bold cyan]Search: {query}[/]",
            border_style="cyan",
        )
    )
    console.print()

    while current_page < total_pages:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total)
        page_models = models[start_idx:end_idx]

        # Show models of the current page
        for i, model in enumerate(page_models):
            _display_model_row(model, start_idx + i + 1)

        # Show pagination status
        console.print()
        page_info = f"[dim]-- Page {current_page + 1}/{total_pages} ({start_idx + 1}-{end_idx} of {total}) --[/]"

        if current_page < total_pages - 1:
            console.print(f"{page_info}  [dim]SPACE[/] more  [dim]q[/] quit", end="")

            # Wait for user input
            try:
                key = _get_key()
            except Exception:
                # Fallback if no interactive terminal
                try:
                    user_input = input("\n[Press ENTER to continue, 'q' to quit]: ")
                    key = "q" if user_input.lower() == "q" else " "
                except (EOFError, KeyboardInterrupt):
                    key = "q"

            # Clear status line
            console.print("\r" + " " * 80 + "\r", end="")

            if key in ("q", "Q", "\x1b", "\x03"):  # q, Q, ESC, Ctrl+C
                console.print(
                    f"\n[dim]Search finished. Showing {end_idx} of {total} results.[/]"
                )
                break
            elif key == "p" and current_page > 0:
                current_page -= 1
                console.print()  # New line before previous page
            else:
                current_page += 1
                console.print()  # New line before next page
        else:
            # Last page
            console.print(f"{page_info}  [bold green]End of results[/]")
            current_page += 1

    # Show help at the end
    console.print()
    console.print("[dim]To download a model use:[/] hfl pull <model>")


@app.command()
def rm(model: str = typer.Argument(help="Name of the model to delete")):
    """Delete a local model."""
    import shutil
    from pathlib import Path

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]Model not found:[/] {model}")
        raise typer.Exit(1)

    # Confirm
    confirm = typer.confirm(
        f"Delete {manifest.name} ({manifest.display_size})?",
    )
    if not confirm:
        return

    # Delete files
    path = Path(manifest.local_path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()

    registry.remove(model)
    console.print(f"[green]Deleted:[/] {manifest.name}")


@app.command()
def inspect(model: str = typer.Argument(help="Model name")):
    """Show detailed information about a model."""
    from rich.panel import Panel
    from rich.text import Text

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]Model not found:[/] {model}")
        raise typer.Exit(1)

    info = Text()
    info.append(f"Name:          {manifest.name}\n")
    if manifest.alias:
        info.append(f"Alias:         {manifest.alias}\n")
    info.append(f"HF Repo:       {manifest.repo_id}\n")
    info.append(f"Local path:    {manifest.local_path}\n")
    info.append(f"Format:        {manifest.format}\n")
    info.append(f"Quantization:  {manifest.quantization or 'N/A'}\n")
    info.append(f"Architecture:  {manifest.architecture or 'auto-detect'}\n")
    info.append(f"Parameters:    {manifest.parameters or 'unknown'}\n")
    info.append(f"Context:       {manifest.context_length} tokens\n")
    info.append(f"Size:          {manifest.display_size}\n")
    info.append(f"Downloaded:    {manifest.created_at}\n")

    # R1 - Show license information
    info.append("\n[License]\n")
    info.append(f"License:       {manifest.license or 'unknown'}\n")
    if manifest.license_url:
        info.append(f"URL:           {manifest.license_url}\n")
    if manifest.gated:
        info.append("Gated:         Yes (required acceptance on HF)\n")
    if manifest.license_restrictions:
        info.append("Restrictions:\n")
        for r in manifest.license_restrictions:
            info.append(f"  - {r}\n")
    if manifest.license_accepted_at:
        info.append(f"Accepted:      {manifest.license_accepted_at[:10]}\n")

    console.print(Panel(info, title=f"[bold]{manifest.name}[/]", border_style="cyan"))


@app.command(name="alias")
def set_alias(
    model: str = typer.Argument(help="Model name"),
    alias: str = typer.Argument(help="Alias to assign"),
):
    """
    Assign an alias to an existing model.

    Examples:
      hfl alias huihui-qwen3-coder-30b-a3b-instruct-abliterated-i1-gguf-q4_k_m coder
      hfl run coder
    """
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]Model not found:[/] {model}")
        raise typer.Exit(1)

    # Verify that the alias is not in use
    existing = registry.get(alias)
    if existing and existing.name != manifest.name:
        console.print(f"[red]The alias '{alias}' is already in use by:[/] {existing.name}")
        raise typer.Exit(1)

    if registry.set_alias(manifest.name, alias):
        console.print(f"[green]Alias assigned:[/] {alias} -> {manifest.name}")
        console.print(f"[dim]You can now use:[/] hfl run {alias}")
    else:
        console.print(f"[red]Error assigning alias[/]")
        raise typer.Exit(1)


@app.command()
def login(
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        help="HuggingFace token (if not provided, will be requested interactively)",
    ),
):
    """
    Configure your HuggingFace token for faster downloads.

    Get your token at: https://huggingface.co/settings/tokens

    The token is saved securely using huggingface_hub.
    This allows:
      - Faster downloads (higher rate limits)
      - Access to gated models (after accepting their license on HF)
    """
    from huggingface_hub import login as hf_login
    from huggingface_hub import whoami

    try:
        if token:
            hf_login(token=token, add_to_git_credential=False)
        else:
            console.print("[bold]Configure HuggingFace token[/]\n")
            console.print("Get your token at: [cyan]https://huggingface.co/settings/tokens[/]\n")
            hf_login(add_to_git_credential=False)

        # Verify it works
        user_info = whoami()
        console.print(f"\n[green]Authenticated as:[/] {user_info['name']}")
        console.print("[dim]Token saved. Downloads will now be faster.[/]")
    except Exception as e:
        console.print(f"[red]Error authenticating:[/] {e}")
        raise typer.Exit(1)


@app.command()
def logout():
    """Remove the saved HuggingFace token."""
    from huggingface_hub import logout as hf_logout

    try:
        hf_logout()
        console.print("[green]Token removed successfully.[/]")
    except Exception as e:
        console.print(f"[yellow]Warning:[/] {e}")


@app.command()
def version():
    """Show the version of hfl."""
    from hfl import __version__

    console.print(f"hfl v{__version__} — Licensed under HRUL v1.0")
    console.print("[dim]https://github.com/ggalancs/hfl[/]")


if __name__ == "__main__":
    app()

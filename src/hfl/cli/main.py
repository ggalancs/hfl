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

Language:
  Set HFL_LANG environment variable to change language (en, es).
  Default: en (English)
"""

import sys

import typer
from rich.console import Console
from rich.panel import Panel

from hfl.i18n import t

app = typer.Typer(
    name="hfl",
    help=t("app.description"),
    no_args_is_help=True,
)
console = Console()


@app.command()
def pull(
    model: str = typer.Argument(help=t("commands.pull.args.model")),
    quantize: str = typer.Option(
        "Q4_K_M", "--quantize", "-q", help=t("commands.pull.options.quantize")
    ),
    format: str = typer.Option(
        "auto",
        "--format",
        "-f",
        help=t("commands.pull.options.format"),
    ),
    alias: str = typer.Option(
        None,
        "--alias",
        "-a",
        help=t("commands.pull.options.alias"),
    ),
    skip_license: bool = typer.Option(
        False, "--skip-license", help=t("commands.pull.options.skip_license")
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
    console.print(f"[bold]{t('messages.resolving')}[/] {model}...")
    try:
        resolved = resolve(model, quantization=quantize)
    except (ValueError, Exception) as e:
        error_msg = str(e)
        if "Repo id must" in error_msg or "repo_name" in error_msg:
            console.print(f"[red]{t('errors.format_error')}[/]")
            console.print(f"\n[yellow]{t('errors.supported_formats')}[/]")
            console.print(f"  - {t('errors.format_org_model')}")
            console.print(f"  - {t('errors.format_org_model_quant')}")
            console.print(f"  - {t('errors.format_model_name')}")
            console.print(f"\n[dim]{t('errors.input_received')}:[/] {model}")
            console.print(f"[dim]{t('errors.detail')}:[/] {e}")
        elif "not found" in error_msg.lower() or "No se encontró" in error_msg:
            console.print(f"[red]Error:[/] {e}")
            console.print(f"[dim]{t('errors.check_name_or_search')}[/]")
        else:
            console.print(f"[red]{t('errors.error_resolving')}:[/] {e}")
        raise typer.Exit(1)

    console.print(f"  {t('messages.repo')}: {resolved.repo_id}")
    console.print(f"  {t('messages.format')}: {resolved.format}")
    if resolved.filename:
        console.print(f"  {t('messages.file')}: {resolved.filename}")

    # 2. Verify license (R1 - Legal Audit)
    license_info = None
    license_accepted_at = None
    if not skip_license:
        try:
            license_info = check_model_license(resolved.repo_id)
            if not require_user_acceptance(license_info, resolved.repo_id):
                console.print(f"[yellow]{t('warnings.download_cancelled')}[/]")
                raise typer.Exit(0)
            license_accepted_at = datetime.now().isoformat()
        except Exception as e:
            console.print(f"[yellow]{t('warnings.could_not_verify_license')}:[/] {e}")
            if not typer.confirm(t("warnings.continue_without_license"), default=False):
                raise typer.Exit(0)

    # 3. Download
    local_path = pull_model(resolved)
    console.print(f"[green]{t('messages.downloaded_to')}:[/] {local_path}")

    # 3. Convert if necessary
    fmt = detect_format(local_path)
    final_path = local_path

    if fmt != ModelFormat.GGUF and format != "safetensors":
        # Check if the model can be converted to GGUF
        from hfl.converter.gguf_converter import GGUFConverter, check_model_convertibility

        is_convertible, reason = check_model_convertibility(local_path)

        if not is_convertible:
            console.print(f"\n[yellow]{t('errors.cannot_convert_gguf')}:[/] {reason}")
            console.print(f"\n[dim]{t('errors.model_downloaded_but')}[/]")
            console.print(f"[dim]{t('errors.consider_searching_gguf')}[/]")
            console.print(f"  hfl search {resolved.repo_id.split('/')[-1]} --gguf\n")
            raise typer.Exit(1)

        console.print(f"[yellow]{t('messages.converting_to_gguf', quantize=quantize)}[/]")
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
    ready_msg = f"{t('messages.model_ready')}: {manifest.name} ({manifest.display_size})"
    if alias:
        console.print(f"\n[bold green]{ready_msg}[/]")
        console.print(f"[cyan]{t('messages.alias_label')}:[/] {alias}")
        console.print(f"[dim]{t('messages.use_command')}:[/] hfl run {alias}")
    else:
        console.print(f"\n[bold green]{ready_msg}[/]")


@app.command()
def run(
    model: str = typer.Argument(help=t("commands.run.args.model")),
    backend: str = typer.Option("auto", "--backend", "-b", help=t("commands.run.options.backend")),
    ctx: int = typer.Option(4096, "--ctx", "-c", help=t("commands.run.options.ctx")),
    system: str = typer.Option(None, "--system", "-s", help=t("commands.run.options.system")),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help=t("commands.run.options.verbose"),
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
        console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
        console.print(t("errors.use_list_to_see"))
        raise typer.Exit(1)

    console.print(f"[cyan]{t('messages.loading')}[/] {manifest.name}...")
    try:
        engine = select_engine(Path(manifest.local_path), backend=backend)
        engine.load(manifest.local_path, n_ctx=ctx, verbose=verbose)
    except MissingDependencyError as e:
        console.print(f"[red]{t('errors.missing_dependency')}:[/]\n\n{e}")
        raise typer.Exit(1)
    console.print(f"[green]{t('messages.model_loaded')}[/]\n")

    # R9 - Legal disclaimer before starting chat
    console.print(f"[dim]{t('legal.ai_disclaimer')}[/]\n")

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
    console.print(f"\n[dim]{t('messages.session_ended')}[/]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help=t("commands.serve.options.host")),
    port: int = typer.Option(11434, "--port", "-p", help=t("commands.serve.options.port")),
    model: str = typer.Option(None, "--model", "-m", help=t("commands.serve.options.model")),
    api_key: str = typer.Option(None, "--api-key", help=t("commands.serve.options.api_key")),
):
    """Start the API server (OpenAI + Ollama compatible)."""
    from hfl.api.server import start_server, state

    # R6 - Privacy warning when exposing to the network
    if host == "0.0.0.0":
        console.print(f"[yellow]Warning:[/] {t('warnings.network_exposure')}")
        if api_key:
            console.print(f"[green]{t('messages.api_key_enabled')}[/]")
        else:
            console.print(f"[yellow]{t('warnings.no_api_key')}[/]")
        if not typer.confirm(t("warnings.continue_question"), default=True):
            raise typer.Exit(0)

    if model:
        from pathlib import Path

        from hfl.engine.selector import MissingDependencyError, select_engine
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        manifest = registry.get(model)
        if manifest:
            console.print(f"[cyan]{t('messages.pre_loading')}[/] {manifest.name}...")
            try:
                state.engine = select_engine(Path(manifest.local_path))
                state.engine.load(manifest.local_path)
                state.current_model = manifest
            except MissingDependencyError as e:
                console.print(f"[red]{t('errors.missing_dependency')}:[/]\n\n{e}")
                raise typer.Exit(1)

    console.print(f"[bold green]{t('messages.server_at', host=host, port=port)}[/]")
    console.print("  OpenAI: POST /v1/chat/completions")
    console.print("  Ollama: POST /api/chat")
    if api_key:
        console.print(f"  [cyan]{t('messages.auth_required')}[/]")
    start_server(host=host, port=port, api_key=api_key)


@app.command(name="list")
def list_models():
    """List all downloaded models."""
    from rich.table import Table

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    models = registry.list_all()

    if not models:
        console.print(f"[dim]{t('table.no_models')}[/]")
        return

    table = Table(title=t("table.local_models"))
    table.add_column(t("table.name"), style="cyan")
    table.add_column(t("table.alias"), style="green")
    table.add_column(t("table.format"))
    table.add_column(t("table.quantization"))
    table.add_column(t("table.license"))
    table.add_column(t("table.size"), justify="right")

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
        return t("inspect.na")
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
    query: str = typer.Argument(help=t("commands.search.args.query")),
    limit: int = typer.Option(100, "--limit", "-l", help=t("commands.search.options.limit")),
    page_size: int = typer.Option(
        10, "--page-size", "-n", help=t("commands.search.options.page_size")
    ),
    gguf_only: bool = typer.Option(
        False, "--gguf", "-g", help=t("commands.search.options.gguf_only")
    ),
    max_params: float = typer.Option(
        None,
        "--max-params",
        "-p",
        help=t("commands.search.options.max_params"),
    ),
    min_params: float = typer.Option(
        None, "--min-params", help=t("commands.search.options.min_params")
    ),
    sort: str = typer.Option(
        "downloads",
        "--sort",
        "-s",
        help=t("commands.search.options.sort"),
    ),
):
    """Search models on HuggingFace Hub with interactive pagination."""
    from huggingface_hub import HfApi

    # Validate minimum length
    if len(query.strip()) < 3:
        console.print(f"[red]Error:[/] {t('errors.search_min_chars')}")
        raise typer.Exit(1)

    console.print(f"[bold]{t('messages.searching', query=query)}[/]\n")

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
        console.print(f"[red]{t('errors.error_searching')}:[/] {e}")
        raise typer.Exit(1)

    if not models:
        console.print(f"[yellow]{t('errors.no_models_found', query=query)}[/]")
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
            console.print(f"[yellow]{t('errors.no_gguf_models_found', query=query)}[/]")
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
            filter_str = " and ".join(filter_desc)
            msg = t("errors.no_models_params_found", filter=filter_str, query=query)
            console.print(f"[yellow]{msg}[/]")
            return

    total = len(models)
    total_pages = (total + page_size - 1) // page_size
    current_page = 0

    # Show header
    console.print(
        Panel(
            f"[bold]{t('messages.models_found', count=total)}[/]  |  "
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
        page_msg = t(
            "messages.page_info",
            current=current_page + 1,
            total=total_pages,
            start=start_idx + 1,
            end=end_idx,
            count=total,
        )
        page_info = f"[dim]-- {page_msg} --[/]"

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
                    f"\n[dim]{t('messages.search_finished', shown=end_idx, total=total)}[/]"
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
            console.print(f"{page_info}  [bold green]{t('messages.end_of_results')}[/]")
            current_page += 1

    # Show help at the end
    console.print()
    console.print(f"[dim]{t('messages.to_download')}[/]")


@app.command()
def rm(
    model: str = typer.Argument(help=t("commands.rm.args.model")),
):
    """Delete a local model."""
    import shutil
    from pathlib import Path

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
        raise typer.Exit(1)

    # Confirm
    confirm = typer.confirm(
        t("confirm.delete_model", name=manifest.name, size=manifest.display_size),
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
    console.print(f"[green]{t('messages.deleted')}:[/] {manifest.name}")


@app.command()
def inspect(model: str = typer.Argument(help=t("commands.inspect.args.model"))):
    """Show detailed information about a model."""
    from rich.panel import Panel
    from rich.text import Text

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
        raise typer.Exit(1)

    info = Text()
    info.append(f"{t('inspect.name')}:          {manifest.name}\n")
    if manifest.alias:
        info.append(f"{t('inspect.alias')}:         {manifest.alias}\n")
    info.append(f"{t('inspect.hf_repo')}:       {manifest.repo_id}\n")
    info.append(f"{t('inspect.local_path')}:    {manifest.local_path}\n")
    info.append(f"{t('inspect.format')}:        {manifest.format}\n")
    info.append(f"{t('inspect.quantization')}:  {manifest.quantization or t('inspect.na')}\n")
    info.append(
        f"{t('inspect.architecture')}:  {manifest.architecture or t('inspect.auto_detect')}\n"
    )
    info.append(f"{t('inspect.parameters')}:    {manifest.parameters or t('inspect.unknown')}\n")
    info.append(f"{t('inspect.context')}:       {manifest.context_length} {t('inspect.tokens')}\n")
    info.append(f"{t('inspect.size')}:          {manifest.display_size}\n")
    info.append(f"{t('inspect.downloaded')}:    {manifest.created_at}\n")

    # R1 - Show license information
    info.append(f"\n[{t('inspect.license_section')}]\n")
    info.append(f"{t('inspect.license')}:       {manifest.license or t('inspect.unknown')}\n")
    if manifest.license_url:
        info.append(f"{t('inspect.url')}:           {manifest.license_url}\n")
    if manifest.gated:
        info.append(f"{t('inspect.gated')}:         {t('inspect.gated_yes')}\n")
    if manifest.license_restrictions:
        info.append(f"{t('inspect.restrictions')}:\n")
        for r in manifest.license_restrictions:
            info.append(f"  - {r}\n")
    if manifest.license_accepted_at:
        info.append(f"{t('inspect.accepted')}:      {manifest.license_accepted_at[:10]}\n")

    console.print(Panel(info, title=f"[bold]{manifest.name}[/]", border_style="cyan"))


@app.command(name="alias")
def set_alias(
    model: str = typer.Argument(help=t("commands.alias.args.model")),
    alias: str = typer.Argument(help=t("commands.alias.args.alias")),
):
    """Assign an alias to an existing model."""
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
        raise typer.Exit(1)

    # Verify that the alias is not in use
    existing = registry.get(alias)
    if existing and existing.name != manifest.name:
        console.print(f"[red]{t('errors.alias_in_use', alias=alias, model=existing.name)}[/]")
        raise typer.Exit(1)

    if registry.set_alias(manifest.name, alias):
        console.print(f"[green]{t('messages.alias_assigned')}:[/] {alias} -> {manifest.name}")
        console.print(f"[dim]{t('messages.you_can_now_use')}:[/] hfl run {alias}")
    else:
        console.print(f"[red]{t('errors.error_assigning_alias')}[/]")
        raise typer.Exit(1)


@app.command()
def login(
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        help=t("commands.login.options.token"),
    ),
):
    """Configure your HuggingFace token for faster downloads."""
    from huggingface_hub import login as hf_login
    from huggingface_hub import whoami

    try:
        if token:
            hf_login(token=token, add_to_git_credential=False)
        else:
            console.print(f"[bold]{t('messages.configure_hf_token')}[/]\n")
            console.print(
                f"{t('messages.get_token_at')}: [cyan]https://huggingface.co/settings/tokens[/]\n"
            )
            hf_login(add_to_git_credential=False)

        # Verify it works
        user_info = whoami()
        console.print(f"\n[green]{t('messages.authenticated_as')}:[/] {user_info['name']}")
        console.print(f"[dim]{t('messages.token_saved')}[/]")
    except Exception as e:
        console.print(f"[red]{t('errors.error_authenticating')}:[/] {e}")
        raise typer.Exit(1)


@app.command()
def logout():
    """Remove the saved HuggingFace token."""
    from huggingface_hub import logout as hf_logout

    try:
        hf_logout()
        console.print(f"[green]{t('messages.token_removed')}[/]")
    except Exception as e:
        console.print(f"[yellow]Warning:[/] {e}")


@app.command()
def version():
    """Show the hfl version."""
    from hfl import __version__

    console.print(f"hfl v{__version__} — Licensed under HRUL v1.0")
    console.print("[dim]https://github.com/ggalancs/hfl[/]")


if __name__ == "__main__":
    app()

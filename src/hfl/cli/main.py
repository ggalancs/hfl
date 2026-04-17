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

import json
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel

from hfl.cli.commands._utils import (
    console,
    display_model_row,
    get_key,
    get_model_type,
    get_params_value,
    progress_spinner,
)
from hfl.i18n import t

app = typer.Typer(
    name="hfl",
    help=t("app.description"),
    no_args_is_help=True,
)


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
    alias: str | None = typer.Option(
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

    # Detect model type from pipeline_tag (before download)
    from hfl.converter.formats import (
        ModelType,
        get_model_type_display_name,
        is_model_type_supported,
        model_type_from_pipeline_tag,
    )

    resolved_model_type = model_type_from_pipeline_tag(resolved.pipeline_tag)

    console.print(f"  {t('messages.repo')}: {resolved.repo_id}")
    console.print(f"  {t('messages.format')}: {resolved.format}")
    if resolved.filename:
        console.print(f"  {t('messages.file')}: {resolved.filename}")
    if resolved_model_type:
        type_name = get_model_type_display_name(resolved_model_type)
        console.print(f"  {t('messages.type')}: {type_name}")

        # Check if model type is supported
        if not is_model_type_supported(resolved_model_type):
            console.print(f"\n[red]{t('errors.unsupported_model_type')}[/]")
            console.print(f"[dim]{t('errors.unsupported_model_type_hint', type=type_name)}[/]")
            raise typer.Exit(1)

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

    # 4. Detect model type and convert if necessary
    fmt = detect_format(local_path)
    final_path = local_path

    # Use pipeline_tag from resolver if available, fallback to local detection
    from hfl.converter.formats import detect_model_type

    if resolved_model_type:
        detected_type = resolved_model_type
    else:
        detected_type = detect_model_type(local_path)

        # Check if model type is supported (only when detected locally after download)
        if detected_type != ModelType.LLM and not is_model_type_supported(detected_type):
            type_name = get_model_type_display_name(detected_type)
            console.print(f"\n[red]{t('errors.unsupported_model_type')}:[/] {type_name}")
            console.print(f"[dim]{t('errors.unsupported_model_type_hint', type=type_name)}[/]")
            # Clean up downloaded files
            import shutil

            if local_path.exists():
                shutil.rmtree(local_path) if local_path.is_dir() else local_path.unlink()
            raise typer.Exit(1)

    # Only attempt GGUF conversion for LLM models
    if fmt != ModelFormat.GGUF and format != "safetensors":
        # Non-LLM models (TTS, STT, etc.) don't need GGUF conversion
        if detected_type != ModelType.LLM:
            console.print(f"[dim]{t('messages.no_conversion_needed')}[/]")
            # Keep as safetensors - no conversion needed
        else:
            # LLM model - attempt GGUF conversion
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
        model_type=detected_type.value,
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
    ctx: int = typer.Option(0, "--ctx", "-c", help=t("commands.run.options.ctx")),
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

    from hfl.converter.formats import (
        ModelType,
        get_model_type_display_name,
        is_model_type_supported,
    )
    from hfl.engine.base import ChatMessage
    from hfl.engine.selector import MissingDependencyError, select_engine
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)
    if not manifest:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
        console.print(t("errors.use_list_to_see"))
        raise typer.Exit(1)

    # Check if model type is supported for chat
    model_type = get_model_type(manifest)
    if model_type != ModelType.LLM:
        type_name = get_model_type_display_name(model_type)
        console.print(f"[red]{t('errors.wrong_model_type')}:[/] {model}")
        console.print(f"  {t('errors.detected_type')}: [yellow]{type_name}[/]")
        console.print(f"  {t('errors.expected_type')}: [green]LLM (Text Generation)[/]")

        if not is_model_type_supported(model_type):
            console.print(f"\n[dim]{t('errors.unsupported_type_hint')}[/]")
            raise typer.Exit(1)

        # Model type is supported but not LLM (e.g., TTS)
        if model_type == ModelType.TTS:
            console.print(f"\n[dim]{t('errors.use_tts_command')}[/]")
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
        try:
            for token in engine.chat_stream(messages):
                console.print(token, end="", highlight=False, markup=False, style=green_style)
                full_response.append(token)
        except KeyboardInterrupt:
            pass  # Stop streaming gracefully on Ctrl+C
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
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    ),
    json_logs: bool = typer.Option(False, "--json-logs", help="Output logs in JSON format"),
    ctx: int = typer.Option(
        0,
        "--ctx",
        "-c",
        help="Context size override (0 = use model default or 4096)",
    ),
    tray: bool = typer.Option(
        False,
        "--tray",
        "--gui",
        help="Show system tray icon for server management",
    ),
):
    """Start the API server (OpenAI + Ollama + Anthropic compatible)."""
    from hfl.api.server import start_server
    from hfl.api.state import get_state
    from hfl.core.observability_setup import setup_event_listeners
    from hfl.logging_config import configure_logging

    # Initialize structured logging
    configure_logging(level=log_level, json_format=json_logs)

    # Connect events to metrics
    setup_event_listeners()

    # Tray mode: launch system tray icon with server control
    if tray:
        try:
            from hfl.tray.icon import run_tray

            run_tray(
                host=host,
                port=port,
                api_key=api_key,
                model=model,
                log_level=log_level,
                json_logs=json_logs,
                auto_start=True,
            )
            return
        except ImportError:
            console.print(
                "[red]Error:[/] Tray mode requires pystray and Pillow.\n"
                "Install with: [cyan]pip install hfl[tray][/]"
            )
            raise typer.Exit(1)

    # R6 - Privacy warning when exposing to the network
    if host == "0.0.0.0":
        console.print(f"[yellow]Warning:[/] {t('warnings.network_exposure')}")
        if api_key:
            console.print(f"[green]{t('messages.api_key_enabled')}[/]")
        else:
            console.print(f"[yellow]{t('warnings.no_api_key')}[/]")
        if not typer.confirm(t("warnings.continue_question"), default=True):
            raise typer.Exit(0)

    # Store context size override in state for lazy-load path
    state = get_state()
    if ctx > 0:
        state.context_size_override = ctx

    if model:
        from pathlib import Path

        from hfl.engine.selector import MissingDependencyError, select_engine
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        manifest = registry.get(model)
        if manifest:
            console.print(f"[cyan]{t('messages.pre_loading')}[/] {manifest.name}...")
            try:
                n_ctx = ctx if ctx > 0 else 0  # 0 = auto-detect from model
                state.engine = select_engine(Path(manifest.local_path))
                state.engine.load(manifest.local_path, n_ctx=n_ctx)
                state.current_model = manifest
            except MissingDependencyError as e:
                console.print(f"[red]{t('errors.missing_dependency')}:[/]\n\n{e}")
                raise typer.Exit(1)

    console.print(f"[bold green]{t('messages.server_at', host=host, port=port)}[/]")
    console.print("  OpenAI:    POST /v1/chat/completions")
    console.print("  Anthropic: POST /v1/messages")
    console.print("  Ollama:    POST /api/chat")
    if api_key:
        console.print(f"  [cyan]{t('messages.auth_required')}[/]")
    start_server(host=host, port=port, api_key=api_key)


@app.command(name="list")
def list_models(
    supported_only: bool = typer.Option(
        False,
        "--supported-only",
        "-s",
        help=t("commands.list.options.supported_only"),
    ),
):
    """List all downloaded models."""
    from rich.table import Table

    from hfl.converter.formats import is_model_type_supported
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    models = registry.list_all()

    if not models:
        console.print(f"[dim]{t('table.no_models')}[/]")
        return

    # Filter unsupported models if requested
    if supported_only:
        filtered_models = []
        for m in models:
            model_type = get_model_type(m)
            if is_model_type_supported(model_type):
                filtered_models.append(m)
        models = filtered_models

        if not models:
            console.print(f"[dim]{t('table.no_supported_models')}[/]")
            return

    table = Table(title=t("table.local_models"))
    table.add_column(t("table.name"), style="cyan")
    table.add_column(t("table.alias"), style="green")
    table.add_column(t("table.type"))
    table.add_column(t("table.format"))
    table.add_column(t("table.quantization"))
    table.add_column(t("table.license"))
    table.add_column(t("table.size"), justify="right")

    for m in models:
        # Get model type
        model_type = get_model_type(m)
        is_supported = is_model_type_supported(model_type)

        # Format type display with color
        if is_supported:
            type_str = f"[green]{model_type.value.upper()}[/]"
        else:
            type_str = f"[red]{t('table.unsupported')}[/]"

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
            type_str,
            m.format,
            m.quantization or "-",
            license_str,
            m.display_size,
        )

    console.print(table)

    # Show tip about unsupported models if any
    if not supported_only:
        unsupported_count = sum(1 for m in models if not is_model_type_supported(get_model_type(m)))
        if unsupported_count > 0:
            console.print(f"\n[dim]{t('messages.unsupported_tip', count=unsupported_count)}[/]")


def _pull_selected_model(model) -> None:
    """Pull a model selected from search results."""
    model_id = model.id

    # Check if model has GGUF files
    has_gguf = False
    siblings = getattr(model, "siblings", None)
    if siblings:
        has_gguf = any(s.rfilename.endswith(".gguf") for s in siblings)

    # Show selection
    console.print(f"\n[bold cyan]{t('messages.selected_model')}:[/] {model_id}")

    # Confirm download
    if not typer.confirm(t("confirm.pull_model"), default=True):
        console.print(f"[dim]{t('warnings.download_cancelled')}[/]")
        return

    # Build pull arguments
    quantize = "Q4_K_M"
    if has_gguf:
        # If model has GGUF, format will be auto-detected
        console.print(f"[dim]{t('messages.gguf_detected')}[/]")

    # Execute pull command
    console.print()

    # Call pull function directly
    try:
        pull(
            model=model_id,
            quantize=quantize,
            format="auto",
            alias=None,
            skip_license=False,
        )
    except SystemExit:
        pass  # pull raises Exit on completion


@app.command(name="cp")
def cp(
    source: str = typer.Argument(help="Existing model name"),
    destination: str = typer.Argument(help="New name to create"),
) -> None:
    """Copy a model to a new name (Ollama-compatible).

    Creates a new registry entry pointing at the same blob as the
    source, so the operation is nearly free. ``hfl cp`` mirrors
    ``ollama cp`` byte-for-byte.
    """
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    if registry.get(source) is None:
        console.print(f"[red]Source model not found:[/] {source}")
        raise typer.Exit(1)
    if registry.get(destination) is not None:
        console.print(f"[red]Destination already exists:[/] {destination}")
        raise typer.Exit(1)

    try:
        ok = registry.copy(source, destination)
    except Exception as exc:
        console.print(f"[red]Copy failed:[/] {exc}")
        raise typer.Exit(1)

    if ok:
        console.print(f"[green]Copied[/] [cyan]{source}[/] → [cyan]{destination}[/]")
    else:
        console.print(f"[red]Copy failed[/] (concurrent write?): {source} → {destination}")
        raise typer.Exit(1)


@app.command(name="stop")
def stop(
    model: str = typer.Argument(None, help="Model name to unload. Omit to unload all."),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="HFL server host"),
    port: int = typer.Option(11434, "--port", "-p", help="HFL server port"),
) -> None:
    """Unload a model without restarting the server (Ollama-compatible).

    Sends ``POST /api/stop`` to the running HFL server. If a model
    name is given, only that one is evicted; otherwise every loaded
    model (LLM + TTS) is released.
    """
    import httpx

    url = f"http://{host}:{port}/api/stop"
    body: dict = {}
    if model:
        body["model"] = model

    try:
        response = httpx.post(url, json=body, timeout=10.0)
        response.raise_for_status()
    except httpx.ConnectError:
        console.print(
            f"[red]Cannot reach HFL server at {url}[/]\n"
            f"[dim]Start it first with:[/] [cyan]hfl serve[/]"
        )
        raise typer.Exit(1)
    except httpx.HTTPError as exc:
        console.print(f"[red]Server error:[/] {exc}")
        raise typer.Exit(1)

    data = response.json()
    status = data.get("status")
    target = data.get("model") or "(all)"
    if status == "stopped":
        console.print(f"[green]Stopped[/] [cyan]{target}[/]")
    elif status == "not_loaded":
        console.print(f"[yellow]{target}[/] is not currently loaded.")
    elif status == "nothing_loaded":
        console.print("[dim]No models were loaded; nothing to stop.[/]")
    else:
        console.print(f"[dim]Unexpected response:[/] {data}")


@app.command(name="show")
def show(
    model: str = typer.Argument(help="Model name, alias or repo_id to inspect"),
    modelfile: bool = typer.Option(
        False, "--modelfile", help="Print only the rendered Modelfile body"
    ),
    parameters: bool = typer.Option(
        False, "--parameters", help="Print only the PARAMETER block (one per line)"
    ),
    template: bool = typer.Option(False, "--template", help="Print only the chat template"),
    license_only: bool = typer.Option(False, "--license", help="Print only the license text"),
) -> None:
    """Show model information (Ollama-compatible).

    Mirrors ``ollama show`` — by default prints a summary with details,
    capabilities, and the license; flags narrow the output to a single
    section for scripting.
    """
    from hfl.converter.modelfile import render_modelfile
    from hfl.models.capabilities import detect_capabilities
    from hfl.models.registry import ModelRegistry

    manifest = ModelRegistry().get(model)
    if manifest is None:
        console.print(f"[red]Model not found:[/] {model}")
        raise typer.Exit(1)

    # Single-section flags first (mirror ollama show --modelfile).
    if modelfile:
        console.print(render_modelfile(manifest), end="")
        return
    if parameters:
        from hfl.api.routes_show import _format_parameters

        console.print(_format_parameters(manifest))
        return
    if template:
        console.print(manifest.chat_template or "")
        return
    if license_only:
        console.print(manifest.license_name or manifest.license or "")
        return

    # Default: summary table — same columns ollama show prints.
    from rich.panel import Panel
    from rich.table import Table

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right")
    summary.add_column()
    summary.add_row("Name", manifest.name)
    summary.add_row("Architecture", manifest.architecture or "unknown")
    summary.add_row("Parameters", manifest.parameters or "?")
    summary.add_row("Quantization", manifest.quantization or "?")
    summary.add_row("Format", manifest.format or "?")
    if manifest.context_length:
        summary.add_row("Context", f"{manifest.context_length} tokens")
    summary.add_row("Size", manifest.display_size)
    summary.add_row(
        "Capabilities",
        ", ".join(detect_capabilities(manifest)) or "—",
    )
    summary.add_row("License", manifest.license or "—")

    console.print(Panel(summary, title=f"Model: {manifest.name}", expand=False))


@app.command(name="ps")
def ps(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-H", help="Host where the HFL server is running"
    ),
    port: int = typer.Option(11434, "--port", "-p", help="Port where the HFL server is running"),
) -> None:
    """List models currently loaded in memory (Ollama-compatible).

    Hits the server's ``/api/ps`` endpoint and renders a table with
    NAME, ID, SIZE, PROCESSOR and UNTIL — matching the output layout
    of ``ollama ps`` so scripts written against Ollama work unchanged.
    """
    import httpx
    from rich.table import Table

    url = f"http://{host}:{port}/api/ps"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
    except httpx.ConnectError:
        console.print(
            f"[red]Cannot reach HFL server at {url}[/]\n"
            f"[dim]Start it first with:[/] [cyan]hfl serve[/]"
        )
        raise typer.Exit(1)
    except httpx.HTTPError as exc:
        console.print(f"[red]Server error:[/] {exc}")
        raise typer.Exit(1)

    data = response.json()
    models = data.get("models", [])

    if not models:
        console.print("[dim]No models loaded. Send a request to /api/chat to load one.[/]")
        return

    table = Table(title="Running models")
    table.add_column("NAME", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("SIZE", justify="right")
    table.add_column("PROCESSOR")
    table.add_column("UNTIL")

    for m in models:
        size_gb = (m.get("size") or 0) / (1024**3)
        size_vram = m.get("size_vram") or 0
        # Classify processor: any VRAM usage counts as GPU-resident;
        # pure CPU engines report 0.
        processor = "GPU" if size_vram > 0 and size_vram != (m.get("size") or 0) else "CPU"
        if size_vram and size_vram == (m.get("size") or 0):
            # Engine reported size_vram but we couldn't distinguish
            # from weights size — assume GPU (conservative; llama.cpp
            # with n_gpu_layers=-1 on Metal puts everything on GPU).
            processor = "GPU"
        expires_at = m.get("expires_at") or "—"
        digest = (m.get("digest") or "")[:12]
        table.add_row(
            m.get("name", "?"),
            digest,
            f"{size_gb:.1f} GB" if size_gb >= 0.1 else f"{(m.get('size') or 0) / (1024**2):.0f} MB",
            processor,
            expires_at,
        )

    console.print(table)


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

    api = HfApi()

    try:
        # Search models with progress spinner
        with progress_spinner(t("messages.searching", query=query)):
            kwargs: dict = {
                "search": query,
                "sort": sort,
                "limit": limit,
                "fetch_config": False,
                "full": True,  # To get siblings and detect GGUF
            }
            # direction parameter was removed in newer huggingface_hub versions
            try:
                models = list(
                    api.list_models(**kwargs, direction=-1)  # type: ignore[call-arg,unused-ignore]
                )
            except TypeError:
                models = list(api.list_models(**kwargs))
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
            params = get_params_value(m.id)
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
            f"[dim]0-9[/] {t('messages.select_to_pull')}  |  "
            f"[dim]SPACE[/] {t('messages.next_page')}  |  "
            f"[dim]q[/] {t('messages.quit')}",
            title=f"[bold cyan]Search: {query}[/]",
            border_style="cyan",
        )
    )
    console.print()

    while current_page < total_pages:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total)
        page_models = models[start_idx:end_idx]

        # Show models of the current page (0-9 index per page)
        for i, model in enumerate(page_models):
            display_model_row(model, i)

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
                key = get_key()
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
            elif key.isdigit():
                # User selected a model by number (0-9)
                selection = int(key)
                if selection < len(page_models):
                    selected_model = page_models[selection]
                    console.print()
                    _pull_selected_model(selected_model)
                    return
                else:
                    console.print()  # Invalid selection, continue
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


@app.command(name="create")
def create(
    model: str = typer.Argument(help="Name for the new model"),
    modelfile: Path = typer.Option(
        ...,
        "--file",
        "-f",
        help="Path to the Modelfile",
        exists=True,
        readable=True,
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="HFL server host"),
    port: int = typer.Option(11434, "--port", "-p", help="HFL server port"),
) -> None:
    """Create a new model from a Modelfile (Ollama-compatible).

    Mirrors ``ollama create`` — sends the Modelfile body to
    ``POST /api/create`` on a running HFL server and prints the NDJSON
    progress events as they stream in.
    """
    import httpx

    body = modelfile.read_text()
    url = f"http://{host}:{port}/api/create"
    payload: dict[str, Any] = {
        "model": model,
        "modelfile": body,
        "stream": True,
    }

    try:
        with httpx.stream("POST", url, json=payload, timeout=120.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    console.print(f"[dim]{line}[/]")
                    continue
                if "error" in event:
                    console.print(f"[red]Error:[/] {event['error']}")
                    raise typer.Exit(1)
                status = event.get("status", "")
                console.print(f"[cyan]{status}[/]")
    except httpx.ConnectError:
        console.print(
            f"[red]Cannot reach HFL server at {url}[/]\n"
            f"[dim]Start it first with:[/] [cyan]hfl serve[/]"
        )
        raise typer.Exit(1)
    except httpx.HTTPError as exc:
        console.print(f"[red]Server error:[/] {exc}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show the hfl version."""
    from hfl import __version__

    console.print(f"hfl v{__version__} — Licensed under HRUL v1.0")
    console.print("[dim]https://github.com/ggalancs/hfl[/]")


@app.command()
def config():
    """Show current configuration."""
    from rich.panel import Panel
    from rich.text import Text

    from hfl.config import config as cfg

    info = Text()
    info.append("[bold]Directories[/]\n")
    info.append(f"  Home:     {cfg.home_dir}\n")
    info.append(f"  Models:   {cfg.models_dir}\n")
    info.append(f"  Cache:    {cfg.cache_dir}\n")
    info.append(f"  Registry: {cfg.registry_path}\n")

    info.append("\n[bold]Server[/]\n")
    info.append(f"  Host: {cfg.host}\n")
    info.append(f"  Port: {cfg.port}\n")

    info.append("\n[bold]Rate Limiting[/]\n")
    info.append(f"  Enabled:  {cfg.rate_limit_enabled}\n")
    info.append(f"  Requests: {cfg.rate_limit_requests}/min\n")

    info.append("\n[bold]Inference Defaults[/]\n")
    info.append(f"  Context Size: {cfg.default_ctx_size}\n")
    info.append(f"  GPU Layers:   {cfg.default_n_gpu_layers} (-1 = all)\n")
    info.append(f"  Threads:      {cfg.default_threads} (0 = auto)\n")

    info.append("\n[bold]Timeouts (seconds)[/]\n")
    info.append(f"  Model Load:   {cfg.model_load_timeout}\n")
    info.append(f"  Generation:   {cfg.generation_timeout}\n")
    info.append(f"  API Request:  {cfg.api_request_timeout}\n")

    info.append("\n[bold]SLO Targets[/]\n")
    info.append(f"  Availability: {cfg.slo.availability_target * 100:.1f}%\n")
    info.append(f"  Latency P50:  {cfg.slo.latency_p50_ms}ms\n")
    info.append(f"  Latency P95:  {cfg.slo.latency_p95_ms}ms\n")
    info.append(f"  Latency P99:  {cfg.slo.latency_p99_ms}ms\n")
    info.append(f"  Error Rate:   {cfg.slo.error_rate_target * 100:.1f}%\n")

    info.append("\n[bold]HuggingFace[/]\n")
    if cfg.hf_token:
        info.append("  Token: [green]Configured[/]\n")
    else:
        info.append("  Token: [dim]Not set[/] (use HF_TOKEN env var)\n")

    console.print(Panel(info, title="[bold]HFL Configuration[/]", border_style="cyan"))


@app.command()
def check():
    """Run diagnostic checks (dependencies, backends, GPU)."""
    from hfl.engine.dependency_check import check_engine_availability

    console.print("[bold]Running HFL Diagnostics[/]\n")

    # Check dependencies
    console.print("[bold cyan]Backend Availability[/]")
    availability = check_engine_availability()

    for backend in ["llama-cpp", "transformers", "vllm"]:
        status = availability.get(backend, "unknown")
        if status is True:
            console.print(f"  [green]✓[/] {backend}")
        else:
            console.print(f"  [red]✗[/] {backend}: {status}")

    # GPU check
    console.print("\n[bold cyan]GPU Support[/]")
    if availability.get("torch") is True:
        if availability.get("torch_cuda"):
            device = availability.get("cuda_device", "unknown")
            console.print(f"  [green]✓[/] CUDA: {device}")
        elif availability.get("torch_mps"):
            console.print("  [green]✓[/] MPS (Apple Silicon)")
        else:
            console.print("  [yellow]○[/] CPU only")
    else:
        console.print("  [red]✗[/] PyTorch not installed")

    # TTS check
    console.print("\n[bold cyan]TTS Support[/]")
    if availability.get("transformers") is True:
        console.print("  [green]✓[/] Bark (via transformers)")
    else:
        console.print("  [red]✗[/] Bark: requires transformers")

    if availability.get("soundfile"):
        console.print("  [green]✓[/] soundfile")
    else:
        console.print("  [dim]○[/] soundfile: not installed")

    if availability.get("torchaudio"):
        console.print("  [green]✓[/] torchaudio")
    else:
        console.print("  [dim]○[/] torchaudio: not installed")

    # Registry check
    console.print("\n[bold cyan]Storage[/]")
    from hfl.models.registry import ModelRegistry

    try:
        registry = ModelRegistry()
        models = registry.list_all()
        console.print(f"  [green]✓[/] Registry: {len(models)} models")
    except Exception as e:
        console.print(f"  [red]✗[/] Registry: {e}")

    from hfl.config import config as cfg

    if cfg.models_dir.exists():
        console.print(f"  [green]✓[/] Models dir: {cfg.models_dir}")
    else:
        console.print("  [yellow]○[/] Models dir: not created")

    console.print("\n[green]Diagnostics complete.[/]")


@app.command()
def debug():
    """Show debug information for troubleshooting."""
    import platform
    import sys

    from rich.panel import Panel
    from rich.text import Text

    from hfl import __version__
    from hfl.config import config as cfg
    from hfl.engine.dependency_check import check_engine_availability

    info = Text()

    # System info
    info.append("[bold]System[/]\n")
    info.append(f"  Python:   {sys.version.split()[0]}\n")
    info.append(f"  Platform: {platform.system()} {platform.release()}\n")
    info.append(f"  Machine:  {platform.machine()}\n")

    # HFL info
    info.append("\n[bold]HFL[/]\n")
    info.append(f"  Version:  {__version__}\n")
    info.append(f"  Home:     {cfg.home_dir}\n")

    # Dependency versions
    info.append("\n[bold]Dependencies[/]\n")

    def get_version(module_name: str) -> str:
        try:
            import importlib.metadata

            return importlib.metadata.version(module_name)
        except Exception:
            return "not installed"

    info.append(f"  typer:            {get_version('typer')}\n")
    info.append(f"  rich:             {get_version('rich')}\n")
    info.append(f"  huggingface-hub:  {get_version('huggingface-hub')}\n")
    info.append(f"  fastapi:          {get_version('fastapi')}\n")
    info.append(f"  uvicorn:          {get_version('uvicorn')}\n")
    info.append(f"  pydantic:         {get_version('pydantic')}\n")

    # Optional deps
    info.append("\n[bold]Optional Dependencies[/]\n")
    availability = check_engine_availability()

    for dep in ["llama-cpp-python", "transformers", "torch", "vllm", "soundfile", "torchaudio"]:
        version = get_version(dep)
        if version != "not installed":
            info.append(f"  {dep}: {version}\n")
        else:
            info.append(f"  {dep}: [dim]not installed[/]\n")

    # GPU info
    info.append("\n[bold]GPU[/]\n")
    if availability.get("torch") is True:
        if availability.get("torch_cuda"):
            device = availability.get("cuda_device", "unknown")
            info.append(f"  CUDA: {device}\n")
            try:
                import torch

                info.append(f"  CUDA Version: {torch.version.cuda}\n")
                info.append(f"  cuDNN: {torch.backends.cudnn.version()}\n")
            except Exception:
                pass
        elif availability.get("torch_mps"):
            info.append("  MPS: Apple Silicon\n")
        else:
            info.append("  GPU: [dim]none available[/]\n")
    else:
        info.append("  GPU: [dim]torch not installed[/]\n")

    # Memory info
    try:
        import psutil

        mem = psutil.virtual_memory()
        info.append("\n[bold]Memory[/]\n")
        info.append(f"  Total:     {mem.total / 1024**3:.1f} GB\n")
        info.append(f"  Available: {mem.available / 1024**3:.1f} GB\n")
        info.append(f"  Used:      {mem.percent}%\n")
    except ImportError:
        pass

    console.print(Panel(info, title="[bold]HFL Debug Info[/]", border_style="yellow"))
    console.print("\n[dim]For support: https://github.com/ggalancs/hfl/issues[/]")


@app.command("compliance-report")
def compliance_report(
    output: Path = typer.Option(Path("compliance_report.json"), help="Output file path"),
    format: str = typer.Option("json", help="Output format: json or markdown"),
):
    """Generate compliance report for all downloaded models."""
    import json
    from datetime import datetime
    from pathlib import Path as PathClass

    from hfl.models.registry import get_registry

    registry = get_registry()
    models = registry.list_all()

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "hfl_version": "0.1.0",
        "total_models": len(models),
        "models": [],
    }

    for model in models:
        entry = {
            "name": model.name,
            "repo_id": model.repo_id,
            "license": getattr(model, "license", "unknown"),
            "local_path": model.local_path,
            "created_at": model.created_at,
        }
        # Add alias info if available
        if hasattr(model, "alias") and model.alias:
            entry["alias"] = model.alias
        report["models"].append(entry)

    output_path = PathClass(str(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        output_path.write_text(json.dumps(report, indent=2, default=str))
    elif format == "markdown":
        lines = [
            "# HFL Compliance Report",
            "",
            f"Generated: {report['generated_at']}",
            f"Total models: {report['total_models']}",
            "",
        ]
        for m in report["models"]:
            lines.append(f"## {m['name']}")
            lines.append(f"- Repository: {m['repo_id']}")
            lines.append(f"- License: {m.get('license', 'unknown')}")
            lines.append(f"- Path: {m['local_path']}")
            lines.append("")
        output_path.write_text("\n".join(lines))

    console.print(f"[green]Report saved to {output_path}[/green]")


@app.command(name="help")
def help_command(
    extras: bool = typer.Option(
        False,
        "--extras",
        help=t("help.options.extras"),
    ),
):
    """Show help information and available options."""
    import importlib

    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if extras:
        # Show detailed extras information
        console.print(
            Panel(
                t("help.extras_intro"),
                title=f"[bold cyan]{t('help.extras_title')}[/]",
                border_style="cyan",
            )
        )
        console.print()

        extra_order = [
            "llama",
            "transformers",
            "vllm",
            "convert",
            "tts",
            "coqui",
            "audio",
            "tray",
            "all",
        ]

        table = Table(show_header=True, border_style="dim")
        table.add_column(t("table.name"), style="cyan", min_width=14)
        table.add_column("", min_width=40)
        table.add_column("Status", justify="center", min_width=14)
        table.add_column("pip install", style="dim", min_width=22)

        for extra_name in extra_order:
            info = t(f"help.extras.{extra_name}.summary")
            install_cmd = t(f"help.extras.{extra_name}.install")
            check_module = t(f"help.extras.{extra_name}.check_module")

            # Check if installed
            if check_module and check_module != "null":
                try:
                    importlib.import_module(check_module)
                    status = f"[green]{t('help.extras_installed')}[/]"
                except ImportError:
                    status = f"[dim]{t('help.extras_not_installed')}[/]"
            else:
                status = "[dim]—[/]"

            table.add_row(extra_name, info, status, install_cmd)

        console.print(table)

        # Show multi-install syntax (escape brackets for Rich markup)
        console.print(f"\n[dim]{t('help.extras_multiple')}:[/]")
        console.print("  [cyan]pip install hfl\\[llama,tray][/]")
        console.print("  [cyan]pip install hfl\\[all][/]")
        console.print()

        # Detail each extra
        for extra_name in extra_order:
            desc = t(f"help.extras.{extra_name}.description")
            packages = t(f"help.extras.{extra_name}.packages")
            install_cmd = t(f"help.extras.{extra_name}.install")

            detail = Text()
            detail.append(f"{desc}\n\n")
            detail.append(f"Packages: {packages}\n", style="dim")
            detail.append(f"Install:  {install_cmd}", style="cyan")

            console.print(
                Panel(detail, title=f"[bold]{extra_name}[/]", border_style="dim", width=80)
            )

    else:
        # General help
        help_text = Text()
        help_text.append(f"{t('help.general_help')}\n\n", style="bold")
        help_text.append(f"{t('help.usage')}:\n", style="bold cyan")
        help_text.append("  hfl <command> [options]\n\n")

        help_text.append(f"{t('help.common_commands')}:\n", style="bold cyan")
        commands = [
            ("pull <model>", t("help.common_pull")),
            ("run <model>", t("help.common_run")),
            ("serve [--tray]", t("help.common_serve")),
            ("search <query>", t("help.common_search")),
            ("list", t("help.common_list")),
        ]
        for cmd, desc in commands:
            help_text.append(f"  hfl {cmd:<22}", style="cyan")
            help_text.append(f" {desc}\n")

        help_text.append(f"\n{t('help.more_info')}\n\n", style="dim")
        help_text.append(t("help.extras_hint") + "\n", style="bold")

        console.print(Panel(help_text, title="[bold]hfl[/]", border_style="cyan"))


if __name__ == "__main__":
    app()

# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
CLI principal de hfl.

Uso:
  hfl pull <modelo> [--quantize Q4_K_M]
  hfl run <modelo> [--backend auto]
  hfl serve [--port 11434]
  hfl list
  hfl search <texto>
  hfl rm <modelo>
  hfl inspect <modelo>
"""

import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="hfl",
    help="Ejecuta modelos de HuggingFace localmente.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def pull(
    model: str = typer.Argument(help="Modelo HF (ej: meta-llama/Llama-3.3-70B-Instruct)"),
    quantize: str = typer.Option("Q4_K_M", "--quantize", "-q", help="Nivel de cuantización"),
    format: str = typer.Option("auto", "--format", "-f", help="Formato: auto, gguf, safetensors"),
    alias: str = typer.Option(
        None, "--alias", "-a", help="Alias corto para referirse al modelo (ej: 'coder')"
    ),
    skip_license: bool = typer.Option(
        False, "--skip-license", help="Saltar verificación de licencia"
    ),
):
    """Descarga un modelo desde HuggingFace Hub."""
    from datetime import datetime

    from hfl.converter.formats import ModelFormat, detect_format
    from hfl.converter.gguf_converter import GGUFConverter
    from hfl.hub.downloader import pull_model
    from hfl.hub.license_checker import check_model_license, require_user_acceptance
    from hfl.hub.resolver import resolve
    from hfl.models.manifest import ModelManifest
    from hfl.models.registry import ModelRegistry

    # 1. Resolver modelo
    console.print(f"[bold]Resolviendo[/] {model}...")
    try:
        resolved = resolve(model, quantization=quantize)
    except (ValueError, Exception) as e:
        error_msg = str(e)
        if "Repo id must" in error_msg or "repo_name" in error_msg:
            console.print(f"[red]Error de formato:[/] El nombre del modelo es inválido.")
            console.print(f"\n[yellow]Formatos soportados:[/]")
            console.print("  • org/modelo                  → repo directo de HuggingFace")
            console.print("  • org/modelo:Q4_K_M           → repo con cuantización")
            console.print("  • nombre-modelo               → búsqueda por nombre")
            console.print(f"\n[dim]Entrada recibida:[/] {model}")
            console.print(f"[dim]Detalle:[/] {e}")
        elif "No se encontró modelo" in error_msg:
            console.print(f"[red]Error:[/] {e}")
            console.print("[dim]Verifica el nombre o usa 'hfl search' para buscar.[/]")
        else:
            console.print(f"[red]Error al resolver modelo:[/] {e}")
        raise typer.Exit(1)

    console.print(f"  Repo: {resolved.repo_id}")
    console.print(f"  Formato: {resolved.format}")
    if resolved.filename:
        console.print(f"  Archivo: {resolved.filename}")

    # 2. Verificar licencia (R1 - Auditoría Legal)
    license_info = None
    license_accepted_at = None
    if not skip_license:
        try:
            license_info = check_model_license(resolved.repo_id)
            if not require_user_acceptance(license_info, resolved.repo_id):
                console.print("[yellow]Descarga cancelada.[/]")
                raise typer.Exit(0)
            license_accepted_at = datetime.now().isoformat()
        except Exception as e:
            console.print(f"[yellow]Advertencia:[/] No se pudo verificar licencia: {e}")
            if not typer.confirm("¿Continuar sin verificar licencia?", default=False):
                raise typer.Exit(0)

    # 3. Descargar
    local_path = pull_model(resolved)
    console.print(f"[green]Descargado en:[/] {local_path}")

    # 3. Convertir si es necesario
    fmt = detect_format(local_path)
    final_path = local_path

    if fmt != ModelFormat.GGUF and format != "safetensors":
        console.print(f"[yellow]Convirtiendo a GGUF ({quantize})...[/]")
        converter = GGUFConverter()
        output_name = resolved.repo_id.replace("/", "--")
        output_path = local_path.parent / output_name
        final_path = converter.convert(local_path, output_path, quantize)

    # 4. Registrar
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
        # R1 - Información de licencia
        license=license_info.license_id if license_info else None,
        license_name=license_info.license_name if license_info else None,
        license_url=license_info.url if license_info else None,
        license_restrictions=license_info.restrictions if license_info else [],
        gated=license_info.gated if license_info else False,
        license_accepted_at=license_accepted_at,
    )

    registry = ModelRegistry()
    registry.add(manifest)

    # Mostrar resultado con alias si se definió
    if alias:
        console.print(f"\n[bold green]Modelo listo:[/] {manifest.name} ({manifest.display_size})")
        console.print(f"[cyan]Alias:[/] {alias}")
        console.print(f"[dim]Usa:[/] hfl run {alias}")
    else:
        console.print(f"\n[bold green]Modelo listo:[/] {manifest.name} ({manifest.display_size})")


@app.command()
def run(
    model: str = typer.Argument(help="Nombre del modelo local"),
    backend: str = typer.Option("auto", "--backend", "-b"),
    ctx: int = typer.Option(4096, "--ctx", "-c", help="Tamaño de contexto"),
    system: str = typer.Option(None, "--system", "-s", help="Prompt de sistema"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Mostrar logs del backend (Metal, CUDA)"
    ),
):
    """Inicia un chat interactivo con un modelo."""
    from pathlib import Path

    from hfl.engine.base import ChatMessage
    from hfl.engine.selector import MissingDependencyError, select_engine
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)
    if not manifest:
        console.print(f"[red]Modelo no encontrado:[/] {model}")
        console.print("Usa 'hfl list' para ver modelos disponibles.")
        raise typer.Exit(1)

    console.print(f"[cyan]Cargando[/] {manifest.name}...")
    try:
        engine = select_engine(Path(manifest.local_path), backend=backend)
        engine.load(manifest.local_path, n_ctx=ctx, verbose=verbose)
    except MissingDependencyError as e:
        console.print(f"[red]Dependencia faltante:[/]\n\n{e}")
        raise typer.Exit(1)
    console.print(f"[green]Modelo cargado.[/] Escribe '/exit' para salir.\n")

    # R9 - Disclaimer legal antes de iniciar chat
    console.print(
        "[dim]Los modelos AI pueden generar información incorrecta, "
        "sesgada o inapropiada. El usuario es responsable del uso "
        "que haga de las respuestas generadas.[/]\n"
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

        # Streaming de respuesta con estilo
        # markup=False evita que Rich interprete [] como tags de formato
        from rich.style import Style

        green_style = Style(color="green")

        full_response = []
        for token in engine.chat_stream(messages):
            console.print(token, end="", highlight=False, markup=False, style=green_style)
            full_response.append(token)
        console.print()  # Nueva línea al final

        messages.append(ChatMessage(role="assistant", content="".join(full_response)))

    engine.unload()
    console.print("\n[dim]Sesión terminada.[/]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(11434, "--port", "-p"),
    model: str = typer.Option(None, "--model", "-m", help="Pre-cargar modelo"),
):
    """Inicia el servidor API (compatible OpenAI + Ollama)."""
    from hfl.api.server import start_server, state

    # R6 - Advertencia de privacidad al exponer a la red
    if host == "0.0.0.0":
        console.print(
            "[yellow]Advertencia:[/] Exponiendo el servidor a la red. "
            "Los prompts enviados a la API serán accesibles desde "
            "cualquier dispositivo en la red."
        )
        if not typer.confirm("¿Continuar?", default=True):
            raise typer.Exit(0)

    if model:
        from pathlib import Path

        from hfl.engine.selector import MissingDependencyError, select_engine
        from hfl.models.registry import ModelRegistry

        registry = ModelRegistry()
        manifest = registry.get(model)
        if manifest:
            console.print(f"[cyan]Pre-cargando[/] {manifest.name}...")
            try:
                state.engine = select_engine(Path(manifest.local_path))
                state.engine.load(manifest.local_path)
                state.current_model = manifest
            except MissingDependencyError as e:
                console.print(f"[red]Dependencia faltante:[/]\n\n{e}")
                raise typer.Exit(1)

    console.print(f"[bold green]hfl server[/] en http://{host}:{port}")
    console.print("  OpenAI: POST /v1/chat/completions")
    console.print("  Ollama: POST /api/chat")
    start_server(host=host, port=port)


@app.command(name="list")
def list_models():
    """Lista todos los modelos descargados."""
    from rich.table import Table

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    models = registry.list_all()

    if not models:
        console.print("[dim]No hay modelos descargados. Usa 'hfl pull' para descargar.[/]")
        return

    table = Table(title="Modelos Locales")
    table.add_column("Nombre", style="cyan")
    table.add_column("Alias", style="green")
    table.add_column("Formato")
    table.add_column("Cuantización")
    table.add_column("Licencia")
    table.add_column("Tamaño", justify="right")

    for m in models:
        # R1 - Mostrar licencia con indicador de riesgo
        license_str = m.license or "?"
        if m.license:
            # Indicador de riesgo según tipo de licencia
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
    """Formatea tamaño en bytes a formato legible."""
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
    """Lee una tecla sin necesidad de presionar Enter."""
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
    """Extrae el número de parámetros del nombre del modelo (ej: '70B', '7B')."""
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
    """Estima el tamaño del modelo basado en parámetros y cuantización."""
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
    """Muestra una fila de modelo formateada."""
    # Obtener información del modelo
    model_id = model.id
    downloads = getattr(model, "downloads", 0) or 0
    likes = getattr(model, "likes", 0) or 0

    # Detectar si tiene GGUF
    has_gguf = False
    siblings = getattr(model, "siblings", None)
    if siblings:
        has_gguf = any(s.rfilename.endswith(".gguf") for s in siblings)

    # Formatear tags
    tags = getattr(model, "tags", []) or []
    pipeline_tag = getattr(model, "pipeline_tag", None)

    # Icono de formato
    format_icon = "[green]●[/] GGUF" if has_gguf else "[dim]○[/] HF"

    # Formatear descargas
    if downloads >= 1_000_000:
        dl_str = f"{downloads / 1_000_000:.1f}M"
    elif downloads >= 1_000:
        dl_str = f"{downloads / 1_000:.1f}K"
    else:
        dl_str = str(downloads)

    # Extraer parámetros y estimar tamaño
    params = _extract_params_from_name(model_id)
    size_q4 = _estimate_model_size(params, "Q4")
    size_str = f"[magenta]~{size_q4}[/]" if params else ""

    # Número de índice
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
    """Extrae el valor numérico de parámetros del nombre (ej: 70 para '70B')."""
    params = _extract_params_from_name(model_id)
    if not params:
        return None
    try:
        return float(params.replace("B", "").replace("b", ""))
    except ValueError:
        return None


@app.command()
def search(
    query: str = typer.Argument(help="Texto a buscar (mínimo 3 caracteres)"),
    limit: int = typer.Option(100, "--limit", "-l", help="Número máximo de resultados"),
    page_size: int = typer.Option(10, "--page-size", "-n", help="Resultados por página"),
    gguf_only: bool = typer.Option(False, "--gguf", "-g", help="Mostrar solo modelos con GGUF"),
    max_params: float = typer.Option(
        None, "--max-params", "-p", help="Máximo de parámetros en B (ej: 70 para <70B)"
    ),
    min_params: float = typer.Option(
        None, "--min-params", help="Mínimo de parámetros en B (ej: 7 para >7B)"
    ),
    sort: str = typer.Option(
        "downloads", "--sort", "-s", help="Ordenar por: downloads, likes, created"
    ),
):
    """
    Busca modelos en HuggingFace Hub con paginación interactiva.

    Controles:
      ESPACIO/ENTER  Siguiente página
      q/Q/ESC        Salir
      p              Página anterior (si está disponible)

    Ejemplos:
      hfl search llama
      hfl search mistral --gguf
      hfl search phi --limit 50 --page-size 5
      hfl search qwen --max-params 14        # Modelos de menos de 14B
      hfl search llama --min-params 70       # Modelos de 70B o más
    """
    from huggingface_hub import HfApi

    # Validar longitud mínima
    if len(query.strip()) < 3:
        console.print("[red]Error:[/] La búsqueda debe tener al menos 3 caracteres.")
        raise typer.Exit(1)

    console.print(f"[bold]Buscando[/] '{query}' en HuggingFace Hub...\n")

    api = HfApi()

    try:
        # Buscar modelos
        models = list(
            api.list_models(
                search=query,
                sort=sort,
                direction=-1,
                limit=limit,
                fetch_config=False,
                full=True,  # Para obtener siblings y detectar GGUF
            )
        )
    except Exception as e:
        console.print(f"[red]Error al buscar:[/] {e}")
        raise typer.Exit(1)

    if not models:
        console.print(f"[yellow]No se encontraron modelos para:[/] '{query}'")
        return

    # Filtrar por GGUF si se solicita
    if gguf_only:
        models = [
            m
            for m in models
            if hasattr(m, "siblings")
            and m.siblings
            and any(s.rfilename.endswith(".gguf") for s in m.siblings)
        ]
        if not models:
            console.print(f"[yellow]No se encontraron modelos GGUF para:[/] '{query}'")
            return

    # Filtrar por número de parámetros
    if max_params is not None or min_params is not None:
        filtered = []
        for m in models:
            params = _get_params_value(m.id)
            if params is None:
                continue  # Excluir modelos sin parámetros detectables
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
                f"[yellow]No se encontraron modelos {' y '.join(filter_desc)} para:[/] '{query}'"
            )
            return

    total = len(models)
    total_pages = (total + page_size - 1) // page_size
    current_page = 0

    # Mostrar encabezado
    console.print(
        Panel(
            f"[bold]{total}[/] modelos encontrados  |  "
            f"[dim]ESPACIO/ENTER[/] continuar  |  "
            f"[dim]q[/] salir  |  "
            f"[dim]p[/] anterior",
            title=f"[bold cyan]Búsqueda: {query}[/]",
            border_style="cyan",
        )
    )
    console.print()

    while current_page < total_pages:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total)
        page_models = models[start_idx:end_idx]

        # Mostrar modelos de la página actual
        for i, model in enumerate(page_models):
            _display_model_row(model, start_idx + i + 1)

        # Mostrar estado de paginación
        console.print()
        page_info = f"[dim]── Página {current_page + 1}/{total_pages} ({start_idx + 1}-{end_idx} de {total}) ──[/]"

        if current_page < total_pages - 1:
            console.print(f"{page_info}  [dim]ESPACIO[/] más  [dim]q[/] salir", end="")

            # Esperar input del usuario
            try:
                key = _get_key()
            except Exception:
                # Fallback si no hay terminal interactivo
                try:
                    user_input = input("\n[Presiona ENTER para continuar, 'q' para salir]: ")
                    key = "q" if user_input.lower() == "q" else " "
                except (EOFError, KeyboardInterrupt):
                    key = "q"

            # Limpiar línea de estado
            console.print("\r" + " " * 80 + "\r", end="")

            if key in ("q", "Q", "\x1b", "\x03"):  # q, Q, ESC, Ctrl+C
                console.print(
                    f"\n[dim]Búsqueda terminada. Mostrando {end_idx} de {total} resultados.[/]"
                )
                break
            elif key == "p" and current_page > 0:
                current_page -= 1
                console.print()  # Nueva línea antes de la página anterior
            else:
                current_page += 1
                console.print()  # Nueva línea antes de la siguiente página
        else:
            # Última página
            console.print(f"{page_info}  [bold green]Fin de resultados[/]")
            current_page += 1

    # Mostrar ayuda al final
    console.print()
    console.print("[dim]Para descargar un modelo usa:[/] hfl pull <modelo>")


@app.command()
def rm(model: str = typer.Argument(help="Nombre del modelo a eliminar")):
    """Elimina un modelo local."""
    import shutil
    from pathlib import Path

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]Modelo no encontrado:[/] {model}")
        raise typer.Exit(1)

    # Confirmar
    confirm = typer.confirm(
        f"¿Eliminar {manifest.name} ({manifest.display_size})?",
    )
    if not confirm:
        return

    # Borrar archivos
    path = Path(manifest.local_path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()

    registry.remove(model)
    console.print(f"[green]Eliminado:[/] {manifest.name}")


@app.command()
def inspect(model: str = typer.Argument(help="Nombre del modelo")):
    """Muestra información detallada de un modelo."""
    from rich.panel import Panel
    from rich.text import Text

    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]Modelo no encontrado:[/] {model}")
        raise typer.Exit(1)

    info = Text()
    info.append(f"Nombre:        {manifest.name}\n")
    if manifest.alias:
        info.append(f"Alias:         {manifest.alias}\n")
    info.append(f"Repo HF:       {manifest.repo_id}\n")
    info.append(f"Ruta local:    {manifest.local_path}\n")
    info.append(f"Formato:       {manifest.format}\n")
    info.append(f"Cuantización:  {manifest.quantization or 'N/A'}\n")
    info.append(f"Arquitectura:  {manifest.architecture or 'auto-detect'}\n")
    info.append(f"Parámetros:    {manifest.parameters or 'desconocido'}\n")
    info.append(f"Contexto:      {manifest.context_length} tokens\n")
    info.append(f"Tamaño:        {manifest.display_size}\n")
    info.append(f"Descargado:    {manifest.created_at}\n")

    # R1 - Mostrar información de licencia
    info.append("\n[Licencia]\n")
    info.append(f"Licencia:      {manifest.license or 'desconocida'}\n")
    if manifest.license_url:
        info.append(f"URL:           {manifest.license_url}\n")
    if manifest.gated:
        info.append("Gated:         Sí (requirió aceptación en HF)\n")
    if manifest.license_restrictions:
        info.append("Restricciones:\n")
        for r in manifest.license_restrictions:
            info.append(f"  - {r}\n")
    if manifest.license_accepted_at:
        info.append(f"Aceptada:      {manifest.license_accepted_at[:10]}\n")

    console.print(Panel(info, title=f"[bold]{manifest.name}[/]", border_style="cyan"))


@app.command(name="alias")
def set_alias(
    model: str = typer.Argument(help="Nombre del modelo"),
    alias: str = typer.Argument(help="Alias a asignar"),
):
    """
    Asigna un alias a un modelo existente.

    Ejemplos:
      hfl alias huihui-qwen3-coder-30b-a3b-instruct-abliterated-i1-gguf-q4_k_m coder
      hfl run coder
    """
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]Modelo no encontrado:[/] {model}")
        raise typer.Exit(1)

    # Verificar que el alias no esté en uso
    existing = registry.get(alias)
    if existing and existing.name != manifest.name:
        console.print(f"[red]El alias '{alias}' ya está en uso por:[/] {existing.name}")
        raise typer.Exit(1)

    if registry.set_alias(manifest.name, alias):
        console.print(f"[green]Alias asignado:[/] {alias} → {manifest.name}")
        console.print(f"[dim]Ahora puedes usar:[/] hfl run {alias}")
    else:
        console.print(f"[red]Error al asignar alias[/]")
        raise typer.Exit(1)


@app.command()
def login(
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        help="Token de HuggingFace (si no se proporciona, se pedirá interactivamente)",
    ),
):
    """
    Configura tu token de HuggingFace para descargas más rápidas.

    Obtén tu token en: https://huggingface.co/settings/tokens

    El token se guarda de forma segura usando huggingface_hub.
    Esto permite:
      - Descargas más rápidas (rate limits más altos)
      - Acceso a modelos gated (tras aceptar su licencia en HF)
    """
    from huggingface_hub import login as hf_login
    from huggingface_hub import whoami

    try:
        if token:
            hf_login(token=token, add_to_git_credential=False)
        else:
            console.print("[bold]Configurar token de HuggingFace[/]\n")
            console.print("Obtén tu token en: [cyan]https://huggingface.co/settings/tokens[/]\n")
            hf_login(add_to_git_credential=False)

        # Verificar que funciona
        user_info = whoami()
        console.print(f"\n[green]✓ Autenticado como:[/] {user_info['name']}")
        console.print("[dim]Token guardado. Las descargas ahora serán más rápidas.[/]")
    except Exception as e:
        console.print(f"[red]Error al autenticar:[/] {e}")
        raise typer.Exit(1)


@app.command()
def logout():
    """Elimina el token de HuggingFace guardado."""
    from huggingface_hub import logout as hf_logout

    try:
        hf_logout()
        console.print("[green]✓ Token eliminado correctamente.[/]")
    except Exception as e:
        console.print(f"[yellow]Advertencia:[/] {e}")


@app.command()
def version():
    """Muestra la versión de hfl."""
    from hfl import __version__

    console.print(f"hfl v{__version__} — Licensed under HRUL v1.0")
    console.print("[dim]https://github.com/ggalancs/hfl[/]")


if __name__ == "__main__":
    app()

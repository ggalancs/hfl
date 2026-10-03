# SPDX-License-Identifier: Apache-2.0
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
import os
import sys
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
from hfl.utils.terminal import stdin_is_terminal

app = typer.Typer(
    name="hfl",
    help=t("app.description"),
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Back the top-level ``--version`` flag: print the version and exit."""
    if value:
        from hfl import __version__

        console.print(f"hfl v{__version__} — Licensed under Apache-2.0")
        console.print("[dim]https://github.com/ggalancs/hfl[/]")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the hfl version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    # No docstring on purpose: Typer would use it as the group help and
    # override the i18n ``help=t("app.description")`` set on the Typer() above.
    pass


def _resolve_or_exit(model: str, quantize: str, revision: str | None) -> Any:
    """``model`` resolved on the Hub; exits 1 saying why when it cannot be."""
    from hfl.hub.resolver import resolve

    console.print(f"[bold]{t('messages.resolving')}[/] {model}...")
    try:
        resolved = resolve(model, quantization=quantize, revision=revision)
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
        elif _hub_unreachable(e):
            _print_hub_unreachable()
        else:
            console.print(f"[red]{t('errors.error_resolving')}:[/] {e}")
        raise typer.Exit(1) from e
    return resolved


def _show_resolved_or_exit(resolved: Any) -> None:
    """What will be fetched; exits 1 for a model type HFL cannot serve."""
    from hfl.converter.formats import (
        get_model_type_display_name,
        is_model_type_supported,
        model_type_from_pipeline_tag,
    )

    resolved_model_type = model_type_from_pipeline_tag(resolved.pipeline_tag)

    console.print(f"  {t('messages.repo')}: {resolved.repo_id}")
    console.print(f"  {t('messages.format')}: {resolved.format}")
    if resolved.filename:
        console.print(f"  {t('messages.file')}: {resolved.filename}")
    # Show the pinned revision + the immutable commit it resolved to, so the
    # user can see (and later reproduce) exactly what was fetched.
    if resolved.revision and resolved.revision != "main":
        console.print(f"  {t('messages.revision')}: {resolved.revision}")
    if resolved.commit_sha:
        console.print(f"  {t('messages.commit')}: {resolved.commit_sha[:12]}")
    if resolved_model_type:
        type_name = get_model_type_display_name(resolved_model_type)
        console.print(f"  {t('messages.type')}: {type_name}")

        # Check if model type is supported
        if not is_model_type_supported(resolved_model_type):
            console.print(f"\n[red]{t('errors.unsupported_model_type')}[/]")
            console.print(f"[dim]{t('errors.unsupported_model_type_hint', type=type_name)}[/]")
            raise typer.Exit(1)


def _license_or_exit(
    repo_id: str, skip_license: bool, revision: str | None = None
) -> tuple[Any, str | None]:
    """The model's license, accepted by the user (and when); exits 0 when
    they decline."""
    from datetime import datetime

    from hfl.hub.license_checker import check_model_license, require_user_acceptance

    license_info = None
    license_accepted_at = None
    if not skip_license:
        try:
            # Read at the commit that is downloaded, not at a moving "main".
            license_info = check_model_license(repo_id, revision=revision)
            if not require_user_acceptance(license_info, repo_id):
                console.print(f"[yellow]{t('warnings.download_cancelled')}[/]")
                raise typer.Exit(0)
            license_accepted_at = datetime.now().isoformat()
        except typer.Exit:
            # The user declined: typer.Exit is a RuntimeError, and the handler
            # below took it for a failed check and offered to download anyway.
            raise
        except Exception as e:
            console.print(f"[yellow]{t('warnings.could_not_verify_license')}:[/] {e}")
            if not typer.confirm(t("warnings.continue_without_license"), default=False):
                raise typer.Exit(0) from e
    return license_info, license_accepted_at


def _causes(exc: BaseException) -> list[BaseException]:
    """``exc`` and what caused it, outermost first."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _disk_full(exc: BaseException) -> bool:
    """A full disk anywhere in the chain: ``OSError`` ENOSPC, or hf_xet's
    RuntimeError, which carries it only as text ("No space left on device
    (os error 28)")."""
    import errno

    for cause in _causes(exc):
        if isinstance(cause, OSError) and cause.errno == errno.ENOSPC:
            return True
        text = str(cause)
        if "No space left on device" in text or "os error 28" in text:
            return True
    return False


def _hub_lost(exc: BaseException) -> bool:
    """The Hub unreachable: a transport error anywhere in the chain, or
    huggingface_hub's LocalEntryNotFoundError (what it raises when the
    network is gone and the file is not cached)."""
    import httpx
    from huggingface_hub.errors import LocalEntryNotFoundError

    return any(
        isinstance(cause, (httpx.TransportError, LocalEntryNotFoundError, ConnectionError))
        for cause in _causes(exc)
    )


def _download_or_exit(resolved: Any) -> Any:
    """The download's local path; exits 1 saying why when it fails."""
    from huggingface_hub.utils import GatedRepoError

    from hfl.exceptions import DownloadIntegrityError
    from hfl.hub.downloader import pull_model
    from hfl.utils.retry import RetryExhausted

    try:
        local_path = pull_model(resolved)
    except GatedRepoError as e:
        # Gated repo: HF access control is separate from our local license
        # acceptance — the user must request access on the Hub and provide a
        # token. Show a clean, actionable message instead of a traceback.
        console.print(f"\n[red]{t('errors.gated_model')}[/]")
        console.print(f"[yellow]{t('errors.gated_model_hint', repo=resolved.repo_id)}[/]")
        raise typer.Exit(1) from e
    except RetryExhausted as e:
        # Network/transport failure that exhausted retries — surface the real
        # underlying cause (RetryExhausted carries it), not just the wrapper.
        console.print(f"\n[red]{t('errors.download_failed')}:[/] {e.last_exception or e}")
        raise typer.Exit(1) from e
    except DownloadIntegrityError as e:
        console.print(f"\n[red]{t('errors.download_failed')}:[/] {e.details}")
        raise typer.Exit(1) from e
    except Exception as e:
        # A full disk or a lost Hub in the middle of a download ended in a
        # traceback (audit G2, G3): say what happened and what to do.
        if _disk_full(e):
            from hfl.config import config as hfl_config

            folder = str(hfl_config.home_dir)
            console.print(f"\n[red]{t('errors.disk_full_during_download', folder=folder)}[/]")
            raise typer.Exit(1) from e
        if _hub_lost(e):
            console.print(f"\n[red]{t('errors.hub_lost_during_download')}[/]")
            raise typer.Exit(1) from e
        raise
    console.print(f"[green]{t('messages.downloaded_to')}:[/] {local_path}")
    return local_path


def _print_ready(manifest: Any, detected_type: Any) -> None:
    """The model is ready: its name, and how to use it by its alias."""
    from hfl.converter.formats import ModelType

    alias = manifest.alias
    ready_msg = f"{t('messages.model_ready')}: {manifest.name} ({manifest.display_size})"
    if alias:
        console.print(f"\n[bold green]{ready_msg}[/]")
        console.print(f"[cyan]{t('messages.alias_label')}:[/] {alias}")
        # What the model is for: chat, embeddings over the API, or speech.
        if detected_type == ModelType.EMBEDDING:
            hint = t("messages.use_embed", name=alias)
        elif detected_type == ModelType.TTS:
            hint = f'hfl tts {alias} "..."'
        else:
            hint = f"hfl run {alias}"
        console.print(f"[dim]{t('messages.use_command')}:[/] {hint}")
    else:
        console.print(f"\n[bold green]{ready_msg}[/]")


def _print_pull_step_error(e: Any, repo_id: str) -> None:
    """A pull step that could not go on, said as ``hfl pull`` always said it."""
    if e.key == "errors.unsupported_model_type":
        type_name = e.values.get("type", "")
        console.print(f"\n[red]{t('errors.unsupported_model_type')}:[/] {type_name}")
        console.print(f"[dim]{t('errors.unsupported_model_type_hint', type=type_name)}[/]")
    elif e.key == "errors.cannot_convert_gguf":
        console.print(f"\n[yellow]{t('errors.cannot_convert_gguf')}:[/] {e.values['reason']}")
        console.print(f"\n[dim]{t('errors.model_downloaded_but')}[/]")
        console.print(f"[dim]{t('errors.consider_searching_gguf')}[/]")
        console.print(f"  hfl search {repo_id.split('/')[-1]} --gguf\n")
    else:
        console.print(f"\n[red]{t('errors.conversion_failed')}:[/] {e.values.get('reason', '')}")
        console.print(f"[dim]{t('errors.model_downloaded_but')}[/]")


def _print_kept(finished: Any, quantize: str) -> None:
    """What was done with a download that is not a GGUF."""
    from hfl.hub.pull_service import Kept

    kept = finished.kept
    if kept == Kept.AS_IS and finished.model_type.value != "llm":
        console.print(f"[dim]{t('messages.no_conversion_needed')}[/]")
    elif kept == Kept.MLX_NATIVE:
        console.print(
            "[cyan]MLX pre-quantized model detected — serving natively with the MLX backend.[/]"
        )
    elif kept == Kept.MLX_ELSEWHERE:
        console.print(
            "[yellow]MLX pre-quantized model detected.[/] This repo is not convertible to "
            "GGUF. To serve it you need Apple Silicon with the MLX backend: "
            "`pip install 'hfl[mlx]'`."
        )
    elif kept == Kept.FOR_MLX:
        console.print(
            "[cyan]Apple Silicon + MLX available — keeping safetensors for the MLX backend.[/]"
        )
    if "template_recovered" in finished.notes:
        console.print("[dim]Recovered missing chat_template from the base repo.[/]")
    if "template_missing" in finished.notes:
        console.print(
            "[yellow]Warning:[/] this repo's tokenizer has no chat_template and none could "
            "be recovered. Chat endpoints will fail until you add ``chat_template.jinja`` "
            "manually."
        )


@app.command()
def pull(
    model: str = typer.Argument(help=t("commands.pull.args.model")),
    quantize: str | None = typer.Option(
        None, "--quantize", "-q", help=t("commands.pull.options.quantize")
    ),
    format: str = typer.Option(
        "auto",
        "--format",
        "-f",
        help=t("commands.pull.options.format"),
    ),
    revision: str | None = typer.Option(
        None,
        "--revision",
        help=t("commands.pull.options.revision"),
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
    yes: bool = typer.Option(False, "--yes", "-y", help=t("shortname.option_yes")),
):
    """Download a model from HuggingFace Hub."""

    # 0. A short name (``qwen3-coder``, ``qwen3:8b``) is chosen on the Hub
    # first and remembered as an alias. ``yes is True``: called as a plain
    # function the option holds Typer's (truthy) default, never a yes.
    from hfl.hub.resolver import parse_model_spec
    from hfl.hub.shortname import is_short_name
    from hfl.models.registry import ModelRegistry

    if parse_model_spec(model).repo_id is None and is_short_name(model):
        from hfl.hub.shortname import alias_for

        choice = _choose_short_name(model, assume_yes=yes is True)
        if choice is None:
            console.print(f"[red]{t('errors.model_not_found')}:[/] {escape_markup(model)}")
            raise typer.Exit(1)
        if not isinstance(alias, str) or not alias:
            alias = alias_for(model)
        model, quantize = choice.reference, choice.quantization

    # 1. Resolve, show what was resolved, refuse a type HFL cannot serve.
    # No -q: a GGUF repo's file is its Q4_K_M, as always; a conversion gets
    # the most precise level that fits (``_conversion_level``).
    if not isinstance(quantize, str) or not quantize:
        quantize = None
    resolved = _resolve_or_exit(model, quantize or "Q4_K_M", revision)
    _show_resolved_or_exit(resolved)
    if quantize is None:
        quantize, format = _conversion_level(resolved, format, assume_yes=yes is True)
    _disk_space_or_exit(resolved, quantize if _will_convert(resolved, format) else None)

    # 2. License (R1 - legal audit), 3. download
    license_info, license_accepted_at = _license_or_exit(
        resolved.repo_id, skip_license, revision=resolved.commit_sha or resolved.revision
    )
    local_path = _download_or_exit(resolved)

    # 4. Its type, and — an LLM that is not a GGUF — MLX or a conversion
    from hfl.hub.pull_service import PullStepError, finish_download, register_pulled

    try:
        finished = finish_download(
            resolved,
            local_path,
            requested_format=format,
            quantize=quantize,
            on_convert=lambda: console.print(
                f"[yellow]{t('messages.converting_to_gguf', quantize=quantize)}[/]"
            ),
        )
    except PullStepError as e:
        _print_pull_step_error(e, resolved.repo_id)
        raise typer.Exit(1) from e
    _print_kept(finished, quantize)
    detected_type = finished.model_type

    # 5. Register (with its provenance)
    manifest = register_pulled(
        resolved,
        finished,
        registry=ModelRegistry(),
        license_info=license_info,
        accepted_at=license_accepted_at,
        alias=alias,
        quantize=quantize,
        source="hfl pull",
    )
    _print_ready(manifest, detected_type)


def _check_chat_model(manifest: Any, model: str) -> None:
    """Exit with the reason when ``manifest`` is not a text-generation model."""
    from hfl.converter.formats import (
        ModelType,
        get_model_type_display_name,
        is_model_type_supported,
    )

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


def _load_for_chat(manifest: Any, backend: str, ctx: int, verbose: bool) -> Any:
    """The engine for ``manifest``, loaded; exits when a dependency is missing."""
    from pathlib import Path

    from hfl.engine.selector import MissingDependencyError, select_engine

    console.print(f"[cyan]{t('messages.loading')}[/] {manifest.name}...")
    try:
        from hfl.api.model_loader import load_kwargs_for

        engine = select_engine(Path(manifest.local_path), backend=backend)
        engine.load(manifest.local_path, **load_kwargs_for(manifest, ctx), verbose=verbose)
    except MissingDependencyError as e:
        console.print(f"[red]{t('errors.missing_dependency')}:[/]\n\n{escape_markup(str(e))}")
        raise typer.Exit(1) from e
    console.print(f"[green]{t('messages.model_loaded')}[/]\n")
    return engine


def _open_session(session: str | None, model: str, system: str | None) -> tuple[list[Any], Any]:
    """The conversation so far, and the saved session it lives in (None
    without ``--session``)."""
    from hfl.engine.base import ChatMessage

    messages: list[ChatMessage] = []
    chat_session = None
    if session:
        from hfl.core.sessions import ChatSession, load_session

        try:
            chat_session = load_session(session)
        except FileNotFoundError:
            chat_session = ChatSession(name=session, model=model, system=system)
            console.print(f"[dim]{t('messages.session_new', name=session)}[/]")
        else:
            messages = [ChatMessage(**m) for m in chat_session.messages]
            console.print(
                f"[dim]{t('messages.session_resumed', name=session, count=len(messages))}[/]"
            )
    return messages, chat_session


def _chat_loop(engine: Any, messages: list[Any], gen_config: Any, persist: Any) -> None:
    """Read a line, stream the reply, save; until /exit, EOF or Ctrl-C."""
    from hfl.engine.base import ChatMessage

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
            for token in engine.chat_stream(messages, gen_config):
                console.print(token, end="", highlight=False, markup=False, style=green_style)
                full_response.append(token)
        except KeyboardInterrupt:
            pass  # Stop streaming gracefully on Ctrl+C
        console.print()  # New line at the end

        messages.append(ChatMessage(role="assistant", content="".join(full_response)))
        persist()


@app.command()
def run(
    model: str = typer.Argument(help=t("commands.run.args.model")),
    backend: str = typer.Option("auto", "--backend", "-b", help=t("commands.run.options.backend")),
    ctx: int = typer.Option(0, "--ctx", "-c", help=t("commands.run.options.ctx")),
    system: str = typer.Option(None, "--system", "-s", help=t("commands.run.options.system")),
    session: str = typer.Option(None, "--session", help=t("commands.run.options.session")),
    yes: bool = typer.Option(False, "--yes", "-y", help=t("shortname.option_yes")),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help=t("commands.run.options.verbose"),
    ),
):
    """Start an interactive chat with a model."""

    from hfl.engine.base import ChatMessage
    from hfl.models.registry import ModelRegistry

    manifest = _local_or_pulled(model, ModelRegistry, assume_yes=yes)
    if not manifest:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
        console.print(t("errors.use_list_to_see"))
        raise typer.Exit(1)

    _check_chat_model(manifest, model)

    _memory_check_or_exit(manifest, ctx)
    engine = _load_for_chat(manifest, backend, ctx, verbose)

    # R9 - Legal disclaimer before starting chat
    console.print(f"[dim]{t('legal.ai_disclaimer')}[/]\n")

    messages, chat_session = _open_session(session, model, system)

    # A created model's Modelfile: SYSTEM unless --system, MESSAGE exemplars
    # to open a new conversation, PARAMETER for every reply.
    from hfl.api.modelfile_defaults import (
        apply_parameters,
        baked_messages,
        default_system,
        splice_baked,
    )
    from hfl.engine.base import GenerationConfig

    system_prompt = default_system(manifest, system)
    if system_prompt and not any(m.role == "system" for m in messages):
        messages.append(ChatMessage(role="system", content=system_prompt))
    if all(m.role == "system" for m in messages):
        messages[:] = splice_baked(messages, baked_messages(manifest))
    gen_config = GenerationConfig()
    apply_parameters(manifest, gen_config, ())

    def _persist() -> None:
        """Write after every exchange, not once at exit.

        The feature exists to survive a restart, and the restarts worth
        surviving are the ones nobody planned — a crash, a Ctrl-C, an OOM
        kill. Saving only on a clean exit would lose exactly the sessions
        the user wanted back. A chat session is a few KB of JSON, so the
        write costs nothing next to a token.
        """
        if chat_session is None:
            return
        from hfl.core.sessions import save_session

        chat_session.messages = [{"role": m.role, "content": m.content} for m in messages]
        chat_session.model = model
        chat_session.touch()
        save_session(chat_session)

    _chat_loop(engine, messages, gen_config, _persist)

    _persist()
    engine.unload()
    if chat_session is not None:
        from hfl.core.sessions import sessions_dir

        console.print(
            f"[dim]{t('messages.session_saved', path=sessions_dir() / f'{session}.json')}[/]"
        )
    console.print(f"\n[dim]{t('messages.session_ended')}[/]")


@app.command(
    name="launch",
    help=t("commands.launch.description"),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def launch(
    ctx: typer.Context,
    tool: str = typer.Argument(..., help=t("commands.launch.args.tool")),
    model: str = typer.Option(None, "--model", "-m", help=t("commands.launch.options.model")),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help=t("commands.launch.options.host")),
    port: int | None = typer.Option(None, "--port", "-p", help=t("commands.launch.options.port")),
    # HFL_API_KEY too, as `serve` reads it: the key a running server wants.
    api_key: str = typer.Option(
        None, "--api-key", envvar="HFL_API_KEY", help=t("commands.launch.options.api_key")
    ),
    print_only: bool = typer.Option(False, "--print", help=t("commands.launch.options.print")),
    parallel: int = typer.Option(0, "--parallel", help=t("commands.launch.options.parallel")),
    yes: bool = typer.Option(False, "--yes", "-y", help=t("shortname.option_yes")),
) -> None:
    """Open Claude Code or Codex on a local model (``hfl launch claude -m NAME``)."""
    from hfl.cli.commands import launch as launcher
    from hfl.config import config
    from hfl.models.registry import ModelRegistry

    port = _configured_port(port)
    if not model:
        console.print(f"[red]{escape_markup(t('commands.launch.messages.model_required'))}[/]")
        raise typer.Exit(2)
    try:
        if print_only:
            plan = launcher.build_launch(
                tool, f"http://{host}:{port}", model, api_key=api_key, extra=list(ctx.args)
            )
            for line in launcher.shell_lines(plan):
                typer.echo(line)
            return
        launcher.check_tool(tool)
        manifest = _local_or_pulled(model, ModelRegistry, assume_yes=yes)
        if manifest is None:
            console.print(
                f"[red]{escape_markup(t('commands.launch.messages.model_missing', model=model))}[/]"
            )
            raise typer.Exit(1)
        code = launcher.run(
            tool,
            manifest.name,
            host=host,
            port=port,
            api_key=api_key,
            extra=list(ctx.args),
            log_path=config.home_dir / "logs" / "launch-server.log",
            say=lambda message: console.print(f"[dim]{escape_markup(message)}[/]"),
            parallel=parallel,
        )
    except launcher.LaunchError as exc:
        console.print(f"[red]{escape_markup(str(exc))}[/]")
        raise typer.Exit(1) from None
    raise typer.Exit(code)


def _load_tts_or_exit(model: str) -> tuple[Any, Any]:
    """Resolve ``model`` to a text-to-speech manifest and a loaded engine.

    Same resolution as ``hfl run`` (a local name, or a Hub reference pulled
    on first use), then the same memory check, then the type check: a chat
    model is pointed at ``hfl run`` instead of failing inside the engine.
    """
    from rich.markup import escape

    from hfl.converter.formats import ModelType, get_model_type_display_name
    from hfl.engine.selector import MissingDependencyError, select_tts_engine
    from hfl.models.registry import ModelRegistry

    manifest = _local_or_pulled(model, ModelRegistry, short_names=False)
    if manifest is None:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {escape(model)}")
        console.print(t("errors.use_list_to_see"))
        raise typer.Exit(1)
    model_type = get_model_type(manifest)
    if model_type != ModelType.TTS:
        console.print(f"[red]{t('errors.wrong_model_type')}:[/] {escape(model)}")
        console.print(
            f"  {t('errors.detected_type')}: "
            f"[yellow]{escape(get_model_type_display_name(model_type))}[/]"
        )
        console.print(f"  {t('errors.expected_type')}: [green]TTS (Text-to-Speech)[/]")
        if model_type == ModelType.LLM:
            console.print(f"\n[dim]{t('errors.use_run_command')}[/]")
        raise typer.Exit(1)

    _memory_check_or_exit(manifest, 0)
    console.print(f"[cyan]{t('messages.loading')}[/] {escape(manifest.name)}...")
    try:
        engine = select_tts_engine(Path(manifest.local_path))
        engine.load(manifest.local_path)
    except MissingDependencyError as e:
        console.print(f"[red]{t('errors.missing_dependency')}:[/]\n\n{escape_markup(str(e))}")
        raise typer.Exit(1) from e
    console.print(f"[green]{t('messages.tts_model_loaded')}[/]")
    return manifest, engine


def _synthesize(engine: Any, text: str, config: Any) -> Any:
    from rich.markup import escape

    preview = text[:50] + "..." if len(text) > 50 else text
    console.print(f'[cyan]{t("messages.synthesizing")}[/] "{escape(preview)}"')
    try:
        return engine.synthesize(text, config)
    finally:
        engine.unload()


def _check_speed(speed: float) -> None:
    if not 0.25 <= speed <= 4.0:
        console.print(f"[red]--speed must be between 0.25 and 4.0 (got {speed})[/]")
        raise typer.Exit(2)


@app.command()
def tts(
    model: str = typer.Argument(help=t("commands.tts.args.model")),
    text: str = typer.Argument(help=t("commands.tts.args.text")),
    output: str = typer.Option(
        "output.wav", "--output", "-o", help=t("commands.tts.options.output")
    ),
    language: str = typer.Option("en", "--lang", "-l", help=t("commands.tts.options.lang")),
    voice: str = typer.Option("default", "--voice", "-v", help=t("commands.tts.options.voice")),
    speed: float = typer.Option(1.0, "--speed", "-s", help=t("commands.tts.options.speed")),
    sample_rate: int = typer.Option(22050, "--rate", "-r", help=t("commands.tts.options.rate")),
    audio_format: str = typer.Option(
        "wav", "--format", "-f", help=t("commands.tts.options.format")
    ),
) -> None:
    """Synthesize text to an audio file."""
    from rich.markup import escape

    from hfl.engine.base import TTSConfig

    _check_speed(speed)
    if audio_format not in ("wav", "mp3", "ogg"):
        console.print(f"[red]--format must be wav, mp3 or ogg (got {escape(audio_format)})[/]")
        raise typer.Exit(2)
    _, engine = _load_tts_or_exit(model)
    config = TTSConfig(
        voice=voice, speed=speed, language=language, sample_rate=sample_rate, format=audio_format
    )
    result = _synthesize(engine, text, config)

    output_path = Path(output)
    output_path.write_bytes(result.audio)
    console.print(f"\n[bold green]{t('messages.audio_saved')}:[/] {escape(str(output_path))}")
    console.print(f"  {t('messages.duration')}: {result.duration:.2f}s")
    console.print(f"  {t('messages.sample_rate')}: {result.sample_rate} Hz")
    console.print(f"  {t('messages.format')}: {result.format}")


@app.command()
def speak(
    model: str = typer.Argument(help=t("commands.speak.args.model")),
    text: str = typer.Argument(help=t("commands.speak.args.text")),
    language: str = typer.Option("en", "--lang", "-l", help=t("commands.speak.options.lang")),
    voice: str = typer.Option("default", "--voice", "-v", help=t("commands.speak.options.voice")),
    speed: float = typer.Option(1.0, "--speed", "-s", help=t("commands.speak.options.speed")),
) -> None:
    """Synthesize text and play it directly."""
    from rich.markup import escape

    from hfl.engine.base import TTSConfig

    _check_speed(speed)
    _, engine = _load_tts_or_exit(model)
    result = _synthesize(
        engine, text, TTSConfig(voice=voice, speed=speed, language=language, format="wav")
    )
    console.print(f"[cyan]{t('messages.playing')}[/]...")
    try:
        _play_audio(result.audio, result.sample_rate)
        console.print(f"[green]{t('messages.playback_finished')}[/]")
    except Exception as e:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(result.audio)
        console.print(f"[yellow]{t('warnings.playback_failed')}:[/] {escape(str(e))}")
        console.print(f"[dim]{t('messages.audio_saved_to')}:[/] {escape(f.name)}")


def _play_audio(audio: bytes, sample_rate: int) -> None:
    """Play WAV bytes: sounddevice when installed (``hfl[audio]``), else the
    system player (afplay on macOS, aplay or paplay on Linux). Raises when
    neither is available, so the caller can save the file instead."""
    try:
        import io

        import sounddevice
        import soundfile

        data, rate = soundfile.read(io.BytesIO(audio))
        sounddevice.play(data, rate or sample_rate)
        sounddevice.wait()
        return
    except ImportError:
        pass

    import shutil
    import subprocess
    import tempfile

    player = next((p for p in ("afplay", "aplay", "paplay") if shutil.which(p)), None)
    if player is None:
        raise RuntimeError('no audio player found; install one with: pip install "hfl[audio]"')
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(audio)
        f.flush()
        subprocess.run([player, f.name], check=True, timeout=600)


def _local_or_pulled(
    model: str, registry_cls: Any, *, assume_yes: bool = False, short_names: bool = True
) -> Any:
    """The manifest ``hfl run`` should open, pulling it first if needed.

    ``model`` may be a local name or alias, or a Hub reference in the form
    the Hub's "Use this model" snippets use: ``[hf.co/]org/model[:QUANT]``.
    A reference already on disk (same repo, same quantization) is used as
    is, without touching the network; otherwise it is pulled with the same
    flow as ``hfl pull`` — license check included — and then opened. A bare
    name that is not local (``qwen3-coder``, ``qwen3:8b``) is looked up on
    the Hub and offered for confirmation (:func:`_from_short_name`).
    """
    from hfl.hub.resolver import parse_model_spec

    registry = registry_cls()
    manifest = registry.get(model)
    if manifest is not None:
        return manifest
    spec = parse_model_spec(model)
    if spec.repo_id is None:
        from hfl.hub.shortname import is_short_name

        if not short_names or not is_short_name(model):
            return None
        return _from_short_name(model, registry_cls, assume_yes=assume_yes)
    local = registry.find_pulled(spec.repo_id, spec.quantization)
    if local is not None:
        return local
    console.print(f"[cyan]{escape_markup(t('messages.run_pulling', model=model))}[/]")
    pull(
        model=model,
        quantize=spec.quantization or "Q4_K_M",
        format="auto",
        revision=spec.revision,
        alias=None,
        skip_license=False,
    )
    return registry_cls().find_pulled(spec.repo_id, spec.quantization)


def _will_convert(resolved: Any, requested_format: str) -> bool:
    """Whether this pull will convert to GGUF (as ``finish_download`` decides)."""
    from hfl.converter.formats import ModelType, model_type_from_pipeline_tag
    from hfl.engine.selector import _mlx_preferred

    model_type = model_type_from_pipeline_tag(getattr(resolved, "pipeline_tag", None))
    return (
        resolved.format == "safetensors"
        and requested_format != "safetensors"
        and model_type in (None, ModelType.LLM)
        and "mlx" not in resolved.repo_id.lower()
        and not (_mlx_preferred() and requested_format != "gguf")
    )


def _disk_space_or_exit(resolved: Any, convert_to: str | None) -> None:
    """Refuse, before downloading, a pull the disk cannot hold."""
    from hfl.config import config
    from hfl.hub.pull_service import disk_space

    space = disk_space(resolved, convert_to)
    if space is None or space.fits:
        return
    gb = 1e9
    conversion = (
        t("errors.no_disk_space_conversion", gb=space.conversion / gb) if space.conversion else ""
    )
    message = t(
        "errors.no_disk_space",
        name=resolved.repo_id,
        need=space.needed / gb,
        download=space.download / gb,
        conversion=conversion,
        folder=str(config.models_dir),
        free=space.free / gb,
    )
    console.print(f"[red]{escape_markup(message)}[/]")
    raise typer.Exit(1)


def _conversion_level(resolved: Any, requested_format: str, assume_yes: bool) -> tuple[str, str]:
    """The GGUF level (and the format) for a pull that did not name one.

    A safetensors LLM is converted to GGUF; it was always at Q4_K_M. The
    level is now the most precise one that fits this machine, shown with
    the others and asked for (Enter, ``--yes`` or no terminal: the
    recommended one), before anything is downloaded. On Apple Silicon a
    model that fits in its own precision stays for MLX, as before; one that
    does not is converted. A model that fits at no level is refused.
    """
    from hfl.converter.formats import ModelType, model_type_from_pipeline_tag
    from hfl.engine.selector import _mlx_preferred
    from hfl.hub.quant_choice import LADDER, LOWEST_RECOMMENDED, choose

    default = "Q4_K_M"
    model_type = model_type_from_pipeline_tag(getattr(resolved, "pipeline_tag", None))
    if (
        resolved.format == "gguf"
        or requested_format == "safetensors"
        or (model_type is not None and model_type != ModelType.LLM)
        or "mlx" in resolved.repo_id.lower()  # already quantized for MLX
    ):
        return default, requested_format
    from huggingface_hub import HfApi

    from hfl.hub.hw_profile import get_hw_profile
    from hfl.hub.params import estimate_params

    try:
        params = estimate_params(resolved.repo_id, api=HfApi())
    except Exception:  # offline, or the Hub would not say: as before
        params = None
    name = escape_markup(resolved.repo_id)
    if params is None or not params.total_b:
        console.print(f"[dim]{t('quant_choice.unknown_size', name=name, level=default)}[/]")
        return default, requested_format
    choice = choose(params.total_b, get_hw_profile(), active_params_b=params.active_b)
    if choice.total_gb <= 0:  # no memory reading (no psutil): as before
        console.print(f"[dim]{t('quant_choice.unknown_memory', level=default)}[/]")
        return default, requested_format
    if _mlx_preferred() and requested_format != "gguf":
        if choice.levels[0].fits:  # F16: MLX serves it in its own precision
            console.print(f"[dim]{t('quant_choice.mlx_fits')}[/]")
            return default, requested_format
        requested_format = "gguf"
    memory = f"~{choice.fast_gb:.0f} GB" + (
        f" (+ RAM: ~{choice.total_gb:.0f} GB)" if choice.total_gb > choice.fast_gb else ""
    )
    console.print(
        t("quant_choice.size", memory=memory, kind=choice.memory, name=name,
          params=f"{params.total_b:g}")
    )  # fmt: skip
    console.print(f"  [bold]{t('quant_choice.header')}[/]")
    for row in choice.levels:
        fits = (
            t("quant_choice.fits") if row.fits
            else t("quant_choice.fits_split") if row.fits_split
            else t("quant_choice.no")
        )  # fmt: skip
        best = row.name == choice.recommended
        mark = f"  [green]← {t('quant_choice.recommended')}[/]" if best else ""
        console.print(f"  {row.name:<9} {row.size_gb:>6.1f} GB  {fits}{mark}")
    if choice.recommended is None:
        refused = t(
            "quant_choice.nothing_fits",
            name=name,
            level=LOWEST_RECOMMENDED,
            need=choice.needed_gb,
            have=choice.total_gb,
        )
        console.print(f"[red]{refused}[/]")
        raise typer.Exit(1)
    if choice.split:
        console.print(f"[yellow]{t('quant_choice.split_note')}[/]")
    level = choice.recommended
    if not assume_yes and stdin_is_terminal():
        while True:
            answer = typer.prompt(t("quant_choice.pick"), default=level).strip().upper()
            if answer in LADDER:
                level = answer
                break
            console.print(t("quant_choice.bad_level", levels=", ".join(LADDER)))
    console.print(f"[cyan]{t('quant_choice.chosen', level=level)}[/]")
    return level, requested_format


def _choose_short_name(name: str, assume_yes: bool) -> Any:
    """Look ``name`` up on the Hub and let the user pick; None if nothing.

    Shows every candidate with its size, the most likely first. ``assume_yes``
    takes the first; without a terminal nothing is downloaded — the exact
    reference is printed instead, so a script never pulls gigabytes on a
    guess.
    """
    from hfl.hub.shortname import find_options

    console.print(f"[dim]{escape_markup(t('shortname.searching', name=name))}[/]")
    try:
        options = find_options(name)
    except Exception as exc:
        if _hub_unreachable(exc):
            _print_hub_unreachable()
            raise typer.Exit(1) from None
        raise
    if not options:
        console.print(f"[red]{escape_markup(t('shortname.none', name=name))}[/]")
        return None
    for number, option in enumerate(options, start=1):
        size = option.size_bytes / 1024**3
        console.print(
            f"  [cyan]{number}[/]  {escape_markup(option.repo_id)}:{option.quantization}"
            f"  [dim]{size:.1f} GB · {option.downloads:,} downloads[/]"
        )
    if assume_yes:
        return options[0]
    if not stdin_is_terminal():
        console.print(
            escape_markup(
                t("shortname.not_interactive", command=f"hfl pull {options[0].reference}")
            )
        )
        raise typer.Exit(1)
    answer = typer.prompt(t("shortname.pick"), default="1").strip().lower()
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    console.print(f"[dim]{escape_markup(t('shortname.cancelled'))}[/]")
    raise typer.Exit(1)


def _from_short_name(name: str, registry_cls: Any, *, assume_yes: bool) -> Any:
    """The model a short name stands for: the one chosen before (kept as an
    alias), or one chosen now from the Hub, pulled and remembered."""
    from hfl.hub.shortname import alias_for

    alias = alias_for(name)
    remembered = registry_cls().get(alias)
    if remembered is not None:
        return remembered
    choice = _choose_short_name(name, assume_yes)
    if choice is None:
        return None
    local = registry_cls().find_pulled(choice.repo_id, choice.quantization)
    if local is None:
        pull(
            model=choice.reference,
            quantize=choice.quantization,
            format="auto",
            revision=None,
            alias=alias,
            skip_license=False,
        )
        local = registry_cls().find_pulled(choice.repo_id, choice.quantization)
    elif not local.alias:
        registry_cls().set_alias(local.name, alias)
    return local


def _configured_port(port: int | None) -> int:
    """``--port`` when given, else the configured port: ``HFL_PORT``, the
    port in ``OLLAMA_HOST``, ``OLLAMA_PORT``, then 11434. The option used to
    default to a fixed 11434, so none of those ever applied (local audit
    D38) — not to ``serve``, nor to the commands that talk to it."""
    if port is not None:
        return port
    from hfl.config import config

    return config.port


# Shown first, in this order; any other declared extra follows. ``dev`` and
# ``build`` are for working on HFL itself, not for using it.
_EXTRA_ORDER = [
    "llama", "vulkan", "rocm", "transformers", "vllm", "mlx", "structured", "convert", "tts",
    "coqui", "stt", "imagegen", "audio", "mcp", "otel", "tray", "all",
]  # fmt: skip
_DEVELOPER_EXTRAS = {"dev", "build"}


def _declared_extras() -> tuple[list[str], dict[str, list[str]]]:
    """The extras the installed hfl declares (``Provides-Extra``), in display
    order, and each one's requirements (``Requires-Dist`` with its marker)."""
    import importlib.metadata as metadata
    import re

    try:
        declared = metadata.metadata("hfl").get_all("Provides-Extra") or []
        requires = metadata.requires("hfl") or []
    except metadata.PackageNotFoundError:
        declared, requires = list(_EXTRA_ORDER), []
    packages: dict[str, list[str]] = {}
    for requirement in requires:
        spec, _, marker = requirement.partition(";")
        for extra in re.findall(r"extra\s*==\s*['\"]([\w.-]+)['\"]", marker):
            packages.setdefault(extra, []).append(spec.strip())
    shown = [e for e in declared if e not in _DEVELOPER_EXTRAS]
    known = [e for e in _EXTRA_ORDER if e in shown and e != "all"]
    # One this list does not know yet still comes before ``all``, the last.
    order = known + sorted(set(shown) - set(_EXTRA_ORDER)) + (["all"] if "all" in shown else [])
    return order, packages


def escape_markup(text: str) -> str:
    from rich.markup import escape

    return escape(text)


def _no_controls(text: object) -> str:
    """``text`` without control characters (newline and tab kept).

    Manifest fields come from Hub metadata or an imported entry: an ESC in
    one drove the user's terminal (title, colours, cursor) when printed.
    """
    import re

    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", str(text))


def _plain(text: object) -> str:
    """Untrusted text for a markup context: controls dropped, markup escaped
    (an unbalanced ``[/x]`` in a license string crashed with MarkupError)."""
    return escape_markup(_no_controls(text))


def _memory_check_or_exit(manifest: Any, n_ctx: int) -> None:
    """Say what loading ``manifest`` will do to memory, and refuse if it
    cannot fit under HFL_MEMORY_BUDGET — before any weights are read."""
    from rich.markup import escape

    from hfl.engine.residency import check_standalone_load

    check = check_standalone_load(manifest.local_path, n_ctx)
    if check.plan is None or check.memory is None:
        if check.memory is not None and not check.footprint:
            console.print(f"[dim]{escape(t('messages.memory_unknown', model=manifest.name))}[/]")
        return
    gib = 1024**3
    total = check.memory.total or 1
    fields = {
        "model": manifest.name,
        "in_use": f"{check.memory.in_use / gib:.1f}",
        "total": f"{check.memory.total / gib:.1f}",
        "pct": f"{100 * check.memory.in_use / total:.0f}",
        "need": f"{check.footprint / gib:.1f}",
        "after": f"{check.plan.used_after / gib:.1f}",
        "after_pct": f"{100 * check.plan.used_after / total:.0f}",
        "budget": f"{check.budget * 100:.0f}",
    }
    if check.plan.fits:
        console.print(f"[dim]{escape(t('messages.memory_report', **fields))}[/]")
        return
    if check.plan.constraint == "gpu" and check.plan.gpu_limit:
        gpu_total = int(check.plan.gpu_limit / check.budget) or 1
        fields.update(
            after=f"{check.plan.gpu_floor / gib:.1f}",
            total=f"{gpu_total / gib:.1f}",
            after_pct=f"{100 * check.plan.gpu_floor / gpu_total:.0f}",
        )
        console.print(f"[red]{escape(t('errors.memory_refused_gpu', **fields))}[/]")
    else:
        console.print(f"[red]{escape(t('errors.memory_refused', **fields))}[/]")
    console.print(escape(t("errors.memory_refused_hint")))
    raise typer.Exit(1)


def _hub_unreachable(exc: BaseException) -> bool:
    """Whether a Hub command failed because there is no network at all."""
    from hfl.hub.connectivity import is_network_error

    return is_network_error(exc)


def _print_hub_unreachable() -> None:
    """Say "offline" once, plainly, and what still works.

    Being offline is a normal state for a local runner, not a fault, so
    the message leads with the cause and follows with the remedy instead
    of printing whatever the socket layer raised ("[Errno 8] nodename nor
    servname provided") and leaving the user to translate it.
    """
    console.print(f"[yellow]{t('errors.hub_unreachable')}[/]")
    console.print(f"[dim]{t('errors.hub_unreachable_hint')}[/]")


_BACKENDS = ("auto", "llama-cpp", "llama-server", "transformers", "vllm", "mlx")


_DEFAULT_PARALLEL = 4
# What says "I chose": any of these set means the default is not applied.
_PARALLEL_CHOICES = (
    "HFL_NUM_PARALLEL",
    "OLLAMA_NUM_PARALLEL",
    "HFL_QUEUE_MAX_INFLIGHT",
    "HFL_LLM_LIBRARY",
    "OLLAMA_LLM_LIBRARY",
)


def _parallel_by_default() -> bool:
    """Whether ``hfl serve`` serves GGUF models with parallel slots without
    being asked: llama-server is installed and nothing chose otherwise
    (``HFL_NUM_PARALLEL=1`` or ``--backend llama-cpp`` keep one at a time)."""
    import os

    if any(os.environ.get(name, "").strip() for name in _PARALLEL_CHOICES):
        return False
    from hfl.engine.llama_server import binary

    return binary() is not None


def _choose_backend(backend: str, parallel: int) -> None:
    """Apply ``--backend`` / ``--parallel`` for this server.

    The backend is still chosen per model: ``auto`` keeps the usual choice,
    and a forced ``llama-server`` only takes the GGUF models (vision ones with
    their projector; anything that is not GGUF keeps its own backend). ``--parallel
    N`` above 1 asks for N requests at once per model, which on GGUF needs
    llama-server — so it implies it when no backend was named.
    """
    import os

    from hfl.config import config as hfl_config
    from hfl.engine.selector import plugin_engines

    backend = backend.strip().lower()
    plugins = tuple(plugin_engines())
    if backend not in _BACKENDS and backend not in plugins:
        choices = ", ".join((*_BACKENDS, *plugins))
        message = t("errors.unknown_backend", backend=backend, choices=choices)
        console.print(f"[red]{escape_markup(message)}[/]")
        raise typer.Exit(2)
    if parallel < 0:
        console.print(f"[red]{escape_markup(t('errors.bad_parallel'))}[/]")
        raise typer.Exit(2)
    if parallel == 0 and backend == "auto" and _parallel_by_default():
        # Nothing named a backend or a level of parallelism, and
        # llama-server is here: serve GGUF models with parallel slots. One
        # at a time, four requests took four times one (bench: 1.0x);
        # through llama-server 1.9x, single requests level on real models.
        parallel = _DEFAULT_PARALLEL
        console.print(f"[cyan]{escape_markup(t('messages.parallel_default'))}[/]")
    if parallel > 1 and backend == "auto":
        backend = "llama-server"
    if backend == "llama-server":
        from hfl.engine.llama_server import binary

        if binary() is None:
            console.print(f"[red]{escape_markup(t('errors.llama_server_missing'))}[/]")
            raise typer.Exit(1)
    if backend != "auto":
        os.environ["HFL_LLM_LIBRARY"] = backend
    if parallel > 0:
        hfl_config.queue_max_inflight = parallel
        hfl_config.parallel_explicit = True
    if backend == "llama-server":
        from hfl.engine.llama_server import _slots

        slots = _slots()
        console.print(f"[cyan]{escape_markup(t('messages.parallel_on', slots=slots))}[/]")


def _in_container() -> bool:
    """Whether this process runs in a container (Docker, Podman, Kubernetes).

    There, binding ``0.0.0.0`` opens only the container's own network
    namespace: what reaches the outside is what whoever started it published
    (``docker run -p``, compose ``ports``, a Service).
    """
    import os
    from pathlib import Path

    return (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    )


# Inode of the initial (host) network namespace: fixed since Linux 6.18
# (PROC_NET_INIT_INO); earlier kernels numbered it like any other namespace.
_HOST_NETNS_INO = 0xEFFFFFF9


def _container_network() -> str:
    """``"host"``, ``"own"`` or ``"unknown"``: whose network this container binds.

    ``--network host`` / ``hostNetwork: true`` put the container in the host's
    network namespace: 0.0.0.0 there is the host's real interfaces, not a
    namespace reachable only through published ports. Comparing with PID 1's
    namespace says nothing (PID 1 is the container's own init either way);
    the namespace's inode does, on kernels that fix the host's (measured:
    0xEFFFFFF9 under ``docker run --network host`` on 7.0, a fresh one under
    the default bridge). On older kernels, or without /proc, it is unknown.
    """
    if not sys.platform.startswith("linux"):
        return "unknown"
    try:
        ino = os.stat("/proc/self/ns/net").st_ino
    except OSError:
        return "unknown"
    if ino == _HOST_NETNS_INO:
        return "host"
    try:
        major, minor = (int(x) for x in os.uname().release.split(".")[:2])
    except ValueError:
        return "unknown"
    return "own" if (major, minor) >= (6, 18) else "unknown"


def _is_public_bind(host: str) -> bool:
    """Whether binding to ``host`` exposes the server beyond this machine.

    Anything that is not a loopback address counts as exposure, including
    ``0.0.0.0``, ``::`` and any concrete LAN address. A value that does not
    parse as an IP (a hostname) is also treated as exposure: this gate
    exists to prevent an accidental opening, so the unknown case must fail
    toward warning rather than toward silence.
    """
    import ipaddress

    if host in ("0.0.0.0", "::", ""):  # noqa: S104 - detection, not a bind
        return True
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def _apply_sandbox(sandbox: str | None) -> None:
    # Process hardening, before anything is served. Restrictions that drop
    # privileges only hold if they are applied before the first request, and
    # ``apply_sandbox`` never raises: an unsupported platform logs a warning
    # and serves unhardened, because "opt-in hardening" that refuses to boot
    # is a denial of service the operator did not ask for. The flag falls
    # back to HFL_SANDBOX so a container can set it without changing its
    # command line.
    import os as _os

    from hfl.core.sandbox import apply_sandbox

    _sandbox_result = apply_sandbox(sandbox or _os.environ.get("HFL_SANDBOX"))
    if _sandbox_result.mode != "none" and not _sandbox_result.applied:
        console.print(
            f"[yellow]Sandbox '{_sandbox_result.mode}' requested but not applied: "
            f"{_sandbox_result.reason}[/]"
        )


def _run_tray(
    host: str, port: int, api_key: str | None, model: str | None, log_level: str, json_logs: bool
) -> None:
    """``serve --tray``: the server under a system tray icon."""
    # On a Linux with no desktop session (a server, SSH, a container)
    # pystray fails as it is imported, reaching for an X display: that
    # was a DisplayNameError traceback. Say what is missing instead.
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        console.print(f"[red]{escape_markup(t('errors.tray_no_display'))}[/]")
        raise typer.Exit(1)
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
            "Install with: [cyan]pip install hfl\\[tray][/]"
        )
        raise typer.Exit(1) from None
    except Exception as exc:
        # DISPLAY set but no X server answering it (Xlib's errors).
        if not type(exc).__module__.startswith("Xlib"):
            raise
        console.print(f"[red]{escape_markup(t('errors.tray_no_display'))}[/] ({exc})")
        raise typer.Exit(1) from None


def _confirm_exposure(host: str, api_key: str | None) -> None:
    """Warn — and, unattended, refuse unless decided — when ``host`` is not
    a loopback address."""
    # R6 - Privacy warning when exposing to the network.
    #
    # SEC: this used to test ``host == "0.0.0.0"`` literally, so `--host ::`
    # (every IPv6 interface) and `--host 192.168.1.10` (a LAN address)
    # exposed the server with no warning at all. Anything that is not a
    # loopback address is an exposure; an unparseable value is treated as
    # one too, because guessing in the permissive direction is what this
    # check exists to prevent.
    if _is_public_bind(host):
        console.print(f"[yellow]Warning:[/] {t('warnings.network_exposure')}")
        if api_key:
            console.print(f"[green]{t('messages.api_key_enabled')}[/]")
        else:
            console.print(f"[yellow]{t('warnings.no_api_key')}[/]")
        # Without a TTY (systemd, Docker, launchd) ``typer.confirm`` cannot
        # ask anyone — and those are exactly the deployments where an
        # accidental exposure matters most. An unattended start exposes the
        # server only when someone decided it: the explicit opt-in, an API
        # key (the exposure is authenticated, and someone set the key), or a
        # container (where the bind reaches only as far as the ports its
        # operator published). The container image used to stop here, every
        # time: nobody can answer a prompt in one. The opt-in is consent
        # with a terminal too: a hidden console (a Windows scheduled task)
        # is a terminal nobody reads, and the question waited there forever.
        opted_in = os.environ.get("HFL_ACCEPT_NETWORK_EXPOSURE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if opted_in:
            pass
        elif not stdin_is_terminal():
            # The container convenience holds only while the container has a
            # network of its own: under host networking 0.0.0.0 is the host's
            # interfaces, an exposure nobody published.
            network = _container_network() if (_in_container() and not api_key) else None
            if network == "host":
                console.print(f"[yellow]{t('warnings.container_host_network')}[/]")
            if not (api_key or (network is not None and network != "host")):
                console.print(f"[red]{t('warnings.refuse_unattended_bind', host=host)}[/]")
                raise typer.Exit(1)
            if network == "own":
                console.print(f"[yellow]{t('warnings.container_bind')}[/]")
            elif network == "unknown":
                console.print(f"[yellow]{t('warnings.container_bind_unverified')}[/]")
        elif not typer.confirm(t("warnings.continue_question"), default=True):
            raise typer.Exit(0)


def _preload(model: str, ctx: int, state: Any) -> None:
    """``serve --model``: the model loaded before the first request."""
    from pathlib import Path

    from hfl.engine.selector import MissingDependencyError, select_engine
    from hfl.models.registry import ModelRegistry

    manifest = _local_or_pulled(model, ModelRegistry)
    if manifest is None:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {escape_markup(model)}")
        raise typer.Exit(1)
    if manifest:
        _memory_check_or_exit(manifest, ctx if ctx > 0 else 0)
        console.print(f"[cyan]{t('messages.pre_loading')}[/] {manifest.name}...")
        try:
            n_ctx = ctx if ctx > 0 else 0  # 0 = auto-detect from model
            from hfl.api.model_loader import load_kwargs_for

            state.engine = select_engine(Path(manifest.local_path))
            state.engine.load(manifest.local_path, **load_kwargs_for(manifest, n_ctx))
            state.current_model = manifest
        except MissingDependencyError as e:
            console.print(f"[red]{t('errors.missing_dependency')}:[/]\n\n{escape_markup(str(e))}")
            raise typer.Exit(1) from e


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host", help=t("commands.serve.options.host")),
    port: int | None = typer.Option(None, "--port", "-p", help=t("commands.serve.options.port")),
    model: str = typer.Option(None, "--model", "-m", help=t("commands.serve.options.model")),
    # Also from HFL_API_KEY — better than the flag, which every local user
    # can read in ``ps``; docker-compose passed it and nothing read it.
    api_key: str = typer.Option(
        None, "--api-key", envvar="HFL_API_KEY", help=t("commands.serve.options.api_key")
    ),
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
    sandbox: str = typer.Option(None, "--sandbox", help=t("commands.serve.options.sandbox")),
    backend: str = typer.Option("auto", "--backend", help=t("commands.serve.options.backend")),
    parallel: int = typer.Option(0, "--parallel", help=t("commands.serve.options.parallel")),
):
    """Start the API server (OpenAI + Ollama + Anthropic compatible)."""
    from hfl.api.server import start_server
    from hfl.api.state import get_state
    from hfl.logging_config import configure_logging

    port = _configured_port(port)
    # Initialize structured logging
    configure_logging(level=log_level, json_format=json_logs)

    _apply_sandbox(sandbox)

    # Host resolution: --host wins, then HFL_HOST / OLLAMA_HOST via config,
    # then the loopback default. The flag's default used to be the literal
    # "127.0.0.1", so ``config.host`` (which reads both env vars, and is
    # documented in docs/env-vars.md) could never be reached from `serve`.
    # The failure was in the safe direction — the server stayed on loopback
    # — but an operator who set HFL_HOST expecting exposure got a server
    # that silently ignored them.
    #
    # Resolved BEFORE the tray branch so tray mode binds the same address
    # the headless path would, and before the exposure check below so that
    # a host coming from the environment is warned about too.
    if host is None:
        from hfl.config import config as _cfg

        host = _cfg.host

    # Before the tray branch: the tray serves the same host, and returning
    # into it first let HFL_HOST=0.0.0.0 bind publicly with no key and no
    # question asked.
    _confirm_exposure(host, api_key)

    if tray:
        _run_tray(host, port, api_key, model, log_level, json_logs)
        return

    # Store context size override in state for lazy-load path
    state = get_state()
    if ctx > 0:
        state.context_size_override = ctx
    _choose_backend(backend, parallel)

    if model:
        _preload(model, ctx, state)

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
            shown = _plain(m.license)
            if any(nc in m.license.lower() for nc in nc_licenses):
                license_str = f"[red]{shown}[/]"
            elif m.license.lower() in ["apache-2.0", "mit", "bsd"]:
                license_str = f"[green]{shown}[/]"
            else:
                license_str = f"[yellow]{shown}[/]"

        # Table cells are markup too: every manifest string is escaped.
        table.add_row(
            _plain(m.name),
            _plain(m.alias or "-"),
            type_str,
            _plain(m.format),
            _plain(m.quantization or "-"),
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
            revision=None,
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
        raise typer.Exit(1) from exc

    if ok:
        console.print(f"[green]Copied[/] [cyan]{source}[/] → [cyan]{destination}[/]")
    else:
        console.print(f"[red]Copy failed[/] (concurrent write?): {source} → {destination}")
        raise typer.Exit(1)


@app.command(name="import")
def import_model(
    path: str = typer.Argument(
        help="A GGUF file, or a folder holding one model (a GGUF, or MLX / Hugging Face weights)"
    ),
    name: str | None = typer.Option(None, "--name", "-n", help="Name to register it under"),
    alias: str | None = typer.Option(None, "--alias", "-a", help="Short alias"),
) -> None:
    """Register a model you already have, where it is — no copy, no server.

    For models downloaded by LM Studio, llama.cpp, huggingface-cli or by
    hand: a GGUF, or a folder of MLX or Hugging Face weights (config.json and
    .safetensors). It stays in place; ``hfl rm`` removes the entry and never
    deletes it. An image projector (``mmproj``) beside a GGUF is used for
    images.
    """
    from hfl.models.importer import (
        ImportRefused,
        choose_model,
        default_name,
        manifest_for,
        manifest_for_folder,
    )
    from hfl.models.registry import ModelRegistry

    try:
        model, kind = choose_model(Path(path))
        final = name or default_name(model)
        registry = ModelRegistry()
        if registry.get(final) is not None or (alias and registry.get(alias) is not None):
            taken = final if registry.get(final) is not None else str(alias)
            console.print(f"[red]{escape_markup(t('import.exists', name=taken))}[/]")
            raise typer.Exit(1)
        if kind == "gguf":
            manifest = manifest_for(model, final, alias)
        else:
            manifest = manifest_for_folder(model, final, alias)
    except ImportRefused as refused:
        console.print(f"[red]{escape_markup(t(refused.key, **refused.fields))}[/]")
        raise typer.Exit(1) from refused
    registry.add(manifest)
    console.print(
        "[green]"
        + escape_markup(t("import.done", name=final, size=manifest.display_size, path=str(model)))
        + "[/]"
    )
    if manifest.model_type == "embedding":
        embed = t("messages.use_embed", name=alias or final)
        console.print(f"{t('messages.use_command')}: {embed}")
    elif manifest.model_type == "tts":
        console.print(f'{t("messages.use_command")}: hfl tts {alias or final} "..."')
    else:
        console.print(t("import.use", name=alias or final))


@app.command(name="outdated")
def outdated(
    model: str | None = typer.Argument(None, help="One model to check (default: all)"),
) -> None:
    """Which pulled models have a newer version on the Hub — nothing is downloaded.

    Compares the commit each model was pulled at with the Hub's, and when it
    moved, the model's own files (so a README edit is not an update). Says
    which ``hfl pull`` fetches the new files. Exits 1 when some model could
    not be checked (no network, a gated repo without a token...).
    """
    from huggingface_hub import HfApi
    from rich.table import Table

    from hfl.config import config
    from hfl.hub.outdated import check_all
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    if model:
        found = registry.get(model)
        if found is None:
            console.print(f"[red]{t('errors.model_not_found')}:[/] {escape_markup(model)}")
            raise typer.Exit(1)
        manifests = [found]
    else:
        manifests = registry.list_all()
    if not manifests:
        console.print(t("outdated.none"))
        return
    api = HfApi()
    table = Table(title=t("outdated.title"))
    table.add_column(t("outdated.model"))
    table.add_column(t("outdated.status"))
    table.add_column(t("outdated.update_with"))
    unchecked = 0
    for manifest, result in zip(manifests, check_all(manifests, api, config.models_dir)):
        why = t(f"outdated.why_{result.detail}", error=result.error) if result.detail else ""
        status = t(f"outdated.{result.status}", files=", ".join(result.changed), why=why)
        unchecked += result.status == "unchecked"
        table.add_row(
            escape_markup(manifest.alias or manifest.name),
            escape_markup(status),
            escape_markup(result.command or ""),
        )
    console.print(table)
    if unchecked:
        console.print(f"[yellow]{t('outdated.some_unchecked', count=unchecked)}[/]")
        raise typer.Exit(1)


@app.command(name="stop")
def stop(
    model: str = typer.Argument(None, help="Model name to unload. Omit to unload all."),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="HFL server host"),
    port: int | None = typer.Option(None, "--port", "-p", help=t("options.server_port")),
) -> None:
    """Unload a model without restarting the server (Ollama-compatible).

    Sends ``POST /api/stop`` to the running HFL server. If a model
    name is given, only that one is evicted; otherwise every loaded
    model (LLM + TTS) is released.
    """
    import httpx

    port = _configured_port(port)
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
        raise typer.Exit(1) from None
    except httpx.HTTPError as exc:
        console.print(f"[red]Server error:[/] {exc}")
        raise typer.Exit(1) from exc

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
        console.print(f"[red]Model not found:[/] {_plain(model)}")
        raise typer.Exit(1)

    # Single-section flags first (mirror ollama show --modelfile). Each is
    # manifest text: printed literally (markup=False) and without controls.
    if modelfile:
        console.print(
            _no_controls(render_modelfile(manifest)), end="", markup=False, highlight=False
        )
        return
    if parameters:
        from hfl.api.routes_show import _format_parameters

        console.print(_no_controls(_format_parameters(manifest)), markup=False, highlight=False)
        return
    if template:
        from hfl.models.chat_template import model_template

        # markup=False: templates are full of [ ] that Rich would eat.
        console.print(_no_controls(model_template(manifest)), markup=False, highlight=False)
        return
    if license_only:
        console.print(
            _no_controls(manifest.license_name or manifest.license or ""),
            markup=False,
            highlight=False,
        )
        return

    # Default: summary table — same columns ollama show prints.
    from rich.panel import Panel
    from rich.table import Table

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right")
    summary.add_column()
    summary.add_row("Name", _plain(manifest.name))
    summary.add_row("Architecture", _plain(manifest.architecture or "unknown"))
    summary.add_row("Parameters", _plain(manifest.parameters or "?"))
    summary.add_row("Quantization", _plain(manifest.quantization or "?"))
    summary.add_row("Format", _plain(manifest.format or "?"))
    if manifest.context_length:
        summary.add_row("Context", f"{manifest.context_length} tokens")
    summary.add_row("Size", manifest.display_size)
    summary.add_row(
        "Capabilities",
        _plain(", ".join(detect_capabilities(manifest)) or "—"),
    )
    summary.add_row("License", _plain(manifest.license or "—"))

    console.print(Panel(summary, title=f"Model: {_plain(manifest.name)}", expand=False))


@app.command(name="ps")
def ps(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-H", help="Host where the HFL server is running"
    ),
    port: int | None = typer.Option(None, "--port", "-p", help=t("options.server_port")),
) -> None:
    """List models currently loaded in memory (Ollama-compatible).

    Hits the server's ``/api/ps`` endpoint and renders a table with
    NAME, ID, SIZE, PROCESSOR and UNTIL — matching the output layout
    of ``ollama ps`` so scripts written against Ollama work unchanged.
    """
    import httpx
    from rich.table import Table

    port = _configured_port(port)
    url = f"http://{host}:{port}/api/ps"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
    except httpx.ConnectError:
        console.print(
            f"[red]Cannot reach HFL server at {url}[/]\n"
            f"[dim]Start it first with:[/] [cyan]hfl serve[/]"
        )
        raise typer.Exit(1) from None
    except httpx.HTTPError as exc:
        console.print(f"[red]Server error:[/] {exc}")
        raise typer.Exit(1) from exc

    data = response.json()
    models = data.get("models", [])

    if not models:
        console.print("[dim]No models loaded. Send a request to /api/chat to load one.[/]")
        _print_memory_summary(data.get("memory"))
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
    _print_memory_summary(data.get("memory"))


def _print_memory_summary(memory: Any) -> None:
    """The server's memory against HFL_MEMORY_BUDGET (``/api/ps`` HFL
    extension); silent when an older server does not send it."""
    if not isinstance(memory, dict) or not memory.get("total_bytes"):
        return
    gib = 1024**3
    console.print(
        t(
            "messages.memory_summary",
            in_use=f"{memory.get('in_use_bytes', 0) / gib:.1f}",
            total=f"{memory['total_bytes'] / gib:.1f}",
            pct=f"{memory.get('in_use_percent', 0):.0f}",
            budget=f"{memory.get('budget_percent', 0):.0f}",
            limit=f"{memory.get('budget_bytes', 0) / gib:.1f}",
            free=f"{memory.get('free_within_budget_bytes', 0) / gib:.1f}",
            models=f"{memory.get('models_bytes', 0) / gib:.1f}",
        ),
        markup=False,
    )
    gpu = memory.get("gpu")
    if isinstance(gpu, dict) and gpu.get("total_bytes"):
        console.print(
            t(
                "messages.memory_summary_gpu",
                in_use=f"{gpu.get('in_use_bytes', 0) / gib:.1f}",
                total=f"{gpu['total_bytes'] / gib:.1f}",
                pct=f"{gpu.get('in_use_percent', 0):.0f}",
                free=f"{gpu.get('free_within_budget_bytes', 0) / gib:.1f}",
            ),
            markup=False,
        )


def _search_hub(
    api: Any,
    query: str,
    searches: list[dict],
    sort: str,
    limit: int,
    gguf_only: bool,
    max_params: float | None,
    min_params: float | None,
) -> list[Any]:
    """The Hub's models for ``searches`` (one, or several for a query read
    as an intent), merged; exits 1 when the Hub cannot be reached."""
    try:
        # Search models with progress spinner
        with progress_spinner(t("messages.searching", query=query)):
            size_filter = max_params is not None or min_params is not None
            kwargs: dict = {
                "sort": sort,
                # The size filter can only run on names, after the Hub answers;
                # fetch a wider window so it filters more than the first page.
                "limit": min(limit * 10, 1000) if size_filter else limit,
                "fetch_config": False,
                "full": True,  # To get siblings and detect GGUF
            }
            if gguf_only:
                # Filtered by the Hub itself, over every GGUF repo — not over
                # whatever GGUF happened to be among the top `limit` results.
                kwargs["filter"] = "gguf"
            # ``sort="downloads"`` is already descending on hub API v1;
            # the legacy ``direction=-1`` kwarg was removed in hub 1.0.
            found: dict[str, Any] = {}
            for extra in searches:
                for m in api.list_models(**kwargs, **extra):
                    found.setdefault(m.id, m)
            models = list(found.values())
            if len(searches) > 1 and sort in ("downloads", "likes"):
                # Several searches merged: order them as one list.
                models.sort(key=lambda m: getattr(m, sort, 0) or 0, reverse=True)
    except Exception as e:
        if _hub_unreachable(e):
            _print_hub_unreachable()
        else:
            console.print(f"[red]{t('errors.error_searching')}:[/] {e}")
        raise typer.Exit(1) from e
    return models


def _filter_search(
    models: list[Any],
    query: str,
    gguf_only: bool,
    max_params: float | None,
    min_params: float | None,
) -> list[Any] | None:
    """``models`` kept to GGUF repos and a size range, as asked; None (and
    said why) when nothing is left."""
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
            return None

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
            return None
    return models


def _page_through(models: list[Any], query: str, page_size: int) -> None:
    """``models`` a page at a time: a digit pulls one, SPACE shows the next
    page, ``p`` the previous one, ``q`` stops."""
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

        # Every page — the last one included — waits for a key, so a model
        # can be picked wherever it is listed. (The last page used to print
        # "end of results" and return: a search that fit on one page could
        # not be picked from at all.)
        is_last = current_page >= total_pages - 1
        hints = f"[dim]0-9[/] {t('messages.select_to_pull')}  [dim]q[/] {t('messages.quit')}"
        if is_last:
            status = f"{page_info}  [bold green]{t('messages.end_of_results')}[/]  {hints}"
        else:
            status = f"{page_info}  [dim]SPACE[/] {t('messages.next_page')}  {hints}"
        console.print(status, end="")

        try:
            key = get_key()
        except Exception:
            # No raw keyboard (a pipe, some Windows consoles): read a line
            # instead, where a typed number still selects.
            try:
                typed = input("\n[ENTER / 0-9 / q]: ").strip().lower()
                key = typed if typed else " "
            except (EOFError, KeyboardInterrupt):
                key = "q"

        # Clear status line
        console.print("\r" + " " * 80 + "\r", end="")

        if key in ("q", "\x1b", "\x03") or key.startswith("q"):  # q, ESC, Ctrl+C
            console.print(f"\n[dim]{t('messages.search_finished', shown=end_idx, total=total)}[/]")
            break
        if key == "p" and current_page > 0:
            current_page -= 1
            console.print()  # New line before previous page
        elif key.isdigit():
            selection = int(key)
            if selection < len(page_models):
                console.print()
                _pull_selected_model(page_models[selection])
                return
            console.print()  # Not on this page: show it again
        elif is_last:
            break
        else:
            current_page += 1
            console.print()  # New line before next page

    # Show help at the end
    console.print()
    console.print(f"[dim]{t('messages.to_download')}[/]")


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
    literal: bool = typer.Option(False, "--literal", help=t("commands.search.options.literal")),
):
    """Search models on HuggingFace Hub with interactive pagination."""
    from huggingface_hub import HfApi

    from hfl.hub.query import describe, hub_queries, parse

    # Validate minimum length
    if len(query.strip()) < 3:
        console.print(f"[red]Error:[/] {t('errors.search_min_chars')}")
        raise typer.Exit(1)

    api = HfApi()

    # "coding assistant 7b" means a coding model of about 7B, not repos whose
    # name contains that phrase (the Hub's search matches ids: it found five,
    # the best with 7 downloads). Read the query; say how it was read.
    intent = parse(query)
    searches: list[dict] = [{"search": query}]
    if not literal and intent.interpreted:
        searches = hub_queries(intent)
        gguf_only = gguf_only or intent.gguf
        size_range = intent.size_range()
        if size_range is not None and max_params is None and min_params is None:
            min_params, max_params = size_range
        console.print(f"[dim]{t('messages.search_interpreted', reading=describe(intent))}[/]")

    models = _search_hub(api, query, searches, sort, limit, gguf_only, max_params, min_params)
    if not models:
        console.print(f"[yellow]{t('errors.no_models_found', query=query)}[/]")
        return

    filtered = _filter_search(models, query, gguf_only, max_params, min_params)
    if filtered is None:
        return
    models = filtered
    models = models[:limit]

    _page_through(models, query, page_size)


@app.command()
def rm(
    model: str = typer.Argument(help=t("commands.rm.args.model")),
    yes: bool = typer.Option(False, "--yes", "-y", help=t("commands.rm.options.yes")),
):
    """Delete a local model."""
    from hfl.models.registry import ModelRegistry

    registry = ModelRegistry()
    manifest = registry.get(model)

    if not manifest:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
        raise typer.Exit(1)

    # Confirm — unless --yes was given (for scripting / automation).
    if not yes and not typer.confirm(
        t("confirm.delete_model", name=manifest.name, size=manifest.display_size),
    ):
        return

    # Same rule as DELETE /api/delete (hfl.models.removal): only files inside
    # HFL's models folder are deleted, and a blob another entry shares
    # (``hfl cp`` is zero-copy) is kept so ``cp a b; rm a`` leaves ``b``.
    from hfl.models.removal import ModelInUse, remove_model

    try:
        result = remove_model(registry, manifest)
    except ModelInUse:
        console.print(f"[red]{escape_markup(t('errors.model_in_use', name=manifest.name))}[/]")
        raise typer.Exit(1) from None
    if result.shared_with:
        names = ", ".join(result.shared_with)
        console.print(f"[yellow]{t('messages.blob_shared', names=names)}[/]")
    elif result.kept_outside is not None:
        console.print(
            f"[yellow]{escape_markup(t('messages.file_kept_outside', path=result.kept_outside))}[/]"
        )
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
        raise typer.Exit(1) from e


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
    port: int | None = typer.Option(None, "--port", "-p", help=t("options.server_port")),
) -> None:
    """Create a new model from a Modelfile (Ollama-compatible).

    Mirrors ``ollama create`` — sends the Modelfile body to
    ``POST /api/create`` on a running HFL server and prints the NDJSON
    progress events as they stream in.
    """
    import httpx

    body = modelfile.read_text()
    port = _configured_port(port)
    url = f"http://{host}:{port}/api/create"
    payload: dict[str, Any] = {
        "model": model,
        "modelfile": body,
        "stream": True,
    }

    try:
        status = ""
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
        # Only "success" means the model exists: a stream that ended before it
        # (the server failed mid-way) used to exit 0 with nothing created.
        if status != "success":
            console.print(f"[red]{t('errors.create_incomplete')}[/]")
            raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(
            f"[red]Cannot reach HFL server at {url}[/]\n"
            f"[dim]Start it first with:[/] [cyan]hfl serve[/]"
        )
        raise typer.Exit(1) from None
    except httpx.HTTPError as exc:
        console.print(f"[red]Server error:[/] {exc}")
        raise typer.Exit(1) from exc


@app.command(name="train", help=t("train.description"))
def train(
    model: str = typer.Argument(help="The model to train on (safetensors or MLX)"),
    data: Path = typer.Option(
        ..., "--data", "-d", help="A JSONL file, or a folder with train.jsonl (valid.jsonl)"
    ),
    name: str | None = typer.Option(None, "--name", "-n", help="The trained model's name"),
    iters: int = typer.Option(600, "--iters", min=1),
    batch_size: int = typer.Option(4, "--batch-size", min=1),
    num_layers: int = typer.Option(16, "--num-layers", help="Layers to adapt (-1: all)"),
    learning_rate: float = typer.Option(1e-5, "--learning-rate"),
    max_seq_length: int = typer.Option(2048, "--max-seq-length", min=16),
    save_every: int = typer.Option(100, "--save-every", min=1),
    resume: bool = typer.Option(False, "--resume", help="Continue from the saved adapter"),
    fuse: bool = typer.Option(False, "--fuse", help="Also merge the adapter into a new model"),
    gguf: str | None = typer.Option(
        None, "--gguf", help="Also export the merged model as GGUF at this quantization"
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="auto (MLX on Apple Silicon, else Transformers), mlx, transformers",
    ),
) -> None:
    """Train a LoRA adapter (mlx-lm on Apple Silicon, Transformers + PEFT
    elsewhere) and register the result as a model.

    The data is JSONL in one of mlx-lm's formats: {"messages": [...]},
    {"prompt": ..., "completion": ...} or {"text": ...}. Without a
    valid.jsonl a tenth of the rows validates. Runs in a process of its own;
    Ctrl-C stops it and --resume continues from the last saved adapter.
    """
    from rich.markup import escape

    from hfl.config import config as hfl_config
    from hfl.core.container import get_registry
    from hfl.training import hf_lora, mlx_lora

    if backend not in ("auto", "mlx", "transformers"):
        console.print(f"[red]--backend: auto, mlx or transformers, not {escape(backend)}[/]")
        raise typer.Exit(2)
    use_mlx = backend == "mlx" or (backend == "auto" and mlx_lora.available() is None)
    trainer: Any = mlx_lora if use_mlx else hf_lora
    why_not = trainer.available()
    if why_not:
        console.print(f"[yellow]{escape(why_not)}[/]")
        raise typer.Exit(1)
    base = get_registry().get(model)
    if base is None:
        console.print(f"[red]{t('errors.model_not_found')}:[/] {escape(model)}")
        raise typer.Exit(1)
    target = name or f"{base.name}-lora"
    try:
        trainer.check_name(target)
        problem = trainer.trainable(base)
        if problem:
            raise trainer.TrainingError(problem)
        if get_registry().get(target) is not None and not resume:
            raise trainer.TrainingError(t("train.exists", name=target))
        home = Path(hfl_config.home_dir)
        prepared = trainer.prepare_data(data, home / "training" / target / "data")
    except trainer.TrainingError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from None
    console.print(
        t("train.data", train=prepared.train, valid=prepared.valid, format=prepared.format)
    )
    options = trainer.Options(
        iters=iters,
        batch_size=batch_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        save_every=save_every,
        resume=resume,
    )
    adapter = home / "adapters" / target
    log = home / "logs" / f"train-{target}.log"
    last: dict[str, Any] = {}

    def show(event: dict[str, Any]) -> None:
        last.update(event)
        extra = " · ".join(
            part
            for part in (
                f"val {last['val_loss']:.3f}" if "val_loss" in last else "",
                f"{last['tokens_per_sec']:.0f} tok/s" if "tokens_per_sec" in last else "",
                f"{last['peak_memory_gb']:.1f} GB" if "peak_memory_gb" in last else "",
            )
            if part
        )
        loss = f"{last['train_loss']:.3f}" if "train_loss" in last else "–"
        console.print(
            t("train.progress", iteration=event["iteration"], iters=iters, loss=loss, extra=extra)
        )

    try:
        trainer.run(trainer.command(str(base.local_path), prepared, adapter, options), log, show)
    except KeyboardInterrupt:
        console.print(
            f"[yellow]{escape(t('train.stopped', model=model, data=data, name=target))}[/]"
        )
        raise typer.Exit(130) from None
    except trainer.TrainingError as exc:
        console.print(f"[red]{escape(str(exc))}[/] [dim]({log})[/]")
        raise typer.Exit(1) from None
    if trainer is hf_lora:
        _finish_transformers_training(base, target, adapter, log, gguf)
        return
    trainer.register(base, target, adapter)
    console.print(f"[green]{escape(t('train.done', name=target, adapter=adapter))}[/]")
    if fuse or gguf:
        console.print(t("train.fusing"))
        folder = home / "models" / f"{target}-fused"
        try:
            trainer.fuse(base, adapter, folder, log, dequantize=bool(gguf))
            trainer.register_fused(base, f"{target}-fused", folder)
            console.print(f"[green]{escape(t('train.fused', name=f'{target}-fused'))}[/]")
            if gguf:
                trainer.to_gguf(base, f"{target}-gguf", folder, gguf.upper())
                console.print(f"[green]{escape(t('train.gguf', name=f'{target}-gguf'))}[/]")
        except Exception as exc:  # the trained adapter stands either way
            console.print(f"[red]{escape(str(exc))}[/] [dim]({log})[/]")
            raise typer.Exit(1) from None


def _finish_transformers_training(
    base: Any, target: str, adapter: Path, log: Path, gguf: str | None
) -> None:
    """The Transformers engine does not load a separate adapter: the trained
    model is the adapter merged into a copy of the base (``--fuse`` is
    implied), and ``--gguf`` exports that."""
    from rich.markup import escape

    from hfl.training import hf_lora

    console.print(t("train.fusing"))
    try:
        merged = hf_lora.register(base, target, adapter, log)
        console.print(f"[green]{escape(t('train.done', name=target, adapter=adapter))}[/]")
        if gguf:
            hf_lora.to_gguf(base, f"{target}-gguf", Path(merged.local_path), gguf.upper())
            console.print(f"[green]{escape(t('train.gguf', name=f'{target}-gguf'))}[/]")
    except Exception as exc:  # the trained adapter stands either way
        console.print(f"[red]{escape(str(exc))}[/] [dim]({log})[/]")
        raise typer.Exit(1) from None


@app.command(name="mcp")
def mcp(
    action: str = typer.Argument(help="connect | disconnect | list | serve"),
    server_id: str = typer.Argument(None, help="Server id (for connect/disconnect)"),
    target: str = typer.Argument(None, help="stdio://<cmd> <args> or sse://<url>"),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        help="For ``serve``: stdio (default) or sse.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="SSE bind host"),
    port: int = typer.Option(8765, "--port", "-p", help="SSE bind port"),
    capabilities: str = typer.Option(
        None,
        "--capabilities",
        help="Comma-separated subset of tools to expose when serving.",
    ),
) -> None:
    """Manage Model Context Protocol (MCP) connections and run as a server.

    Examples:

        hfl mcp list
        hfl mcp connect fs stdio://npx @modelcontextprotocol/server-filesystem /tmp
        hfl mcp disconnect fs
        hfl mcp serve --transport stdio
        hfl mcp serve --transport sse --host 0.0.0.0 --port 8765 \
                      --capabilities web_search,web_fetch
    """
    import asyncio

    from hfl.mcp.client import (
        MCPClientUnavailableError,
        MCPConnectionError,
        get_client,
    )

    client = get_client()

    async def _run() -> None:
        if action == "list":
            tools = client.list_tools()
            if not tools:
                console.print("[dim]No MCP servers connected.[/]")
                return
            for tool in tools:
                console.print(f"[cyan]{tool.qualified_name}[/]  [dim]{tool.description}[/]")
        elif action == "connect":
            if not server_id or not target:
                console.print("[red]Usage:[/] hfl mcp connect <id> stdio://... or sse://...")
                raise typer.Exit(1)
            tools = await client.connect(server_id, target)
            console.print(f"[green]Connected[/] {server_id} ({len(tools)} tools)")
            for tool in tools:
                console.print(f"  [cyan]{tool.qualified_name}[/]")
        elif action == "disconnect":
            if not server_id:
                console.print("[red]Usage:[/] hfl mcp disconnect <id>")
                raise typer.Exit(1)
            await client.disconnect(server_id)
            console.print(f"[green]Disconnected[/] {escape_markup(str(server_id))}")
        elif action == "serve":
            from hfl.mcp.server import (
                MCPServerUnavailableError,
                serve_sse,
                serve_stdio,
            )

            cap_list = None
            if capabilities:
                cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]
            try:
                if transport == "stdio":
                    await serve_stdio(cap_list)
                elif transport == "sse":
                    await serve_sse(host, port, cap_list)
                else:
                    console.print(f"[red]Unknown transport:[/] {escape_markup(str(transport))}")
                    raise typer.Exit(1)
            except MCPServerUnavailableError as exc:
                console.print(f"[red]MCP unavailable:[/] {escape_markup(str(exc))}")
                raise typer.Exit(1) from exc
        else:
            console.print(f"[red]Unknown action:[/] {escape_markup(str(action))}")
            raise typer.Exit(1)

    try:
        asyncio.run(_run())
    except MCPClientUnavailableError as exc:
        console.print(f"[red]MCP unavailable:[/] {escape_markup(str(exc))}")
        raise typer.Exit(1) from exc
    except MCPConnectionError as exc:
        console.print(f"[red]MCP error:[/] {escape_markup(str(exc))}")
        raise typer.Exit(1) from exc


@app.command()
def doctor():
    """Diagnose the runtime environment (Phase 15 P2 — V2 row 15).

    Prints detected accelerators (NVIDIA / Metal / ROCm), which HFL
    extras are installed (llama-cpp / transformers / vllm / mlx-lm),
    VRAM probe result + recommended ``num_ctx``, and actionable
    follow-up suggestions.
    """
    from hfl.cli.commands.doctor import build_report, format_report

    report = build_report()
    console.print(format_report(report))


@app.command()
def version():
    """Show the hfl version."""
    from hfl import __version__

    console.print(f"hfl v{__version__} — Licensed under Apache-2.0")
    console.print("[dim]https://github.com/ggalancs/hfl[/]")


@app.command()
def config():
    """Show current configuration."""
    from rich.panel import Panel
    from rich.text import Text

    from hfl.config import config as cfg

    info = Text()
    info.append_text(Text.from_markup("[bold]Directories[/]\n"))
    info.append(f"  Home:     {cfg.home_dir}\n")
    info.append(f"  Models:   {cfg.models_dir}\n")
    info.append(f"  Cache:    {cfg.cache_dir}\n")
    info.append(f"  Registry: {cfg.registry_path}\n")

    info.append_text(Text.from_markup("\n[bold]Server[/]\n"))
    info.append(f"  Host: {cfg.host}\n")
    info.append(f"  Port: {cfg.port}\n")

    info.append_text(Text.from_markup("\n[bold]Rate Limiting[/]\n"))
    info.append(f"  Enabled:  {cfg.rate_limit_enabled}\n")
    info.append(f"  Requests: {cfg.rate_limit_requests}/min\n")

    info.append_text(Text.from_markup("\n[bold]Inference Defaults[/]\n"))
    info.append(f"  Context Size: {cfg.default_ctx_size}\n")
    info.append(f"  GPU Layers:   {cfg.default_n_gpu_layers} (-1 = all)\n")
    info.append(f"  Threads:      {cfg.default_threads} (0 = auto)\n")

    info.append_text(Text.from_markup("\n[bold]Timeouts (seconds)[/]\n"))
    info.append(f"  Model Load:   {cfg.model_load_timeout}\n")
    info.append(f"  Generation:   {cfg.generation_timeout}\n")

    info.append_text(Text.from_markup("\n[bold]SLO Targets[/]\n"))
    info.append(f"  Availability: {cfg.slo.availability_target * 100:.1f}%\n")
    info.append(f"  Latency P50:  {cfg.slo.latency_p50_ms}ms\n")
    info.append(f"  Latency P95:  {cfg.slo.latency_p95_ms}ms\n")
    info.append(f"  Latency P99:  {cfg.slo.latency_p99_ms}ms\n")
    info.append(f"  Error Rate:   {cfg.slo.error_rate_target * 100:.1f}%\n")

    info.append_text(Text.from_markup("\n[bold]HuggingFace[/]\n"))
    if cfg.hf_token:
        info.append_text(Text.from_markup("  Token: [green]Configured[/]\n"))
    else:
        info.append_text(Text.from_markup("  Token: [dim]Not set[/] (use HF_TOKEN env var)\n"))

    console.print(Panel(info, title="[bold]HFL Configuration[/]", border_style="cyan"))


@app.command()
def check():
    """Run diagnostic checks (dependencies, backends, GPU)."""
    from hfl.cli.commands.doctor import accelerator_rows, backend_rows, build_report
    from hfl.engine.dependency_check import check_engine_availability

    console.print("[bold]Running HFL Diagnostics[/]\n")

    # Backends and accelerators come from the same probe as `hfl doctor`
    # and `hfl debug`, so the three commands can no longer disagree.
    report = build_report()
    console.print("[bold cyan]Backend Availability[/]")
    for name, ok, detail in backend_rows(report):
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        suffix = f" [dim]{escape_markup(detail)}[/]" if detail else ""
        console.print(f"  {mark} {name}{suffix}")

    console.print("\n[bold cyan]GPU Support[/]")
    for label, detail in accelerator_rows(report):
        mark = "[yellow]○[/]" if label == "CPU only" else "[green]✓[/]"
        suffix = f": {escape_markup(detail)}" if detail else ""
        console.print(f"  {mark} {label}{suffix}")

    availability = check_engine_availability()

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
    info.append_text(Text.from_markup("[bold]System[/]\n"))
    info.append(f"  Python:   {sys.version.split()[0]}\n")
    info.append(f"  Platform: {platform.system()} {platform.release()}\n")
    info.append(f"  Machine:  {platform.machine()}\n")

    # HFL info
    info.append_text(Text.from_markup("\n[bold]HFL[/]\n"))
    info.append(f"  Version:  {__version__}\n")
    info.append(f"  Home:     {cfg.home_dir}\n")

    # Dependency versions
    info.append_text(Text.from_markup("\n[bold]Dependencies[/]\n"))

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
    info.append_text(Text.from_markup("\n[bold]Optional Dependencies[/]\n"))
    availability = check_engine_availability()

    for dep in ["llama-cpp-python", "transformers", "torch", "vllm", "soundfile", "torchaudio"]:
        version = get_version(dep)
        if version != "not installed":
            info.append(f"  {dep}: {version}\n")
        else:
            info.append_text(Text.from_markup(f"  {dep}: [dim]not installed[/]\n"))
    from hfl.engine.llama_server import binary as llama_server_binary

    server = llama_server_binary()
    if server:
        info.append(f"  llama-server: {server}\n")
    else:
        info.append_text(Text.from_markup("  llama-server: [dim]not installed[/]\n"))

    # GPU info — the same probe as `hfl doctor` and `hfl check`.
    from hfl.cli.commands.doctor import accelerator_rows, build_report

    report = build_report()
    info.append_text(Text.from_markup("\n[bold]GPU[/]\n"))
    for label, detail in accelerator_rows(report):
        info.append(f"  {label}" + (f": {detail}" if detail else "") + "\n")
    if availability.get("torch_cuda"):
        try:
            import torch

            info.append(f"  CUDA Version: {torch.version.cuda}\n")
            info.append(f"  cuDNN: {torch.backends.cudnn.version()}\n")
        except Exception:
            pass

    # Memory info
    try:
        import psutil

        mem = psutil.virtual_memory()
        info.append_text(Text.from_markup("\n[bold]Memory[/]\n"))
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

    from hfl import __version__
    from hfl.models.registry import get_registry

    # Any other format wrote nothing yet said "Report saved" and exited 0
    # (local audit A5: --format pdf).
    if format not in ("json", "markdown"):
        console.print(f"[red]Unknown format {format!r}:[/] use json or markdown")
        raise typer.Exit(2)

    registry = get_registry()
    models = registry.list_all()

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "hfl_version": __version__,
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
    import importlib.util

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

        # Every extra the installed package declares, and its packages as
        # declared: a list kept by hand here missed seven (local audit A13)
        # and quoted pins long since changed.
        extra_order, extra_packages = _declared_extras()

        table = Table(show_header=True, border_style="dim")
        table.add_column(t("table.name"), style="cyan", min_width=14)
        table.add_column("", min_width=40)
        table.add_column("Status", justify="center", min_width=14)
        table.add_column("pip install", style="dim", min_width=22)

        for extra_name in extra_order:
            info = t(f"help.extras.{extra_name}.summary")
            install_cmd = f"pip install 'hfl[{extra_name}]'"
            key = f"help.extras.{extra_name}.check_module"
            check_module = t(key)
            # A null in the locale comes back from t() as the key itself.
            if check_module in ("null", key):
                check_module = ""

            # Installed = findable. Importing it ran the package: slow for
            # torch or vllm, and pystray raises Xlib's DisplayNameError on a
            # Linux without a display (local audit A13 on Linux).
            if check_module:
                try:
                    found = importlib.util.find_spec(check_module) is not None
                except (ImportError, ValueError):
                    found = False
                if found:
                    status = f"[green]{t('help.extras_installed')}[/]"
                else:
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
            packages = ", ".join(extra_packages.get(extra_name, [])) or "—"
            install_cmd = f"pip install 'hfl[{extra_name}]'"

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
            ("launch claude -m <m>", t("help.common_launch")),
            ("search <query>", t("help.common_search")),
            ("list", t("help.common_list")),
        ]
        for cmd, desc in commands:
            help_text.append(f"  hfl {cmd:<22}", style="cyan")
            help_text.append(f" {desc}\n")

        help_text.append(f"\n{t('help.more_info')}\n\n", style="dim")
        help_text.append(t("help.extras_hint") + "\n", style="bold")

        console.print(Panel(help_text, title="[bold]hfl[/]", border_style="cyan"))


@app.command(help=t("commands.discover.description"))
def discover(
    query: str | None = typer.Argument(default=None, help=t("commands.discover.args.query")),
    family: str | None = typer.Option(
        None, "--family", "-f", help=t("commands.discover.options.family")
    ),
    task: str | None = typer.Option(None, "--task", "-t", help=t("commands.discover.options.task")),
    quantization: str | None = typer.Option(
        None, "--quant", "-q", help=t("commands.discover.options.quant")
    ),
    multimodal: bool = typer.Option(
        False, "--multimodal", help=t("commands.discover.options.multimodal")
    ),
    min_likes: int = typer.Option(0, "--min-likes", help=t("commands.discover.options.min_likes")),
    license_filter: str | None = typer.Option(
        None, "--license", help=t("commands.discover.options.license")
    ),
    gated: bool | None = typer.Option(
        None, "--gated/--open", help=t("commands.discover.options.gated")
    ),
    page_size: int = typer.Option(20, "--limit", "-l", help=t("commands.discover.options.limit")),
    refresh: bool = typer.Option(False, "--refresh", help=t("commands.discover.options.refresh")),
):
    """Filter the live Hugging Face Hub by capability and popularity.

    Combines filters: family + quantisation + likes + license +
    multimodal. Cached 5 min on disk (override with ``--refresh``).
    """
    from rich.table import Table

    from hfl.api.routes_discover import _annotate_local_availability, _build_cache
    from hfl.hub.discovery import DiscoveryQuery, format_size_human, search_hub

    q = DiscoveryQuery(
        q=query,
        family=family,
        task=task,
        quantization=quantization,
        multimodal=multimodal,
        min_likes=min_likes,
        license=license_filter,
        gated=gated,
        page_size=page_size,
    )

    cache = _build_cache()
    entries = None if refresh else cache.get(q)
    cached_label = "(cached)"
    if entries is None:
        cached_label = ""
        try:
            entries = search_hub(q)
        except Exception as exc:
            console.print(f"[red]Hub unavailable:[/] {exc}")
            raise typer.Exit(1) from exc
        cache.put(q, entries)

    _annotate_local_availability(entries)

    if not entries:
        console.print("[yellow]No matching models found.[/]")
        return

    table = Table(title=f"HF Hub discovery {cached_label}".strip(), show_lines=False)
    table.add_column("repo_id", style="cyan", no_wrap=False)
    table.add_column("family", style="magenta")
    table.add_column("size", justify="right")
    table.add_column("quant", style="yellow")
    table.add_column("likes", justify="right")
    table.add_column("downloads", justify="right")
    table.add_column("local", justify="center")

    for e in entries:
        table.add_row(
            e.repo_id,
            e.family or "-",
            format_size_human(e.parameter_estimate_b),
            e.quantization or "-",
            f"{e.likes:,}",
            f"{e.downloads:,}",
            "[green]✓[/]" if e.locally_available else "",
        )
    console.print(table)


# The first model offered, by the RAM it needs (GB): Apache-2.0 and not
# gated, so nobody has to sign up or accept terms; small first, so the
# first answer comes in minutes, not after a 15 GB download.
_FIRST_MODELS: tuple[tuple[float, str], ...] = (
    (0, "qwen2.5:1.5b"),
    (16, "qwen2.5:7b"),
    (0, "qwen2.5:0.5b"),
    (32, "qwen2.5:14b"),
)


@app.command("start", help=t("commands.start.description"))
def start(
    yes: bool = typer.Option(False, "--yes", "-y", help=t("commands.start.options.yes")),
    chat: bool = typer.Option(True, "--chat/--no-chat", help=t("commands.start.options.chat")),
) -> None:
    """First run: a model that fits this machine, downloaded, and a chat."""
    from hfl.hub.hw_profile import get_hw_profile
    from hfl.hub.shortname import alias_for

    profile = get_hw_profile()
    host = t(
        "messages.start_host",
        os=profile.os,
        arch=profile.arch,
        ram=round(profile.system_ram_gb),
        gpu=profile.gpu_kind,
    )
    console.print(f"[dim]{host}[/]")
    have = _a_chat_model()
    if have is not None:
        console.print(t("messages.start_have", name=have))
        if chat is True:
            _chat_with(have)
        return
    offers = _first_model_offers(profile.system_ram_gb)
    if not offers:
        console.print(f"[yellow]{t('messages.start_none')}[/]")
        raise typer.Exit(1)
    console.print(t("messages.start_offer"))
    for number, (name, match) in enumerate(offers, 1):
        size = f"{match.size_bytes / 1e9:.1f} GB"
        console.print(f"  {number}. {name}  [dim]{match.repo_id} · {size}[/]")
    name = offers[(1 if yes is True else _pick(len(offers))) - 1][0]
    pull(
        model=name,
        quantize="Q4_K_M",
        format="auto",
        revision=None,
        alias=None,
        skip_license=False,
        yes=True,
    )
    alias = alias_for(name)
    console.print(f"[green]{t('messages.start_ready', alias=alias)}[/]")
    if chat is True:
        _chat_with(alias)


def _chat_with(model: str) -> None:
    # None where Typer would pass None for an option left unset.
    run(
        model=model,
        backend="auto",
        ctx=0,
        system=None,  # type: ignore[arg-type]
        session=None,  # type: ignore[arg-type]
        yes=True,
        verbose=False,
    )


def _a_chat_model() -> str | None:
    """A text model already here (its alias, else its name), or None."""
    from hfl.models.registry import ModelRegistry

    for manifest in ModelRegistry().list_all():
        if (manifest.model_type or "llm") == "llm":
            return str(manifest.alias or manifest.name)
    return None


def _first_model_offers(ram_gb: float) -> list[tuple[str, Any]]:
    """Up to three short names that fit ``ram_gb``, with the GGUF build each
    resolves to (size, repo); those the Hub cannot answer for are left out."""
    from hfl.hub.shortname import find

    names = [name for need, name in _FIRST_MODELS if ram_gb <= 0 or ram_gb >= need][:3]
    offers = []
    with progress_spinner(t("messages.searching", query="qwen2.5")):
        for name in names:
            try:
                match = find(name)
            except Exception:  # offline: said below when nothing is found
                continue
            if match is not None:
                offers.append((name, match))
    return offers


def _pick(count: int) -> int:
    """The number the user typed (Enter: 1)."""
    while True:
        answer = console.input(f"{t('messages.start_choose')}: ").strip()
        if not answer:
            return 1
        if answer.isdigit() and 1 <= int(answer) <= count:
            return int(answer)


@app.command(help=t("commands.recommend.description"))
def recommend(
    task: str | None = typer.Option(
        None, "--task", "-t", help=t("commands.recommend.options.task")
    ),
    family: str | None = typer.Option(
        None, "--family", "-f", help=t("commands.recommend.options.family")
    ),
    quantization: str | None = typer.Option(
        None, "--quant", "-q", help=t("commands.recommend.options.quant")
    ),
    top_n: int = typer.Option(5, "--top", "-n", help=t("commands.recommend.options.top")),
):
    """HW-aware top-N model recommendations.

    Combines the Hub catalogue, your hardware profile (RAM, VRAM,
    MLX availability), and a capability/popularity score to pick
    models that will actually run well on this machine.
    """
    from dataclasses import asdict

    from rich.table import Table

    from hfl.hub.hw_profile import get_hw_profile
    from hfl.hub.recommend import recommend_models

    profile = get_hw_profile()
    valid_tasks = {"chat", "code", "vision", "embeddings", "tools"}
    if task is not None and task not in valid_tasks:
        console.print(f"[red]Error:[/] task must be one of {sorted(valid_tasks)}")
        raise typer.Exit(1)

    try:
        recs = recommend_models(
            task=task,  # type: ignore[arg-type]
            profile=profile,
            family=family,
            quantization=quantization,
            top_n=top_n,
        )
    except Exception as exc:
        if _hub_unreachable(exc):
            _print_hub_unreachable()
        else:
            console.print(f"[red]{t('errors.error_resolving')}:[/] {exc}")
        raise typer.Exit(1) from exc

    profile_dict = asdict(profile)
    console.print(
        f"[dim]Host: {profile_dict['os']}/{profile_dict['arch']} "
        f"RAM={profile_dict['system_ram_gb']}GB GPU={profile_dict['gpu_kind']} "
        f"VRAM={profile_dict['gpu_vram_gb'] or 'n/a'}GB[/]"
    )

    if not recs:
        console.print(
            "[yellow]No models fit this hardware. Try smaller params or relax filters.[/]"
        )
        return

    table = Table(title=f"Top {len(recs)} for {task or 'general use'}", show_lines=False)
    table.add_column("repo_id", style="cyan")
    table.add_column("family", style="magenta")
    table.add_column("quant", style="yellow")
    table.add_column("est. VRAM", justify="right")
    table.add_column("score", justify="right", style="green")
    table.add_column("why", style="dim")
    for r in recs:
        why = "; ".join(r.reasoning[:2])
        table.add_row(
            r.repo_id,
            r.family or "-",
            r.quantization or "-",
            f"{r.estimated_vram_gb:.1f} GB",
            f"{r.score:.2f}",
            why,
        )
    console.print(table)


def _server_request(
    method: str, host: str, port: int | None, path: str, body: dict | None = None
) -> Any:
    """``method path`` on the running HFL server; its JSON. A server that
    cannot be reached, or answers an error, is a message and exit 1 — the
    server's own ``detail`` shown — never a traceback. Sends HFL_API_KEY
    when set, for a server started with one."""
    import httpx

    url = f"http://{host}:{_configured_port(port)}{path}"
    headers = {}
    key = os.environ.get("HFL_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        response = httpx.request(method, url, json=body, headers=headers, timeout=600.0)
    except httpx.ConnectError:
        console.print(
            f"[red]Cannot reach HFL server at {url}[/]\n"
            f"[dim]Start it first with:[/] [cyan]hfl serve[/]"
        )
        raise typer.Exit(1) from None
    except httpx.HTTPError as exc:
        console.print(f"[red]Server error:[/] {escape_markup(str(exc))}")
        raise typer.Exit(1) from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail") or response.json().get("error")
        except ValueError:
            detail = response.text
        console.print(f"[red]{response.status_code}:[/] {escape_markup(str(detail))}")
        raise typer.Exit(1)
    return response.json()


@app.command(name="lora", help=t("commands.lora.description"))
def lora_cmd(
    action: str = typer.Argument(help=t("commands.lora.args.action")),
    model: str = typer.Argument(default="", help=t("commands.lora.args.model")),
    lora_path: str | None = typer.Option(None, "--path", help=t("commands.lora.options.path")),
    adapter_id: str | None = typer.Option(None, "--id", help=t("commands.lora.options.id")),
    scale: float = typer.Option(1.0, "--scale", help=t("commands.lora.options.scale")),
    name: str | None = typer.Option(None, "--name", help=t("commands.lora.options.name")),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="HFL server host"),
    port: int | None = typer.Option(None, "--port", "-p", help=t("options.server_port")),
) -> None:
    """hot-swap LoRA adapters on a model of the running server.

    It used to load the model into this CLI process, change the adapter
    there and exit: the server's model never had it (local audit A20).

    Examples::

        hfl lora apply qwen-7b --path adapters/code.safetensors --scale 0.7
        hfl lora list qwen-7b
        hfl lora remove qwen-7b --id <adapter-uuid>
    """
    from rich.table import Table

    if action not in {"apply", "remove", "list"}:
        console.print("[red]Action must be one of: apply, remove, list[/]")
        raise typer.Exit(1)
    if action != "list" and not model:
        console.print("[red]Model name is required for apply/remove[/]")
        raise typer.Exit(1)

    if action == "apply":
        if not lora_path:
            console.print("[red]--path is required for apply[/]")
            raise typer.Exit(1)
        body = {"model": model, "lora_path": lora_path, "scale": scale, "name": name}
        info = _server_request("POST", host, port, "/api/lora/apply", body)
        console.print(f"[green]Applied[/] adapter id=[cyan]{info['adapter_id']}[/]")
        return
    if action == "remove":
        if not adapter_id:
            console.print("[red]--id is required for remove[/]")
            raise typer.Exit(1)
        body = {"model": model, "adapter_id": adapter_id}
        _server_request("POST", host, port, "/api/lora/remove", body)
        console.print("[green]Removed[/]")
        return

    path = f"/api/lora/{model}" if model else "/api/lora"
    adapters = _server_request("GET", host, port, path)["adapters"]
    if not adapters:
        console.print("[dim]No adapters active.[/]")
        return
    table = Table(title="Active LoRA adapters", show_lines=False)
    table.add_column("id", style="cyan")
    table.add_column("name")
    table.add_column("path", style="dim")
    table.add_column("scale", justify="right")
    for a in adapters:
        table.add_row(a["adapter_id"], a.get("name") or "-", a["path"], f"{a['scale']:.2f}")
    console.print(table)


@app.command(name="pull-smart", help=t("commands.pull-smart.description"))
def pull_smart_cmd(
    model: str = typer.Argument(help=t("commands.pull-smart.args.model")),
    max_vram_gb: float | None = typer.Option(
        None, "--max-vram-gb", help=t("commands.pull-smart.options.max_vram_gb")
    ),
):
    """pull the optimal Hub variant for the current hardware.

    Inspects MLX / GGUF community forks, picks the best (repo, quant)
    pair that fits the host budget, and downloads it. On Apple
    Silicon resolves to ``mlx-community/<name>-4bit``; on CUDA picks
    a ``bartowski/...-GGUF`` quant; on CPU-only falls back to the
    smallest quant that fits.
    """
    import asyncio

    from rich.table import Table

    from hfl.hub.smart_pull import build_smart_plan

    try:
        plan = build_smart_plan(model, max_vram_gb=max_vram_gb)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Hub unavailable:[/] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title="Smart pull plan", show_lines=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("target_repo_id", plan.target_repo_id)
    table.add_row("quantization", plan.quantization)
    table.add_row("estimated_vram_gb", f"{plan.estimated_vram_gb:.1f} GB")
    table.add_row("reason", plan.reason)
    console.print(table)
    if plan.fallback_chain:
        console.print("[dim]Skipped candidates:[/]")
        for skip in plan.fallback_chain:
            console.print(f"  - {skip}")

    # Delegate the actual byte transfer to the existing /api/pull
    # machinery via the public helper added in V5 β3.
    from hfl.api.routes_pull import iter_pull_events

    console.print(f"\n[green]Now pulling[/] {plan.target_repo_id} via the existing pull command...")

    async def _pull() -> None:
        async for line in iter_pull_events(plan.target_repo_id, quantization=plan.quantization):
            console.print(line.rstrip())

    try:
        asyncio.run(_pull())
    except Exception as exc:
        console.print(f"[red]Pull failed:[/] {exc}")
        raise typer.Exit(1) from exc


@app.command(name="verify", help=t("commands.verify.description"))
def verify_cmd(
    model: str = typer.Argument(help=t("commands.verify.args.model")),
):
    """sanity-check a registered model.

    Runs five probes (tokenizer round-trip, chat-template render,
    smoke generation, tool-parser, embedding dim) and prints a pass/
    fail report in seconds.
    """
    import asyncio

    from rich.table import Table

    from hfl.api.model_loader import load_llm
    from hfl.engine.verifier import verify_model
    from hfl.exceptions import ModelNotFoundError

    async def _run() -> None:
        try:
            engine, manifest = await load_llm(model)
        except (ModelNotFoundError, FileNotFoundError) as exc:
            # load_llm raises ModelNotFoundError; only FileNotFoundError was
            # caught, so a missing model was a traceback (local audit A2/A39).
            console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
            console.print(t("errors.use_list_to_see"))
            raise typer.Exit(1) from exc
        if engine is None:
            console.print("[red]Engine not available[/]")
            raise typer.Exit(1)
        result = verify_model(engine, manifest)
        title = (
            f"[green]VERIFY PASS[/] {result.model} ({result.duration_ms:.1f} ms)"
            if result.overall_pass
            else f"[red]VERIFY FAIL[/] {result.model} ({result.duration_ms:.1f} ms)"
        )
        console.print(title)
        table = Table(show_lines=False)
        table.add_column("check", style="cyan")
        table.add_column("status")
        table.add_column("detail", style="dim")
        for c in result.checks:
            if getattr(c, "skipped", False):
                badge = "[yellow]SKIP[/]"
            elif c.passed:
                badge = "[green]PASS[/]"
            else:
                badge = "[red]FAIL[/]"
            table.add_row(c.name, badge, c.detail)
        console.print(table)
        if not result.overall_pass:
            raise typer.Exit(1)

    asyncio.run(_run())


@app.command(name="bench", help=t("commands.bench.description"))
def bench_cmd(
    model: str = typer.Argument(help=t("commands.bench.args.model")),
    runs_per_length: int = typer.Option(
        3, "--runs", "-n", min=1, max=20, help=t("commands.bench.options.runs")
    ),
    max_tokens: int = typer.Option(
        64, "--max-tokens", "-t", min=1, max=2048, help=t("commands.bench.options.max_tokens")
    ),
    prompt_lengths: str = typer.Option(
        "16,256,2048", "--lengths", help=t("commands.bench.options.lengths")
    ),
):
    """benchmark TTFT + tok/s on a registered model.

    Streams per-run measurements and a final p50/p95 summary. Useful
    to validate that a freshly pulled model performs as expected on
    your hardware.
    """
    import asyncio

    from rich.table import Table

    from hfl.api.model_loader import load_llm
    from hfl.engine.benchmark import run_benchmark_stream
    from hfl.exceptions import ModelNotFoundError

    try:
        lengths = tuple(int(v.strip()) for v in prompt_lengths.split(",") if v.strip())
    except ValueError:
        console.print("[red]Invalid --lengths value (use comma-separated integers)[/]")
        raise typer.Exit(1) from None

    async def _run() -> None:
        try:
            engine, _ = await load_llm(model)
        except (ModelNotFoundError, FileNotFoundError) as exc:
            # load_llm raises ModelNotFoundError; only FileNotFoundError was
            # caught, so a missing model was a traceback (local audit A2/A39).
            console.print(f"[red]{t('errors.model_not_found')}:[/] {model}")
            console.print(t("errors.use_list_to_see"))
            raise typer.Exit(1) from exc
        if engine is None:
            console.print("[red]Engine not available[/]")
            raise typer.Exit(1)

        summaries = []
        async for event in run_benchmark_stream(
            engine,
            model_name=model,
            runs_per_length=runs_per_length,
            max_tokens=max_tokens,
            prompt_lengths=lengths,
        ):
            status = event.get("status")
            if status == "starting":
                console.print(
                    f"[dim]Bench {event['model']}: {event['runs_per_length']} runs × "
                    f"{event['prompt_lengths']} chars[/]"
                )
            elif status == "run":
                console.print(
                    f"  [{event['prompt_length']} chars run] "
                    f"ttft={event['ttft_ms']:.1f}ms "
                    f"total={event['total_ms']:.1f}ms "
                    f"tps={event['tokens_per_second']:.2f}"
                )
            elif status == "summary":
                summaries.append(event)
            elif status == "done":
                pass

        table = Table(title=f"Benchmark — {model}", show_lines=False)
        table.add_column("prompt", justify="right")
        table.add_column("runs", justify="right")
        table.add_column("ttft p50", justify="right")
        table.add_column("ttft p95", justify="right")
        table.add_column("tps mean", justify="right", style="green")
        table.add_column("tps min", justify="right")
        table.add_column("tps max", justify="right")
        for s in summaries:
            ttft50 = s.get("ttft_p50_ms")
            ttft95 = s.get("ttft_p95_ms")
            table.add_row(
                str(s["prompt_length"]),
                str(s["runs"]),
                f"{ttft50:.1f}" if ttft50 is not None else "—",
                f"{ttft95:.1f}" if ttft95 is not None else "—",
                f"{s['tps_mean']:.2f}",
                f"{s['tps_min']:.2f}",
                f"{s['tps_max']:.2f}",
            )
        console.print(table)

    asyncio.run(_run())


@app.command(name="snapshot", help=t("commands.snapshot.description"))
def snapshot_cmd(
    action: str = typer.Argument(help=t("commands.snapshot.args.action")),
    model: str = typer.Argument(default="", help=t("commands.snapshot.args.model")),
    name: str = typer.Option("", "--name", help=t("commands.snapshot.options.name")),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="HFL server host"),
    port: int | None = typer.Option(None, "--port", "-p", help=t("options.server_port")),
) -> None:
    """KV cache snapshot save/restore, on the running server's model.

    Save a "warm" KV cache after loading a long system prompt or
    few-shot context, then restore it on the next server start to
    skip the prefill. It used to load a model of its own in this CLI
    process — an empty cache, saved with tokens=0 — and restore into a
    model that exited with it (local audit A35).

    Examples::

        hfl snapshot save qwen-coder-7b --name warm-1
        hfl snapshot list
        hfl snapshot load qwen-coder-7b --name warm-1
        hfl snapshot delete --name warm-1
    """
    from rich.table import Table

    if action not in {"save", "load", "list", "delete"}:
        console.print("[red]Action must be one of: save, load, list, delete[/]")
        raise typer.Exit(1)
    if action in {"save", "load"} and not model:
        console.print("[red]Model is required for save/load[/]")
        raise typer.Exit(1)
    if action in {"save", "load", "delete"} and not name:
        console.print("[red]--name is required[/]")
        raise typer.Exit(1)

    if action == "list":
        entries = _server_request("GET", host, port, "/api/snapshot")["snapshots"]
        if not entries:
            console.print("[dim]No snapshots saved.[/]")
            return
        table = Table(title="KV cache snapshots", show_lines=False)
        table.add_column("name", style="cyan")
        table.add_column("model")
        table.add_column("tokens", justify="right")
        table.add_column("bytes", justify="right")
        for e in entries:
            table.add_row(e["name"], e["model"], str(e["tokens"]), f"{e['bytes']:,}")
        console.print(table)
        return
    if action == "delete":
        _server_request("DELETE", host, port, f"/api/snapshot/{name}")
        console.print(f"[green]Deleted[/] snapshot {name!r}")
        return

    meta = _server_request(
        "POST", host, port, f"/api/snapshot/{action}", {"model": model, "name": name}
    )
    if action == "save":
        console.print(f"[green]Saved[/] {name!r} — tokens={meta['tokens']} bytes={meta['bytes']:,}")
    else:
        console.print(f"[green]Restored[/] {name!r} — tokens={meta['tokens']}")


@app.command(name="compliance-dashboard", help=t("commands.compliance-dashboard.description"))
def compliance_dashboard_cmd():
    """license / EU AI Act compliance overview of the local registry.

    Different from ``compliance-report`` (which produces a per-model
    Markdown audit trail) — this is the at-a-glance dashboard:
    counts by license risk, gated repos pending HF_TOKEN, models
    with no declared license, EU AI Act warnings.
    """
    from rich.table import Table

    from hfl.api.routes_compliance import _build_compliance_dashboard

    snapshot = _build_compliance_dashboard()

    console.print(f"\n[bold]Compliance dashboard[/]  total={snapshot['total_models']}")
    console.print(f"[dim]HF_TOKEN configured: {snapshot['has_hf_token']}[/]\n")

    risk_table = Table(title="By license risk", show_lines=False)
    risk_table.add_column("risk", style="cyan")
    risk_table.add_column("count", justify="right")
    for risk, count in sorted(snapshot["by_risk"].items()):
        risk_table.add_row(risk, str(count))
    console.print(risk_table)

    if snapshot["by_license"]:
        lic_table = Table(title="By license id", show_lines=False)
        lic_table.add_column("license")
        lic_table.add_column("count", justify="right")
        for lic, count in sorted(snapshot["by_license"].items()):
            lic_table.add_row(lic, str(count))
        console.print(lic_table)

    if snapshot["gated_without_token"]:
        console.print("\n[yellow]Gated models without HF_TOKEN:[/]")
        for name in snapshot["gated_without_token"]:
            console.print(f"  - {name}")

    if snapshot["missing_license"]:
        console.print("\n[yellow]Models without a declared license:[/]")
        for name in snapshot["missing_license"]:
            console.print(f"  - {name}")

    if snapshot["eu_ai_act_warnings"]:
        console.print("\n[red]EU AI Act warnings:[/]")
        for w in snapshot["eu_ai_act_warnings"]:
            console.print(f"  - {w['model']} ({w['license']}): {w['reason']}")


@app.command(name="draft-recommend", help=t("commands.draft-recommend.description"))
def draft_recommend_cmd(
    model: str = typer.Argument(help=t("commands.draft-recommend.args.model")),
    max_ratio: float = typer.Option(
        0.25, "--max-ratio", min=0.01, max=1.0, help=t("commands.draft-recommend.options.max_ratio")
    ),
):
    """recommend a draft model for speculative decoding.

    Looks for a smaller sibling of the target on the HF Hub and
    falls back to the canonical small reference for the family
    (Llama-3.2-1B for Llama, Qwen2.5-1.5B for Qwen, ...) when no
    sibling fits the ratio.
    """
    from hfl.hub.draft_picker import pick_draft_for

    pick = pick_draft_for(model, max_ratio=max_ratio)
    if pick is None:
        console.print(f"[yellow]No draft candidate found for[/] {model}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Draft recommendation for[/] {model}")
    console.print(f"  repo:     [cyan]{pick.repo_id}[/]")
    console.print(f"  family:   {pick.family or '-'}")
    if pick.parameter_estimate_b is not None:
        console.print(f"  size:     ~{pick.parameter_estimate_b}B")
    if pick.quantization:
        console.print(f"  quant:    {pick.quantization}")
    console.print(f"  rationale: [dim]{pick.rationale}[/]")


# ----------------------------------------------------------------------
# Saved chat sessions
# ----------------------------------------------------------------------

install_app = typer.Typer(help=t("commands.install.description"), no_args_is_help=True)
app.add_typer(install_app, name="install")


@install_app.command("llama-server", help=t("commands.install.llama_server.description"))
def install_llama_server(
    variant: str | None = typer.Option(
        None, "--variant", help=t("commands.install.llama_server.options.variant")
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help=t("commands.install.llama_server.options.yes")
    ),
    force: bool = typer.Option(
        False, "--force", help=t("commands.install.llama_server.options.force")
    ),
) -> None:
    from hfl.cli.commands.install import install_llama_server as run

    raise typer.Exit(run(variant=variant, assume_yes=yes is True, force=force is True))


sessions_app = typer.Typer(help=t("commands.sessions.description"), no_args_is_help=True)
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list", help=t("commands.sessions.list.description"))
def sessions_list() -> None:
    from rich.table import Table

    from hfl.core.sessions import list_sessions

    saved = list_sessions()
    if not saved:
        console.print(f"[dim]{t('messages.no_sessions')}[/]")
        return

    table = Table(title="Saved Sessions")
    table.add_column("Name", style="cyan")
    table.add_column("Model")
    table.add_column("Messages", justify="right")
    table.add_column("Updated", style="dim")
    for item in saved:
        table.add_row(item.name, item.model, str(len(item.messages)), item.updated_at)
    console.print(table)


@sessions_app.command("show", help=t("commands.sessions.show.description"))
def sessions_show(
    name: str = typer.Argument(help=t("commands.sessions.args.name")),
) -> None:
    from hfl.core.sessions import SessionNotFoundError, load_session

    try:
        item = load_session(name)
    except (SessionNotFoundError, FileNotFoundError) as exc:
        console.print(f"[red]{t('errors.session_not_found', name=name)}[/]")
        raise typer.Exit(1) from exc

    console.print(f"[bold]{item.name}[/]  [dim]{item.model} · {item.updated_at}[/]\n")
    for message in item.messages:
        role = message.get("role", "?")
        colour = {"user": "blue", "assistant": "green", "system": "yellow"}.get(role, "white")
        console.print(f"[{colour}]{role}:[/] {message.get('content', '')}")


@sessions_app.command("rm", help=t("commands.sessions.rm.description"))
def sessions_rm(
    name: str = typer.Argument(help=t("commands.sessions.args.name")),
) -> None:
    from hfl.core.sessions import delete_session

    if not delete_session(name):
        console.print(f"[red]{t('errors.session_not_found', name=name)}[/]")
        raise typer.Exit(1)
    console.print(f"[green]{t('messages.session_deleted', name=name)}[/]")


def cli_main() -> None:
    """The ``hfl`` command. An unreadable setting (``HFL_PORT=abc``) is a
    one-line message naming the variable, not a traceback."""
    from hfl.exceptions import InvalidConfigError
    from hfl.utils.self_exec import CHILD_GUARD_FLAG

    if len(sys.argv) > 1 and sys.argv[1] == CHILD_GUARD_FLAG:
        # An executable starting llama-server through its guard (see
        # hfl.utils.self_exec): run the guard, not the CLI.
        from hfl.engine import _child_guard

        raise SystemExit(_child_guard.main([sys.argv[0], *sys.argv[2:]]))
    from hfl.utils.self_exec import watch_onefile_launcher

    watch_onefile_launcher()
    try:
        app()
    except InvalidConfigError as exc:
        typer.echo(f"hfl: {exc.details}", err=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    cli_main()

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The steps of a pull that ``hfl pull`` and ``/api/pull`` share.

The two had grown apart: the CLI checked the model type, converted to GGUF
and recorded the model type, but logged no provenance; the server logged
provenance but registered no model type and could keep an unsupported
model. Each edge keeps what is its own — the CLI asks about licenses and
prints, the server applies the owner's license policy and streams
progress — and both register through :func:`register_pulled`.

Failures are exceptions with a message key and its values, so each edge
says them its own way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from hfl.converter.formats import ModelType
    from hfl.hub.license_checker import LicenseInfo
    from hfl.hub.resolver import ResolvedModel
    from hfl.models.manifest import ModelManifest

logger = logging.getLogger(__name__)


class PullStepError(Exception):
    """A step that cannot go on: ``key`` names the message (i18n), ``values``
    fill it; ``kept`` says whether the download stays on disk."""

    def __init__(self, key: str, *, kept: bool = True, **values: Any) -> None:
        super().__init__(key)
        self.key, self.kept, self.values = key, kept, values


class Kept(Enum):
    """What was done with a download that is not a GGUF."""

    AS_IS = "as_is"  # not an LLM, or safetensors asked for
    MLX_NATIVE = "mlx_native"  # an MLX quantized repo, served by MLX here
    MLX_ELSEWHERE = "mlx_elsewhere"  # an MLX quantized repo, no MLX here
    FOR_MLX = "for_mlx"  # safetensors kept for the MLX backend
    CONVERTED = "converted"  # converted to GGUF


@dataclass
class Finished:
    """The model as it will be registered."""

    path: Path
    model_type: ModelType
    kept: Kept | None = None
    notes: list[str] = field(default_factory=list)  # message keys, in order


def unsupported_type(resolved: ResolvedModel) -> str | None:
    """The display name of the model type, when HFL cannot serve it (known
    from the Hub before anything is downloaded); else None."""
    from hfl.converter.formats import (
        get_model_type_display_name,
        is_model_type_supported,
        model_type_from_pipeline_tag,
    )

    model_type = model_type_from_pipeline_tag(getattr(resolved, "pipeline_tag", None))
    if model_type is not None and not is_model_type_supported(model_type):
        return str(get_model_type_display_name(model_type))
    return None


def finish_download(
    resolved: ResolvedModel,
    local_path: Path,
    *,
    requested_format: str = "auto",
    quantize: str = "Q4_K_M",
    convert: bool = True,
    on_convert: Callable[[], None] | None = None,
) -> Finished:
    """The downloaded model's type, and — for an LLM that is not a GGUF —
    whether it is kept as it is, served by MLX, or converted to GGUF
    (``convert=False``: kept as it is, as the server does; a request cannot
    wait for a conversion). Raises :class:`PullStepError` for a model type
    HFL cannot serve (the download is removed) or one that cannot convert.
    """
    from hfl.converter.formats import (
        ModelFormat,
        ModelType,
        detect_format,
        detect_model_type,
        get_model_type_display_name,
        is_model_type_supported,
        model_type_from_pipeline_tag,
    )

    model_type = model_type_from_pipeline_tag(getattr(resolved, "pipeline_tag", None))
    if model_type is None:
        model_type = detect_model_type(local_path)
        if model_type != ModelType.LLM and not is_model_type_supported(model_type):
            _remove(local_path)
            raise PullStepError(
                "errors.unsupported_model_type",
                kept=False,
                type=get_model_type_display_name(model_type),
            )
    done = Finished(path=local_path, model_type=model_type)
    if detect_format(local_path) == ModelFormat.GGUF or requested_format == "safetensors":
        return done
    if model_type != ModelType.LLM or not convert:
        done.kept = Kept.AS_IS
        return done
    return _llm_not_gguf(done, resolved, requested_format, quantize, on_convert)


def _llm_not_gguf(
    done: Finished,
    resolved: ResolvedModel,
    requested_format: str,
    quantize: str,
    on_convert: Callable[[], None] | None,
) -> Finished:
    from hfl.converter.formats import is_mlx_quantized_repo
    from hfl.engine.selector import _mlx_preferred

    if is_mlx_quantized_repo(resolved.repo_id, done.path):
        # llama.cpp's converter rejects the packed MLX architecture.
        done.kept = Kept.MLX_NATIVE if _mlx_preferred() else Kept.MLX_ELSEWHERE
        # MLX quantisation pipelines often drop the chat template.
        from hfl.hub.chat_template_repair import ensure_chat_template, has_chat_template

        if not has_chat_template(done.path):
            recovered = ensure_chat_template(done.path, resolved.repo_id)
            done.notes.append("template_recovered" if recovered else "template_missing")
        return done
    if _mlx_preferred() and requested_format != "gguf":
        done.kept = Kept.FOR_MLX  # an explicit --format gguf still converts
        return done
    if on_convert is not None:
        on_convert()  # before, not after: a conversion can take minutes
    done.path = _convert(done.path, resolved.repo_id, quantize)
    done.kept = Kept.CONVERTED
    return done


def _convert(local_path: Path, repo_id: str, quantize: str) -> Path:
    import subprocess

    from hfl.converter.gguf_converter import GGUFConverter, check_model_convertibility
    from hfl.exceptions import ConversionError

    convertible, reason = check_model_convertibility(local_path)
    if not convertible:
        raise PullStepError("errors.cannot_convert_gguf", reason=reason, repo=repo_id)
    output_path = local_path.parent / repo_id.replace("/", "--")
    try:
        return Path(GGUFConverter().convert(local_path, output_path, quantize))
    except (ConversionError, subprocess.CalledProcessError) as exc:
        detail = (exc.details or exc.message) if isinstance(exc, ConversionError) else str(exc)
        raise PullStepError("errors.conversion_failed", reason=detail) from exc


def _remove(path: Path) -> None:
    import shutil

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def register_pulled(
    resolved: ResolvedModel,
    finished: Finished,
    *,
    registry: Any,
    license_info: LicenseInfo | None,
    accepted_at: str | None,
    alias: str | None,
    quantize: str | None,
    source: str,
) -> ModelManifest:
    """Register the model, and log its provenance (legal traceability).

    Its name is the repo's, plus the quantization when it is a GGUF (a
    safetensors repo kept as is carries the level only as a conversion
    target that never ran). A pull that replaces an entry keeps its alias,
    and an alias already naming another model is never taken from it.
    """
    from hfl.converter.formats import ModelFormat, detect_format
    from hfl.models.manifest import ModelManifest

    final = finished.path
    fmt = detect_format(final)
    files = final.rglob("*") if final.is_dir() else [final]
    size = sum(f.stat().st_size for f in files if f.is_file())
    level = getattr(resolved, "quantization", None) or quantize
    quant = level if fmt == ModelFormat.GGUF else None
    name = resolved.repo_id.split("/")[-1].lower() + (f"-{quant.lower()}" if quant else "")

    previous = registry.get(name)
    if not alias and previous is not None and previous.name == name:
        alias = previous.alias  # an update: clients using the alias still find it
    taken = registry.get(alias) if alias else None
    if taken is not None and taken.name != name:
        logger.info("alias %r already names %s; not reassigned", alias, taken.name)
        alias = None

    manifest = ModelManifest(
        name=name,
        repo_id=resolved.repo_id,
        revision=getattr(resolved, "revision", None),
        commit_sha=getattr(resolved, "commit_sha", None),
        alias=alias,
        local_path=str(final),
        format=fmt.value,
        size_bytes=size,
        quantization=quant,
        model_type=finished.model_type.value,
        license=license_info.license_id if license_info else None,
        license_name=license_info.license_name if license_info else None,
        license_url=license_info.url if license_info else None,
        license_restrictions=license_info.restrictions if license_info else [],
        gated=license_info.gated if license_info else False,
        license_accepted_at=accepted_at,
    )
    registry.add(manifest)
    _log_provenance(resolved, manifest, license_info, source)
    return manifest


def _log_provenance(
    resolved: ResolvedModel, manifest: ModelManifest, license_info: Any, source: str
) -> None:
    """Best-effort: a bookkeeping failure never fails a pull that worked."""
    try:
        from hfl.models.provenance import log_conversion

        log_conversion(
            source_repo=resolved.repo_id,
            source_format=manifest.format,
            target_path=manifest.local_path,
            original_license=(license_info.license_id or "") if license_info else "",
            license_accepted=license_info is not None,
            notes=source,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("provenance not recorded for %s: %s", resolved.repo_id, exc)

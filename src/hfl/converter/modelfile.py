# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Render an HFL ``ModelManifest`` back to an Ollama-compatible Modelfile.

``POST /api/show`` returns the Modelfile text alongside structured
metadata, and Ollama clients use that text as a human-editable source
of truth — it's what gets shown when you run ``ollama show --modelfile
<name>``.

HFL stores its models as JSON manifests, not as Modelfiles. To avoid
breaking that shape while still returning something useful on
``/api/show``, we compile the manifest into a *rendered* Modelfile
string. Any Modelfile that HFL eventually writes to disk (R: `P2-1`
create) will round-trip through the same shape.

The output is deterministic: repeated calls with the same manifest
yield byte-identical strings so snapshot tests don't flake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfl.models.manifest import ModelManifest


def _format_stop_value(value: str) -> str:
    """Escape a stop string for the Modelfile PARAMETER line.

    Ollama quotes the value with double quotes and escapes interior
    backslashes + double quotes. Other control chars are passed through
    as-is (mirroring Ollama's parser, which is permissive here).
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_modelfile(manifest: "ModelManifest") -> str:
    '''Render ``manifest`` as a Modelfile string.

    Layout matches Ollama's ``ollama show --modelfile`` output:

        FROM <path-or-digest>
        TEMPLATE """<jinja>"""
        SYSTEM """<text>"""
        PARAMETER <key> <value>
        ...
        LICENSE """<text>"""

    Sections are omitted when the manifest has no data for them. The
    ``FROM`` line always fires — a Modelfile without ``FROM`` is
    invalid per the Ollama spec, so we emit the local path as a last
    resort.
    '''
    lines: list[str] = []

    # ----- FROM (required) -----
    # Prefer a content-addressed form (sha256 digest) when available
    # because it's portable. Fall back to the on-disk path otherwise.
    if manifest.file_hash:
        digest = manifest.file_hash
        if not digest.startswith("sha"):
            digest = f"sha256:{digest}"
        lines.append(f"FROM {digest}")
    else:
        lines.append(f"FROM {manifest.local_path}")

    # ----- TEMPLATE -----
    if manifest.chat_template:
        lines.append("")
        lines.append(f'TEMPLATE """{manifest.chat_template}"""')

    # ----- SYSTEM -----
    # HFL doesn't yet track per-model system prompts (P2-1 will wire
    # this through Modelfile ingestion). Emit only if some future
    # caller has written to a ``system`` attribute.
    system = getattr(manifest, "system", None)
    if system:
        lines.append("")
        lines.append(f'SYSTEM """{system}"""')

    # ----- PARAMETER block -----
    # We emit what the manifest carries today. Callers creating custom
    # Modelfiles (P2-1) will populate more of these via the create
    # flow.
    parameters: list[tuple[str, str]] = []
    if manifest.context_length and manifest.context_length > 0:
        parameters.append(("num_ctx", str(manifest.context_length)))

    # Defaults carried on the manifest (R9 lifted some of these to
    # config; per-model overrides go here).
    defaults = getattr(manifest, "default_parameters", None) or {}
    for key in sorted(defaults.keys()):
        value = defaults[key]
        if key == "stop":
            # ``stop`` is multi-valued in Ollama Modelfiles — one line
            # per stop sequence.
            if isinstance(value, (list, tuple)):
                for s in value:
                    parameters.append(("stop", _format_stop_value(str(s))))
            else:
                parameters.append(("stop", _format_stop_value(str(value))))
        else:
            parameters.append((key, str(value)))

    if parameters:
        lines.append("")
        for key, value in parameters:
            lines.append(f"PARAMETER {key} {value}")

    # ----- ADAPTER (LoRA) -----
    # R P3-2 introduces LoRA support; this block activates then.
    adapters: list[str] = getattr(manifest, "adapter_paths", None) or []
    for adapter in adapters:
        lines.append("")
        lines.append(f"ADAPTER {adapter}")

    # ----- LICENSE -----
    if manifest.license_name or manifest.license:
        lines.append("")
        text = manifest.license_name or manifest.license
        lines.append(f'LICENSE """{text}"""')

    # Final newline so editors don't scream
    return "\n".join(lines) + "\n"

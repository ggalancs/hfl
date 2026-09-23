# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Bounded Hub calls.

Imported by ``hfl/__init__.py``, so it imports nothing beyond ``sys``:
``import hfl`` must not pay for a network it may never use.
"""

from __future__ import annotations

import sys
from types import ModuleType

__all__ = ["install_hub_timeouts", "HUB_CONNECT_TIMEOUT", "HUB_READ_TIMEOUT"]

# huggingface_hub 1.x builds its shared httpx client with ``timeout=None``,
# and most HfApi methods pass ``timeout=None`` explicitly on top — which
# httpx reads as "no limit", not "use the client's". Downloads carry their
# own timeout; API calls (``model_info``, ``list_models``, …) did not, so on
# a network that drops packets instead of refusing them — captive portal,
# dead VPN, a route that went away after DNS answered — every Hub command
# waited forever. Measured: ``hfl search``, ``hfl pull``, /api/pull,
# /api/pull/smart and /api/discover all still hung when a 150 s watchdog
# killed them. Failing DNS (Wi-Fi off) was never the problem; it fails in
# milliseconds.
#
# So the bound goes on the request, in an httpx request hook: any timeout
# field still None when the request leaves gets a value, and anything a
# caller set is kept.
#
# The connect bound is per address. The stdlib tries each address DNS
# returned in turn, and huggingface.co resolves to 4 IPv4 + 8 IPv6
# CloudFront addresses, so a total blackhole costs up to 12 connect
# timeouts. 5 s each: a TCP handshake to a CDN that takes longer is not
# going to carry a model download either. Read is generous because listing
# endpoints can legitimately be slow.
HUB_CONNECT_TIMEOUT = 5.0
HUB_READ_TIMEOUT = 60.0

_HF_HTTP_MODULE = "huggingface_hub.utils._http"


def _fill_unbounded(request: object) -> None:
    """httpx request hook: give every unset timeout field a value."""
    extensions = getattr(request, "extensions", None)
    if not isinstance(extensions, dict):  # pragma: no cover - httpx drift
        return
    current = extensions.get("timeout")
    if not isinstance(current, dict) or None not in current.values():
        return
    bounded = dict(current)
    for key, value in current.items():
        if value is None:
            bounded[key] = HUB_CONNECT_TIMEOUT if key == "connect" else HUB_READ_TIMEOUT
    extensions["timeout"] = bounded


def _bounded_client_factory() -> object:
    from huggingface_hub.utils._http import default_client_factory

    client = default_client_factory()
    hooks = client.event_hooks
    hooks["request"] = [*hooks.get("request", []), _fill_unbounded]
    client.event_hooks = hooks
    return client


def _bound(module: ModuleType) -> None:
    factory = getattr(module, "_GLOBAL_CLIENT_FACTORY", None)
    default = getattr(module, "default_client_factory", None)
    if factory is not None and factory is not default:
        # Someone chose their own client (proxy, certificates). Theirs wins.
        return
    setter = getattr(module, "set_client_factory", None)
    if setter is None or default is None:  # pragma: no cover - API drift
        import logging

        logging.getLogger(__name__).warning(
            "huggingface_hub has no set_client_factory; Hub calls stay unbounded"
        )
        return
    setter(_bounded_client_factory)


class _BoundOnImport:
    """Bounds the client when huggingface_hub loads its HTTP module.

    Installing the factory means importing httpx and huggingface_hub's HTTP
    layer — ~85 ms, which ``hfl version`` or ``hfl list`` should not pay
    for a network they will never use. So nothing is imported here: the
    finder waits for the first import of that module and bounds it then.
    """

    def find_spec(self, fullname: str, path: object, target: object = None) -> object:
        if fullname != _HF_HTTP_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = getattr(finder, "find_spec")(fullname, path, target)
            if spec is None:
                continue
            loader = getattr(spec, "loader", None)
            real_exec = getattr(loader, "exec_module", None)
            if real_exec is None:  # pragma: no cover - legacy loader
                return spec

            def exec_module(module: ModuleType) -> None:
                real_exec(module)
                _bound(module)

            setattr(loader, "exec_module", exec_module)
            return spec
        return None


def install_hub_timeouts() -> None:
    """Make every Hub call HFL (or a library it loads) makes time out.

    Idempotent. Applies at once if huggingface_hub's HTTP layer is already
    loaded, otherwise the moment it is.
    """
    loaded = sys.modules.get(_HF_HTTP_MODULE)
    if loaded is not None:
        _bound(loaded)
        return
    if not any(isinstance(f, _BoundOnImport) for f in sys.meta_path):
        # Duck-typed finder: annotating it for MetaPathFinderProtocol would
        # import ``typing``/``importlib.machinery`` into every ``import hfl``.
        sys.meta_path.insert(0, _BoundOnImport())  # type: ignore[arg-type]

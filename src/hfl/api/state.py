# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Thread-safe server state management.

Provides atomic access to shared server state using asyncio.Lock
for safe concurrent access in async context.

Features:
- As many LLMs resident as memory allows (``hfl.engine.residency``)
- Per-request engine binding, so a request always talks to its own model
- Leases: a model a request is using is never unloaded under it
- Per-model locks for load coalescing; one admission at a time
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime, timedelta  # noqa: F401 — used in annotations

    from hfl.engine.base import AudioEngine, InferenceEngine
    from hfl.engine.dispatcher import InferenceDispatcher
    from hfl.engine.residency import AdmissionPlan, ResidentView
    from hfl.models.manifest import ModelManifest


@dataclass
class ResidentModel:
    """One loaded LLM."""

    name: str
    engine: "InferenceEngine"
    manifest: "ModelManifest"
    footprint: int
    """Estimated resident bytes (``hfl.engine.footprint``); 0 = unknown."""
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)


# The model a request loaded. ``ServerState.engine`` / ``current_model`` read
# it first, so every route that does ``await load_llm(name)`` and then uses
# ``state.engine`` gets the model it asked for — with several models
# resident, "the current engine" would otherwise be whichever request loaded
# last, and two concurrent requests could answer from each other's model
# without any error. Keyed by the state instance so a stale binding from a
# reset state is ignored.
_BOUND: ContextVar["tuple[int, ResidentModel] | None"] = ContextVar("hfl_bound", default=None)

# Engines leased by the current HTTP request (opened by the lease middleware
# in ``hfl.api.server``). ``None`` outside a request — the CLI, the tray, a
# unit test — where there is no concurrent unloader to protect against.
_LEASES: ContextVar["list[InferenceEngine] | None"] = ContextVar("hfl_leases", default=None)


def open_lease_scope() -> "Token[list[InferenceEngine] | None]":
    """Start collecting the leases of one request. Paired with
    :func:`close_lease_scope`."""
    return _LEASES.set([])


def close_lease_scope(token: "Token[list[InferenceEngine] | None]") -> None:
    """Release every lease the request took. Synchronous on purpose: it runs
    in a ``finally`` that a client disconnect may be cancelling, and a lease
    that survived its request would pin a model forever."""
    leases = _LEASES.get()
    _LEASES.reset(token)
    if not leases:
        return
    from hfl.core.container import get_container

    state = get_container().state.get()
    for engine in leases:
        state._release_lease(engine)


@dataclass
class ServerState:
    """Thread-safe server state container.

    Uses asyncio.Lock for safe concurrent access to mutable state.
    All state modifications should be done through the provided methods.

    Features:
    - Several LLMs resident at once, admitted by memory budget
    - Per-model locks prevent concurrent loads of the same model
    - Tracks loading state for API health checks
    """

    # LLM state. ``_engine`` / ``_current_model`` point at the most recently
    # used resident — what a caller outside any request (health probe, tray,
    # CLI) means by "the model". Requests read their own binding instead.
    _engine: InferenceEngine | None = None
    _current_model: ModelManifest | None = None
    _residents: dict[str, ResidentModel] = field(default_factory=dict)

    # TTS state
    _tts_engine: AudioEngine | None = None
    _current_tts_model: ModelManifest | None = None

    # Security
    _api_key: str | None = None

    # Context size override (0 = use model default)
    context_size_override: int = 0

    # Locks for thread-safe access
    _llm_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _tts_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Per-model locks to serialize loading of the same model
    _model_locks: dict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))

    # One admission at a time: two loads deciding in parallel would both see
    # the same free memory and together overcommit it.
    _admission_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Track which models are currently loading
    _loading_models: set[str] = field(default_factory=set)

    # Leases. Every request that loads a model leases its engine until the
    # request ends (the WebSocket turn, which outlives HTTP semantics, pins
    # explicitly). A leased engine is never unloaded: eviction skips it, and
    # an explicit unload (``hfl stop``, shutdown) defers until the last lease
    # is released — no use-after-free of a non-reentrant model.
    _engine_inuse: dict[int, int] = field(default_factory=dict)
    _engine_retired: dict[int, "InferenceEngine"] = field(default_factory=dict)
    _engine_ref_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _lease_released: asyncio.Event = field(default_factory=asyncio.Event)

    # keep_alive: the duration a model stays loaded after its last use. A
    # value a request set explicitly is remembered per model (None = never
    # expire); otherwise HFL_KEEP_ALIVE / OLLAMA_KEEP_ALIVE applies. The
    # deadline is renewed every time the model is used and when its last
    # request ends — a model in continuous use never expires between turns.
    _keep_alive_durations: dict[str, "timedelta | None"] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Request-facing view
    # ------------------------------------------------------------------

    def _bound(self) -> ResidentModel | None:
        bound = _BOUND.get()
        if bound is None or bound[0] != id(self):
            return None
        return bound[1]

    @property
    def engine(self) -> InferenceEngine | None:
        """The LLM engine for this request, else the most recently used one."""
        bound = self._bound()
        if bound is not None:
            return bound.engine
        return self._engine

    @engine.setter
    def engine(self, value: InferenceEngine | None) -> None:
        """Set the LLM pointer directly (tests, CLI and tray preloads).

        Kept in step with the resident set, so what is assigned here is
        what ``/api/ps`` lists and what eviction accounts for.
        """
        old = self._engine
        self._engine = value
        if value is None and old is not None:
            for name, resident in list(self._residents.items()):
                if resident.engine is old:
                    del self._residents[name]
        self._sync_pointer()

    @property
    def current_model(self) -> ModelManifest | None:
        """The manifest for this request, else the most recently used one."""
        bound = self._bound()
        if bound is not None:
            return bound.manifest
        return self._current_model

    @current_model.setter
    def current_model(self, value: ModelManifest | None) -> None:
        """Set current model (for testing and preloads)."""
        old = self._current_model
        self._current_model = value
        if value is None and old is not None:
            resident = self._residents.get(old.name)
            if resident is not None and resident.engine is self._engine:
                del self._residents[old.name]
        self._sync_pointer()

    def _sync_pointer(self) -> None:
        """Register the pointer pair as a resident when both halves are set."""
        engine, manifest = self._engine, self._current_model
        if engine is None or manifest is None:
            return
        name = manifest.name
        for other, resident in list(self._residents.items()):
            if resident.engine is engine and other != name:
                del self._residents[other]
        existing = self._residents.get(name)
        if existing is not None and existing.engine is engine:
            existing.manifest = manifest
            return
        self._residents[name] = ResidentModel(
            name, engine, manifest, _measure_safely(manifest, engine)
        )
        # A preloaded model follows keep_alive from load time; otherwise one
        # nobody uses would never expire.
        self.refresh_keep_alive(name)

    @property
    def tts_engine(self) -> AudioEngine | None:
        """Get current TTS engine."""
        return self._tts_engine

    @tts_engine.setter
    def tts_engine(self, value: AudioEngine | None) -> None:
        """Set TTS engine (for testing)."""
        self._tts_engine = value

    @property
    def current_tts_model(self) -> ModelManifest | None:
        """Get current TTS model manifest."""
        return self._current_tts_model

    @current_tts_model.setter
    def current_tts_model(self, value: ModelManifest | None) -> None:
        """Set current TTS model (for testing)."""
        self._current_tts_model = value

    @property
    def api_key(self) -> str | None:
        """Get API key."""
        return self._api_key

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        """Set API key (thread-safe for simple assignment)."""
        self._api_key = value

    # ------------------------------------------------------------------
    # Resident set
    # ------------------------------------------------------------------

    def resident_models(self) -> list[ResidentModel]:
        """Loaded LLMs, most recently used first."""
        return sorted(self._residents.values(), key=lambda r: r.last_used, reverse=True)

    def resident(self, name: str) -> ResidentModel | None:
        return self._residents.get(name)

    def bind_request(self, name: str) -> ResidentModel | None:
        """Make ``name`` this request's model and lease it.

        Called by ``load_llm`` once the model is resident. Returns the
        resident, or None when ``name`` is not loaded (the caller then
        keeps the pointer semantics).
        """
        resident = self._residents.get(name)
        if resident is None:
            self._sync_pointer()
            resident = self._residents.get(name)
            if resident is None:
                return None
        resident.last_used = time.monotonic()
        self.refresh_keep_alive(name)
        self._engine, self._current_model = resident.engine, resident.manifest
        _BOUND.set((id(self), resident))
        leases = _LEASES.get()
        if leases is not None:
            leases.append(resident.engine)
            key = id(resident.engine)
            self._engine_inuse[key] = self._engine_inuse.get(key, 0) + 1
        return resident

    def _release_lease(self, engine: "InferenceEngine") -> None:
        """Drop one lease. Synchronous: no await between read and write, so
        it is atomic on the event loop and safe inside a cancelled finally."""
        key = id(engine)
        remaining = self._engine_inuse.get(key, 0) - 1
        if remaining > 0:
            self._engine_inuse[key] = remaining
        else:
            self._engine_inuse.pop(key, None)
            retired = self._engine_retired.pop(key, None)
            if retired is not None and retired.is_loaded:
                _spawn_unload(retired)
            for resident in self._residents.values():
                if resident.engine is engine:
                    # The keep_alive clock starts when the model goes idle.
                    resident.last_used = time.monotonic()
                    self.refresh_keep_alive(resident.name)
        self._lease_released.set()

    def _views(self, exclude: str | None = None) -> "list[ResidentView]":
        from hfl.engine.residency import ResidentView

        mine: dict[int, int] = {}
        for engine in _LEASES.get() or []:
            mine[id(engine)] = mine.get(id(engine), 0) + 1
        views = []
        for resident in self._residents.values():
            if resident.name == exclude:
                continue
            key = id(resident.engine)
            held = self._engine_inuse.get(key, 0)
            own = mine.get(key, 0)
            views.append(
                ResidentView(
                    resident.name,
                    resident.footprint,
                    resident.last_used,
                    busy=held - own > 0,
                    mine=own > 0,
                )
            )
        return views

    async def _retire(self, resident: ResidentModel, reason: str) -> None:
        """Remove ``resident`` from the set and unload it once nothing uses it.

        Removed from the set FIRST, so no new request can bind to a model
        that is being unloaded. Then in-flight dispatcher work is drained (a
        path that read the pointer without a lease — the health probe — must
        not be mid-call) and the engine is unloaded now if unleased, or on
        its last lease release. If the unload raises, the set and pointer are
        restored: callers observe "unload failed", not a half-updated state.
        """
        was_member = self._residents.get(resident.name) is resident
        if was_member:
            del self._residents[resident.name]
        pointer = (self._engine, self._current_model)
        if self._engine is resident.engine:
            nxt = self.resident_models()
            self._engine = nxt[0].engine if nxt else None
            self._current_model = nxt[0].manifest if nxt else None
        logger.info(
            "Unloading %s (~%.1f GB): %s",
            resident.name,
            resident.footprint / 1024**3,
            reason,
        )

        async def _unload() -> None:
            async with self._engine_ref_lock:
                pinned = self._engine_inuse.get(id(resident.engine), 0) > 0
                if pinned:
                    self._engine_retired[id(resident.engine)] = resident.engine
            if not pinned and resident.engine.is_loaded:
                await asyncio.to_thread(resident.engine.unload)

        try:
            dispatcher = self._try_get_dispatcher()
            if dispatcher is not None:
                async with dispatcher.exclusive():
                    await _unload()
            else:  # pragma: no cover — dispatcher always present in running app
                await _unload()
        except BaseException:
            if was_member and resident.name not in self._residents:
                self._residents[resident.name] = resident
            if pointer[0] is resident.engine:
                self._engine, self._current_model = pointer
            raise

    async def evict(self, name: str, reason: str = "requested") -> bool:
        """Unload one resident LLM by name. False when it is not loaded."""
        resident = self._residents.get(name)
        if resident is None:
            return False
        async with self._llm_lock:
            if self._residents.get(name) is not resident:
                return False
            await self._retire(resident, reason)
        return True

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    async def _make_room(self, name: str, estimate: int, deadline: float) -> "AdmissionPlan | None":
        """Evict until ``estimate`` bytes fit, waiting for busy models if
        that is what it takes. Raises when it cannot fit."""
        from hfl.engine.residency import (
            MemoryView,
            budget_fraction,
            current_gpu_memory,
            current_memory,
            describe_memory,
            plan_admission,
        )
        from hfl.exceptions import MemoryBudgetExceededError, ModelsBusyError

        max_models = _effective_max_models()
        while True:
            self._lease_released.clear()
            checks = not _memory_checks_disabled() and estimate > 0
            memory = current_memory() if checks else None
            gpu = current_gpu_memory() if checks else None
            views = self._views(exclude=name)
            if memory is None:
                # No measurement (psutil absent, size unknown, or checks
                # disabled): only the optional count ceiling applies.
                # Sizes are zeroed so memory cannot trigger an eviction.
                counted = [replace(v, footprint=0) for v in views]
                plan = plan_admission(0, MemoryView(1, 0, 0), counted, 1.0, max_models)
            else:
                plan = plan_admission(
                    estimate, memory, views, budget_fraction(), max_models, gpu=gpu
                )
                logger.info(
                    "Loading %s (~%.1f GB): %s; after load ~%.1f GB (%.0f%%)%s. Resident: %s",
                    name,
                    estimate / 1024**3,
                    describe_memory(memory, budget_fraction()),
                    plan.used_after / 1024**3,
                    100.0 * plan.used_after / memory.total if memory.total else 0.0,
                    (
                        f"; GPU {describe_memory(gpu, budget_fraction())}, after load "
                        f"~{plan.gpu_used_after / 1024**3:.1f} GB"
                        if gpu is not None
                        else ""
                    ),
                    ", ".join(f"{v.name} ({v.footprint / 1024**3:.1f} GB)" for v in views)
                    or "none",
                )
            if plan.fits:
                for victim in plan.evict:
                    resident = self._residents.get(victim)
                    if resident is not None:
                        idle = time.monotonic() - resident.last_used
                        await self._retire(resident, f"to make room for {name} (idle {idle:.0f}s)")
                return plan
            if plan.reason == "wait":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModelsBusyError(name, list(plan.wait_for))
                logger.info(
                    "%s waits for %s to finish before it can load",
                    name,
                    ", ".join(plan.wait_for),
                )
                try:
                    await asyncio.wait_for(self._lease_released.wait(), min(5.0, remaining))
                except asyncio.TimeoutError:
                    pass
                continue
            raise MemoryBudgetExceededError(
                name,
                needed=estimate,
                plan=plan,
                total=memory.total if memory is not None else 0,
                budget=budget_fraction(),
                gpu_total=gpu.total if gpu is not None else 0,
            )

    async def _reconcile(self, name: str) -> None:
        """After a load, the measured footprint may exceed the estimate
        (llama.cpp can open a larger context than assumed). Unload idle
        models until the budget holds again; never the one just loaded."""
        from hfl.engine.residency import (
            budget_fraction,
            current_gpu_memory,
            current_memory,
            plan_admission,
        )

        if _memory_checks_disabled():
            return
        memory = current_memory()
        if memory is None:
            return
        views = [v for v in self._views() if v.name != name]
        loaded = self._residents.get(name)
        if loaded is None:
            return
        max_models = _effective_max_models()
        # The loaded model is not in ``views`` and counts as the "+1" the
        # planner adds for the incoming model; its memory is in hfl_rss.
        plan = plan_admission(
            0, memory, views, budget_fraction(), max_models, gpu=current_gpu_memory()
        )
        if plan.fits:
            for victim in plan.evict:
                resident = self._residents.get(victim)
                if resident is not None:
                    await self._retire(resident, f"{name} loaded larger than estimated")
        else:
            logger.warning(
                "%s is loaded but memory in use (~%.1f GB) is over the budget; "
                "the models that could be unloaded are in use",
                name,
                plan.used_after / 1024**3,
            )

    # ------------------------------------------------------------------
    # Compatibility setters
    # ------------------------------------------------------------------

    async def set_llm_engine(
        self,
        engine: InferenceEngine | None,
        model: ModelManifest | None,
    ) -> None:
        """Register ``engine`` as a resident LLM, or unload them all.

        ``(None, None)`` unloads every resident LLM (shutdown, ``hfl stop``
        with no model) — each deferred while leased. Otherwise the pair is
        added to the resident set; only a previous engine under the SAME
        name is replaced and unloaded. Other models stay loaded: with
        several resident, a new load is not an eviction.
        """
        async with self._llm_lock:
            if engine is None:
                # A pointer assigned directly (CLI/tray preload) is a resident
                # too; register it so it is unloaded like the others.
                self._sync_pointer()
                for resident in list(self._residents.values()):
                    await self._retire(resident, "unload requested")
                orphan = self._engine
                if orphan is not None:  # pointer without a manifest
                    await self._retire(
                        ResidentModel("<unnamed>", orphan, None, 0),  # type: ignore[arg-type]
                        "unload requested",
                    )
                self._engine = None
                self._current_model = None
                return
            name = model.name if model is not None else getattr(engine, "model_name", "")
            previous = self._residents.get(name)
            prior_pointer = (self._engine, self._current_model)
            if model is not None and (previous is None or previous.engine is not engine):
                # Register the new copy first, so the set never shows a gap.
                self._residents[name] = ResidentModel(
                    name, engine, model, _measure_safely(model, engine)
                )
            self._engine = engine
            self._current_model = model
            if previous is not None and previous.engine is not engine:
                try:
                    await self._retire(previous, "replaced by a reload")
                except BaseException:
                    self._residents[name] = previous
                    self._engine, self._current_model = prior_pointer
                    raise
            self._enforce_engine_concurrency(engine)

    def _enforce_engine_concurrency(self, engine: "InferenceEngine | None") -> None:
        """Clamp dispatcher concurrency to 1 for a non-reentrant backend.

        ``HFL_NUM_PARALLEL`` / ``OLLAMA_NUM_PARALLEL`` exist for drop-in
        parity with Ollama, where each parallel slot is a separate model
        process. HFL runs one in-process model instance, and llama.cpp /
        Transformers keep a single KV cache: two overlapping
        ``create_chat_completion`` calls interleave their state and yield
        corrupted output — not an exception, just silently wrong text.
        Neither engine has an internal lock, so nothing else would catch it.

        Rather than trust the operator to know that, honour the engine's own
        declaration (:attr:`InferenceEngine.supports_concurrent_inference`)
        and say plainly what happened. vLLM, which batches internally, keeps
        whatever the operator configured.
        """
        if engine is None:
            return
        if getattr(engine, "supports_concurrent_inference", False):
            return
        dispatcher = self._try_get_dispatcher()
        if dispatcher is None:
            return
        if dispatcher.clamp_max_inflight(1):
            logger.warning(
                "%s cannot serve concurrent inference (single non-reentrant model "
                "instance with one KV cache); dispatcher concurrency clamped to 1. "
                "HFL_NUM_PARALLEL / OLLAMA_NUM_PARALLEL only takes effect on a "
                "backend that batches internally, such as vLLM.",
                type(engine).__name__,
            )

    @staticmethod
    def _try_get_dispatcher() -> "InferenceDispatcher | None":
        """Best-effort handle to the inference dispatcher (None if unavailable,
        e.g. in a unit test that never built the container)."""
        try:
            from hfl.core import get_dispatcher

            return get_dispatcher()
        except Exception:  # pragma: no cover — defensive
            return None

    async def pin_engine(self, engine: "InferenceEngine | None") -> None:
        """Lease ``engine`` outside the HTTP lease scope (the WebSocket chat
        turn). Paired with :meth:`unpin_engine`; while pinned, the engine is
        never evicted and an explicit unload defers."""
        if engine is None:
            return
        async with self._engine_ref_lock:
            self._engine_inuse[id(engine)] = self._engine_inuse.get(id(engine), 0) + 1

    async def unpin_engine(self, engine: "InferenceEngine | None") -> None:
        """Release a pinned engine. If it was retired while pinned and this
        was its last lease, unload it now (off-loop)."""
        if engine is None:
            return
        to_unload: "InferenceEngine | None" = None
        async with self._engine_ref_lock:
            key = id(engine)
            remaining = self._engine_inuse.get(key, 0) - 1
            if remaining > 0:
                self._engine_inuse[key] = remaining
            else:
                self._engine_inuse.pop(key, None)
                to_unload = self._engine_retired.pop(key, None)
            self._lease_released.set()
        if to_unload is not None and to_unload.is_loaded:
            await asyncio.to_thread(to_unload.unload)

    @asynccontextmanager
    async def with_llm_engine(self) -> AsyncIterator["InferenceEngine"]:
        """Context manager for safe LLM engine access with lock protection.

        Acquires the LLM lock and yields the engine. Ensures the engine
        cannot be unloaded while in use.

        Usage:
            async with state.with_llm_engine() as engine:
                result = engine.generate(...)

        Raises:
            ModelNotLoadedError: If no LLM model is loaded
        """
        from hfl.exceptions import ModelNotLoadedError

        async with self._llm_lock:
            engine = self.engine
            if engine is None:
                raise ModelNotLoadedError()
            yield engine

    def is_llm_loaded(self) -> bool:
        """Check if any LLM engine is loaded."""
        if self._engine is not None and self._engine.is_loaded:
            return True
        return any(r.engine.is_loaded for r in self._residents.values())

    @property
    def is_loading(self) -> bool:
        """Check if any model is currently loading."""
        return len(self._loading_models) > 0

    @property
    def loading_models(self) -> set[str]:
        """Get set of currently loading model names."""
        return self._loading_models.copy()

    async def ensure_llm_loaded(
        self,
        model_name: str,
        loader: Callable[[], Awaitable[tuple["InferenceEngine", "ModelManifest"]]],
        timeout: float = 300.0,
        required_ctx: int = 0,
        estimate: int = 0,
        measure: Callable[["InferenceEngine", "ModelManifest"], int] | None = None,
    ) -> tuple["InferenceEngine", "ModelManifest"]:
        """Ensure ``model_name`` is resident, making room for it by memory.

        If the model is already resident (with a large enough context), it
        is returned at once. Otherwise, one admission at a time: a stale
        copy of the same model is unloaded, idle models are evicted
        least-recently-used first until ``estimate`` bytes fit the memory
        budget, busy ones are waited for, and the load is refused with the
        numbers when it cannot fit. Concurrent requests for the same model
        wait for one load rather than duplicating it.

        Args:
            model_name: Name of the model to load
            loader: Async function that loads the model and returns (engine, manifest)
            timeout: Maximum time to wait for model loading (seconds)
            required_ctx: When > 0, a resident engine only satisfies the
                request if it was opened with at least this context size.
            estimate: Predicted footprint in bytes (0 = unknown).
            measure: Re-measures the footprint once loaded.

        Returns:
            Tuple of (engine, manifest)

        Raises:
            asyncio.TimeoutError: If loading takes longer than timeout
            MemoryBudgetExceededError: The model cannot fit.
            ModelsBusyError: Room exists only by unloading models in use,
                and they did not finish in time.
        """

        def _resident_is_usable() -> ResidentModel | None:
            resident = self._residents.get(model_name)
            if resident is None:
                self._sync_pointer()
                resident = self._residents.get(model_name)
            if resident is None:
                return None
            if required_ctx <= 0:
                return resident
            resident_ctx = getattr(resident.engine, "context_size", 0)
            # ``0`` means the backend doesn't report a context size —
            # never force a reload we can't justify. A window at least as
            # large as the request already satisfies it; only growing it
            # needs a reload (see ``load_llm`` for why shrinking must not).
            if resident_ctx == 0 or resident_ctx >= required_ctx:
                return resident
            return None

        async def _load_with_lock() -> tuple["InferenceEngine", "ModelManifest"]:
            async with self._model_locks[model_name]:
                usable = _resident_is_usable()
                if usable is not None:
                    return usable.engine, usable.manifest

                self._loading_models.add(model_name)
                engine = None
                try:
                    async with self._admission_lock:
                        deadline = time.monotonic() + _busy_wait_seconds()
                        stale = self._residents.get(model_name)
                        if stale is not None:
                            await self._wait_for_release(stale, deadline)
                            await self._retire(stale, "reloading with a larger context")
                        await self._make_room(model_name, estimate, deadline)
                        engine, manifest = await loader()
                        footprint = estimate
                        if measure is not None:
                            try:
                                footprint = measure(engine, manifest) or estimate
                            except Exception:  # pragma: no cover - defensive
                                logger.debug("footprint re-measure failed", exc_info=True)
                        async with self._llm_lock:
                            # May raise; runs before registration so a failure
                            # leaves an unregistered engine the handler frees.
                            self._enforce_engine_concurrency(engine)
                            self._residents[model_name] = ResidentModel(
                                model_name, engine, manifest, footprint
                            )
                            self._engine, self._current_model = engine, manifest
                            self.refresh_keep_alive(model_name)
                        await self._reconcile(model_name)
                    return engine, manifest
                except Exception:
                    # A load that succeeded but could not be registered must
                    # not leak its weights.
                    if (
                        engine is not None
                        and getattr(engine, "is_loaded", False)
                        and not any(r.engine is engine for r in self._residents.values())
                    ):
                        try:
                            await asyncio.to_thread(engine.unload)
                        except Exception:  # pragma: no cover - best-effort cleanup
                            pass
                    raise
                finally:
                    self._loading_models.discard(model_name)

        # Use asyncio.wait_for for cross-platform timeout (works on Python 3.10+)
        return await asyncio.wait_for(_load_with_lock(), timeout=timeout)

    async def _wait_for_release(self, resident: ResidentModel, deadline: float) -> None:
        """Wait until no other request holds ``resident``."""
        from hfl.exceptions import ModelsBusyError

        while True:
            self._lease_released.clear()
            view = next((v for v in self._views() if v.name == resident.name), None)
            if view is None or not view.busy:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelsBusyError(resident.name, [resident.name])
            try:
                await asyncio.wait_for(self._lease_released.wait(), min(5.0, remaining))
            except asyncio.TimeoutError:
                pass

    async def ensure_tts_loaded(
        self,
        model_name: str,
        loader: Callable[[], Awaitable[tuple["AudioEngine", "ModelManifest"]]],
        timeout: float = 300.0,
    ) -> tuple["AudioEngine", "ModelManifest"]:
        """Ensure TTS model is loaded, with serialization per model.

        Similar to ensure_llm_loaded but for TTS models.
        """

        async def _load_with_lock() -> tuple["AudioEngine", "ModelManifest"]:
            async with self._model_locks[f"tts:{model_name}"]:
                # Check state inside lock to prevent race condition
                if (
                    self._current_tts_model
                    and self._current_tts_model.name == model_name
                    and self._tts_engine is not None
                ):
                    return self._tts_engine, self._current_tts_model

                self._loading_models.add(f"tts:{model_name}")
                try:
                    engine, manifest = await loader()
                    await self.set_tts_engine(engine, manifest)
                    return engine, manifest
                finally:
                    self._loading_models.discard(f"tts:{model_name}")

        # Use asyncio.wait_for for cross-platform timeout (works on Python 3.10+)
        return await asyncio.wait_for(_load_with_lock(), timeout=timeout)

    # Thread-safe TTS operations
    async def set_tts_engine(
        self,
        engine: AudioEngine | None,
        model: ModelManifest | None,
    ) -> None:
        """Set TTS engine and model atomically.

        See ``set_llm_engine`` for the rationale behind the off-loop
        unload.

        Args:
            engine: New audio engine (or None to unload)
            model: Model manifest for the loaded model
        """
        async with self._tts_lock:
            # Unload previous engine off-loop; see set_llm_engine.
            if self._tts_engine is not None and self._tts_engine.is_loaded:
                await asyncio.to_thread(self._tts_engine.unload)
            self._tts_engine = engine
            self._current_tts_model = model

    @asynccontextmanager
    async def with_tts_engine(self) -> AsyncIterator["AudioEngine"]:
        """Context manager for safe TTS engine access with lock protection.

        Acquires the TTS lock and yields the engine. Ensures the engine
        cannot be unloaded while in use.

        Usage:
            async with state.with_tts_engine() as engine:
                result = engine.synthesize(...)

        Raises:
            ModelNotLoadedError: If no TTS model is loaded
        """
        from hfl.exceptions import ModelNotLoadedError

        async with self._tts_lock:
            if self._tts_engine is None:
                raise ModelNotLoadedError()
            yield self._tts_engine

    def is_tts_loaded(self) -> bool:
        """Check if TTS engine is loaded."""
        return self._tts_engine is not None and self._tts_engine.is_loaded

    # -- keep_alive ------------------------------------------------------

    def set_keep_alive(self, model_name: str, duration: "timedelta | None") -> None:
        """Remember an explicit keep_alive for ``model_name`` (None = never
        expire) and restart its clock."""
        self._keep_alive_durations[model_name] = duration
        self.refresh_keep_alive(model_name)

    def keep_alive_duration_for(self, model_name: str) -> "timedelta | None":
        """The explicit value if one was set, else the server default.
        None means the model never expires."""
        if model_name in self._keep_alive_durations:
            return self._keep_alive_durations[model_name]
        return _default_keep_alive()

    def refresh_keep_alive(self, model_name: str) -> None:
        """Deadline = now + the model's keep_alive (cleared for "never")."""
        from datetime import datetime, timezone

        duration = self.keep_alive_duration_for(model_name)
        if duration is None:
            self.set_keep_alive_deadline(model_name, None)
        else:
            self.set_keep_alive_deadline(model_name, datetime.now(timezone.utc) + duration)

    async def reap_expired(self, now: "datetime | None" = None) -> list[str]:
        """Unload resident models whose keep_alive deadline has passed and
        that no request holds. Returns the names unloaded."""
        from datetime import datetime, timezone

        now = now or datetime.now(timezone.utc)
        expired = []
        for resident in list(self._residents.values()):
            deadline = self.keep_alive_deadline_for(resident.name)
            if deadline is None or deadline > now:
                continue
            if self._engine_inuse.get(id(resident.engine), 0) > 0:
                continue  # in use: its release renews the deadline
            if await self.evict(resident.name, reason="keep_alive expired"):
                self.set_keep_alive_deadline(resident.name, None)
                expired.append(resident.name)
        return expired

    # -- keep_alive tracking -------------------------------------------
    # Per-model keep-alive deadline, populated by request handlers
    # (R15). ``None`` means "managed by the default idle timeout /
    # never auto-expires". The storage is per-name rather than per
    # engine so it survives model hot-swaps.
    def keep_alive_deadline_for(self, model_name: str) -> "datetime | None":
        """Return the keep-alive deadline for ``model_name`` (or None).

        Consulted by ``/api/ps`` to emit the ``expires_at`` field.
        """
        from datetime import datetime as _dt

        deadlines: dict[str, _dt] = getattr(self, "_keep_alive_deadlines", {})
        return deadlines.get(model_name)

    def set_keep_alive_deadline(self, model_name: str, deadline: "datetime | None") -> None:
        """Set / clear the keep-alive deadline for ``model_name``.

        Called from request handlers when they receive a ``keep_alive``
        value (R15). Passing ``None`` clears the deadline.
        """
        from datetime import datetime as _dt

        deadlines: dict[str, _dt] = self.__dict__.setdefault("_keep_alive_deadlines", {})
        if deadline is None:
            deadlines.pop(model_name, None)
        else:
            deadlines[model_name] = deadline

    # Cleanup
    async def cleanup(self) -> None:
        """Cleanup all engines on shutdown.

        Routes through ``set_llm_engine``/``set_tts_engine`` so every LLM
        unload DRAINS in-flight inference via ``dispatcher.exclusive()`` (and
        defers a still-leased engine) rather than freeing a non-reentrant
        model out from under a request still running during uvicorn's
        graceful-shutdown window. ``unload()`` still runs off-loop so the
        event loop stays alive to finish in-flight responses. (CON)
        """
        await self.set_llm_engine(None, None)
        await self.set_tts_engine(None, None)


def _measure_safely(manifest: "ModelManifest | None", engine: "InferenceEngine") -> int:
    """Footprint of a loaded model, or 0 when it cannot be read."""
    if manifest is None:
        return 0
    try:
        from hfl.engine.footprint import footprint_of_loaded

        path = getattr(manifest, "local_path", None)
        if not isinstance(path, str) or not path:
            return 0
        return footprint_of_loaded(path, engine).total_bytes
    except Exception:
        return 0


def _default_keep_alive() -> "timedelta | None":
    """HFL_KEEP_ALIVE / OLLAMA_KEEP_ALIVE as a duration; None = never.

    An unreadable value falls back to Ollama's 5 minutes rather than to
    "never", which would pin memory forever on a typo."""
    from datetime import timedelta

    from hfl.config import config
    from hfl.utils.duration import InvalidKeepAliveError, is_never_expire, parse_keep_alive

    raw = getattr(config, "keep_alive_default", "5m")
    try:
        delta = parse_keep_alive(raw)
    except InvalidKeepAliveError:
        logger.warning("HFL_KEEP_ALIVE/OLLAMA_KEEP_ALIVE value %r is invalid; using 5m", raw)
        return timedelta(minutes=5)
    if delta is None:
        return timedelta(minutes=5)
    if is_never_expire(delta):
        return None
    return delta


_WARNED_UNMEASURED_GPU = False


def _effective_max_models() -> int:
    """HFL_MAX_LOADED_MODELS, or 1 on a discrete GPU whose memory cannot be
    read (ROCm, CUDA without nvidia-smi).

    Admission by RAM alone would put several models into a VRAM it cannot
    see — on such a host the pre-multi-residency behaviour, one model at a
    time, is the safe one. An explicit HFL_MAX_LOADED_MODELS still wins:
    the operator knows their card.
    """
    global _WARNED_UNMEASURED_GPU
    from hfl.config import config

    configured = int(getattr(config, "max_loaded_models", 0) or 0)
    if configured or _memory_checks_disabled():
        return configured
    from hfl.engine.residency import discrete_gpu_unmeasured

    if not discrete_gpu_unmeasured():
        return 0
    if not _WARNED_UNMEASURED_GPU:
        _WARNED_UNMEASURED_GPU = True
        logger.warning(
            "A GPU is present but its memory cannot be read (no nvidia-smi); keeping one "
            "model loaded at a time. Set HFL_MAX_LOADED_MODELS to allow more."
        )
    return 1


def _memory_checks_disabled() -> bool:
    import os

    return os.environ.get("HFL_DISABLE_MEMORY_PREFLIGHT", "").lower() in ("1", "true", "yes")


def _busy_wait_seconds() -> float:
    """How long a load waits for busy models to free room: the same bound a
    request waits for an inference slot."""
    from hfl.config import config

    try:
        return float(getattr(config, "queue_acquire_timeout_seconds", 60.0))
    except (TypeError, ValueError):
        return 60.0


def _spawn_unload(engine: "InferenceEngine") -> None:
    """Unload a retired engine off-loop from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        engine.unload()
        return
    task = loop.create_task(asyncio.to_thread(engine.unload))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


# Strong references to fire-and-forget unload tasks (asyncio keeps only weak
# ones, so an unreferenced task can be collected mid-flight).
_BACKGROUND: set["asyncio.Task[None]"] = set()


# Singleton access delegated to container for unified management


def get_state() -> ServerState:
    """Get the singleton server state instance.

    Creates the instance on first call (lazy initialization).
    """
    from hfl.core.container import get_state as _get_state

    return _get_state()


def reset_state() -> None:
    """Reset state (for testing purposes)."""
    from hfl.core.container import get_container

    get_container().state.reset()

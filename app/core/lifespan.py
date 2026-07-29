"""Application resource ownership and staged startup readiness."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol

from fastapi import FastAPI

from app.core.config import Settings, read_api_key
from app.core.readiness import CheckStatus, ReadinessTracker
from app.core.types import EngineKind
from app.engine.base import Engine, TranscriptionRequest, TranscriptionResult
from app.observability.metrics import Metrics
from app.vad.base import VADProvider
from app.vad.factory import build_vad_provider


class Scheduler(Protocol):
    @property
    def running(self) -> bool: ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def submit_final(
        self,
        session_id: str,
        request: TranscriptionRequest,
    ) -> TranscriptionResult: ...

    async def submit_partial(
        self,
        session_id: str,
        request: TranscriptionRequest,
    ) -> TranscriptionResult: ...


def _build_engine(settings: Settings) -> Engine:
    if settings.engine is EngineKind.MOCK:
        from app.engine.mock import MockEngine

        return MockEngine()

    if (
        settings.model_dir is None
        or settings.model_manifest is None
        or settings.model_repo_id is None
        or settings.model_revision is None
    ):
        raise RuntimeError("production engine artifacts are not configured")

    if settings.engine is EngineKind.OFFICIAL:
        from app.engine.official_engine import OfficialIndicConformerEngine

        return OfficialIndicConformerEngine(
            model_dir=settings.model_dir,
            manifest_path=settings.model_manifest,
            repo_id=settings.model_repo_id,
            revision=settings.model_revision,
            require_cuda=settings.require_cuda,
        )

    raise RuntimeError(f"unsupported engine: {settings.engine}")


def _build_scheduler(engine: Engine, metrics: Metrics) -> Scheduler:
    # Scheduler is imported only when lifespan starts, keeping app construction cheap.
    from app.engine.scheduler import InferenceScheduler, SchedulerConfig

    return InferenceScheduler(engine, SchedulerConfig(), metrics=metrics)


def build_lifespan(
    settings: Settings,
    *,
    engine: Engine | None = None,
    scheduler: Scheduler | None = None,
    vad_provider: VADProvider | None = None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build a lifespan with optional dependency injection for CPU-safe tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        tracker: ReadinessTracker = app.state.readiness
        active_engine = engine
        active_scheduler = scheduler
        active_vad_provider = vad_provider
        app.state.engine = None
        app.state.scheduler = None
        app.state.vad_provider = None
        app.state.api_key = None

        try:
            if settings.api_key_file is not None:
                app.state.api_key = read_api_key(settings.api_key_file)

            tracker.update(stage="vad_constructing")
            if active_vad_provider is None:
                active_vad_provider = build_vad_provider(settings, app.state.metrics)
            tracker.update(stage="vad_starting")
            await active_vad_provider.startup()
            app.state.vad_provider = active_vad_provider
            app.state.metrics.set_vad_provider(active_vad_provider.name)

            tracker.update(stage="engine_constructing", engine=CheckStatus.STARTING)
            if active_engine is None:
                active_engine = _build_engine(settings)
            app.state.engine = active_engine

            def engine_progress(stage: str) -> None:
                tracker.update(stage=f"engine:{stage}")

            tracker.update(stage="engine_starting")
            await active_engine.startup(engine_progress)
            tracker.update(
                stage="scheduler_constructing",
                engine=CheckStatus.READY,
                scheduler=CheckStatus.STARTING,
            )

            if active_scheduler is None:
                active_scheduler = _build_scheduler(active_engine, app.state.metrics)
            app.state.scheduler = active_scheduler
            await active_scheduler.start()
            if not active_scheduler.running:
                raise RuntimeError("scheduler did not enter the running state")
            tracker.update(
                stage="ready",
                engine=CheckStatus.READY,
                scheduler=CheckStatus.READY,
            )
            yield
        except BaseException as exc:
            tracker.update(stage="failed", detail=type(exc).__name__)
            raise
        finally:
            tracker.update(
                stage="stopping",
                engine=CheckStatus.STOPPING,
                scheduler=CheckStatus.STOPPING,
            )
            try:
                if active_scheduler is not None:
                    await active_scheduler.close()
            finally:
                try:
                    if active_engine is not None:
                        await active_engine.shutdown()
                finally:
                    if active_vad_provider is not None:
                        await active_vad_provider.close()
            tracker.update(
                stage="stopped",
                engine=CheckStatus.STOPPED,
                scheduler=CheckStatus.STOPPED,
            )

    return lifespan

"""Application builders and scheduler doubles for API-level tests.

Everything here is CPU-only. Apps are built through the real ``create_app``
factory with an injected engine or scheduler, so routing, lifespan, and
readiness wiring are exercised exactly as they are in production, while
inference stays deterministic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import anyio
from fastapi import FastAPI

from app.api.websocket import WebSocketConfig, create_websocket_router
from app.core.config import Settings
from app.engine.base import Engine, TranscriptionRequest, TranscriptionResult
from app.engine.mock import MockEngine
from app.engine.scheduler import ServerBusyError
from app.main import create_app
from app.vad.base import VADProvider
from app.vad.energy import EnergyVADProvider

TEST_SETTINGS_DEFAULTS: dict[str, Any] = {"environment": "test", "require_cuda": False}


def settings_for_tests(**overrides: Any) -> Settings:
    """Settings for a mock-engine service, with per-test overrides."""

    return Settings(**{**TEST_SETTINGS_DEFAULTS, **overrides})


def mock_engine_app(
    engine: Engine | None = None,
    *,
    vad_provider: VADProvider | None = None,
    **settings_overrides: Any,
) -> FastAPI:
    """The real application, wired to a deterministic engine."""

    return create_app(
        settings_for_tests(**settings_overrides),
        engine=engine if engine is not None else MockEngine(),
        vad_provider=vad_provider,
    )


def scheduler_app(
    scheduler: Any,
    *,
    vad_provider: VADProvider | None = None,
    **settings_overrides: Any,
) -> FastAPI:
    """The real application, wired to a scheduler double and a mock engine."""

    return create_app(
        settings_for_tests(**settings_overrides),
        engine=MockEngine(),
        scheduler=scheduler,
        vad_provider=vad_provider,
    )


def realtime_only_app(scheduler: Any = None, config: WebSocketConfig | None = None) -> FastAPI:
    """Build the isolated realtime router with an explicit CPU-only VAD."""

    provider = EnergyVADProvider(
        max_streams=128,
        workers=1,
        pending_capacity=128,
        deadline_seconds=0.1,
        metrics=None,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await provider.startup()
            yield
        finally:
            await provider.close()

    application = FastAPI(lifespan=lifespan)
    application.state.vad_provider = provider
    application.include_router(create_websocket_router(scheduler, config))
    return application


@dataclass(slots=True)
class SchedulerDouble:
    """Minimal stand-in satisfying the scheduler protocol used by the API.

    The realtime and REST layers only need ``running``, ``start``, ``close``,
    ``submit_final``, and ``submit_partial``; keeping the double this small means
    a test failure points at the API contract rather than at scheduler internals.
    """

    text: str = "canned transcript"
    language: str | None = None
    decoder: str | None = None
    audio_duration_ms: int | None = None
    final_error: BaseException | None = None
    partial_error: BaseException | None = None
    delay_seconds: float = 0.0
    running: bool = True
    started: int = 0
    closed: int = 0
    finals: list[TranscriptionRequest] = field(default_factory=list)
    partials: list[TranscriptionRequest] = field(default_factory=list)
    texts: Sequence[str] | None = None

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    def _result(self, request: TranscriptionRequest, index: int) -> TranscriptionResult:
        text = self.text if self.texts is None else self.texts[min(index, len(self.texts) - 1)]
        duration = (
            self.audio_duration_ms
            if self.audio_duration_ms is not None
            else round(request.audio.size * 1_000 / request.sample_rate)
        )
        return TranscriptionResult(
            text=text,
            language=self.language if self.language is not None else request.language,
            decoder=self.decoder if self.decoder is not None else request.decoder,
            audio_duration_ms=duration,
            inference_ms=0.0,
        )

    async def submit_final(
        self, session_id: str, request: TranscriptionRequest
    ) -> TranscriptionResult:
        self.finals.append(request)
        if self.delay_seconds:
            await anyio.sleep(self.delay_seconds)
        if self.final_error is not None:
            raise self.final_error
        return self._result(request, len(self.finals) - 1)

    async def submit_partial(
        self, session_id: str, request: TranscriptionRequest
    ) -> TranscriptionResult:
        self.partials.append(request)
        if self.delay_seconds:
            await anyio.sleep(self.delay_seconds)
        if self.partial_error is not None:
            raise self.partial_error
        return self._result(request, len(self.partials) - 1)


def busy_scheduler() -> SchedulerDouble:
    error = ServerBusyError("inference queue is full")
    return SchedulerDouble(final_error=error, partial_error=error)


def failing_scheduler() -> SchedulerDouble:
    error = RuntimeError("engine exploded")
    return SchedulerDouble(final_error=error, partial_error=error)


__all__ = [
    "SchedulerDouble",
    "TEST_SETTINGS_DEFAULTS",
    "busy_scheduler",
    "failing_scheduler",
    "mock_engine_app",
    "realtime_only_app",
    "scheduler_app",
    "settings_for_tests",
]

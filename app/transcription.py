"""Reusable final-first transcription execution and success telemetry."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, cast

import anyio
from fastapi import Request

from app.core.logging import get_logger
from app.core.types import LanguageCode, ProcessingMode
from app.engine.base import Engine, TranscriptionRequest, TranscriptionResult
from app.observability.metrics import Metrics

_LOGGER = get_logger(__name__)


async def run_transcription(
    request: Request,
    request_id: str,
    transcription: TranscriptionRequest,
) -> TranscriptionResult:
    """Submit a final-priority job, falling back to an injected direct engine."""

    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        return cast(TranscriptionResult, await scheduler.submit_final(request_id, transcription))
    engine: Engine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise RuntimeError("engine is unavailable")
    return await anyio.to_thread.run_sync(
        engine.transcribe,
        transcription,
        abandon_on_cancel=True,
    )


def record_success(
    metrics: Metrics,
    request_id: str,
    language: LanguageCode,
    mode: ProcessingMode,
    result: TranscriptionResult,
    elapsed_seconds: float,
    *,
    logger: Any = _LOGGER,
) -> None:
    """Record a completed transcription without allowing telemetry to discard it."""

    audio_seconds = result.audio_duration_ms / 1_000
    try:
        metrics.record_transcription(language, mode)
        metrics.record_audio_seconds(language, mode, audio_seconds)
        metrics.record_queue_wait(mode, max(0.0, elapsed_seconds - result.inference_ms / 1_000))
        metrics.record_final_latency(language, mode, elapsed_seconds)
        if audio_seconds:
            metrics.record_realtime_factor(language, mode, elapsed_seconds / audio_seconds)
    except Exception:
        metrics.record_telemetry_failure()
        with suppress(Exception):
            logger.exception("transcription_metrics_failed", request_id=request_id)

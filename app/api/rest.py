"""Batch transcription REST endpoint."""

import io
import wave
from contextlib import suppress
from time import perf_counter
from typing import Annotated, cast
from uuid import uuid4

import anyio
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.types import Decoder, LanguageCode, ProcessingMode
from app.engine.base import Engine, TranscriptionRequest, TranscriptionResult
from app.observability.metrics import MetricCode, Metrics
from app.schemas.rest import ErrorResponse, TranscriptionResponse

router = APIRouter(prefix="/v1", tags=["transcription"])
_LOGGER = get_logger(__name__)


def _decode_pcm_upload(payload: bytes) -> np.ndarray:
    """Decode either a mono PCM WAV container or headerless pcm_s16le."""
    pcm = payload
    if payload.startswith(b"RIFF"):
        try:
            with wave.open(io.BytesIO(payload), "rb") as source:
                if source.getnchannels() != 1:
                    raise ValueError("audio must be mono")
                if source.getsampwidth() != 2:
                    raise ValueError("audio must be signed 16-bit PCM")
                if source.getframerate() != 16_000:
                    raise ValueError("audio sample rate must be 16000 Hz")
                if source.getcomptype() != "NONE":
                    raise ValueError("compressed WAV audio is not supported")
                pcm = source.readframes(source.getnframes())
        except (EOFError, wave.Error) as exc:
            raise ValueError("invalid WAV container") from exc
    if not pcm:
        raise ValueError("audio is empty")
    if len(pcm) % 2:
        raise ValueError("pcm_s16le audio must contain complete samples")
    samples = np.frombuffer(pcm, dtype="<i2")
    return samples.astype(np.float32) / 32768.0


async def _run_transcription(
    request: Request,
    request_id: str,
    transcription: TranscriptionRequest,
) -> TranscriptionResult:
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        return cast(TranscriptionResult, await scheduler.submit_final(request_id, transcription))
    engine: Engine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise RuntimeError("engine is unavailable")
    return await anyio.to_thread.run_sync(engine.transcribe, transcription, abandon_on_cancel=True)


def _record_success(
    metrics: Metrics,
    request_id: str,
    language: LanguageCode,
    mode: ProcessingMode,
    result: TranscriptionResult,
    elapsed_seconds: float,
) -> None:
    """Record one completed transcription without ever discarding it.

    Inference has already succeeded here, so a failure in this bookkeeping must
    degrade observability rather than turn a finished transcript into a 5xx.
    """

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
        # The recorder above cannot raise; a broken logging pipeline still can.
        with suppress(Exception):
            _LOGGER.exception("transcription_metrics_failed", request_id=request_id)


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
)
async def transcribe(
    request: Request,
    audio: Annotated[UploadFile, File(description="mono 16 kHz PCM WAV or pcm_s16le")],
    language: Annotated[LanguageCode, Form()],
    mode: Annotated[ProcessingMode, Form()] = ProcessingMode.HYBRID,
) -> TranscriptionResponse:
    settings: Settings = request.app.state.settings
    metrics = request.app.state.metrics
    # The decoder follows server policy exactly as in the realtime handshake, so an
    # explicit client choice is refused rather than silently overridden. The form is
    # already parsed and cached for the declared fields above.
    if "decoder" in await request.form():
        metrics.record_rejection(MetricCode.BAD_REQUEST)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "decoder is selected by the server from mode and cannot be requested",
        )
    if not request.app.state.readiness.snapshot().ready:
        metrics.record_rejection(MetricCode.SERVICE_UNAVAILABLE)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "service is not ready")

    payload = await audio.read(settings.max_upload_bytes + 1)
    await audio.close()
    if len(payload) > settings.max_upload_bytes:
        metrics.record_rejection(MetricCode.UPLOAD_TOO_LARGE)
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "audio upload exceeds configured limit",
        )
    try:
        waveform = _decode_pcm_upload(payload)
    except ValueError as exc:
        metrics.record_rejection(MetricCode.INVALID_AUDIO)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if waveform.size / settings.sample_rate > settings.max_audio_seconds:
        metrics.record_rejection(MetricCode.UPLOAD_TOO_LARGE)
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "audio duration exceeds configured limit",
        )

    request_id = str(uuid4())
    final_decoder = Decoder.CTC if mode is ProcessingMode.LATENCY else Decoder.RNNT
    engine_request = TranscriptionRequest(
        audio=waveform,
        sample_rate=settings.sample_rate,
        language=language.value,
        decoder=final_decoder.value,
    )
    started = perf_counter()
    try:
        with anyio.fail_after(settings.request_timeout_seconds):
            result = await _run_transcription(request, request_id, engine_request)
    except TimeoutError as exc:
        metrics.record_error(MetricCode.TIMEOUT)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "transcription timed out") from exc
    except RuntimeError as exc:
        metrics.record_error(MetricCode.INFERENCE_ERROR)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "transcription is unavailable"
        ) from exc

    elapsed_seconds = perf_counter() - started
    response = TranscriptionResponse(
        text=result.text,
        language=LanguageCode(result.language),
        mode=mode,
        decoder=Decoder(result.decoder),
        audio_duration_ms=result.audio_duration_ms,
        inference_ms=result.inference_ms,
        request_id=request_id,
    )
    _record_success(metrics, request_id, language, mode, result, elapsed_seconds)
    return response

"""Realtime raw-PCM websocket protocol."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from app.api.auth import websocket_admitted
from app.api.websocket.state import WebSocketConfig, _LiveSession, _SessionRegistry
from app.audio.endpoint import (
    AdaptivePartialCadence,
    EndpointConfig,
    EndpointDetector,
    EndpointEvent,
    PartialCadenceConfig,
)
from app.audio.pcm import PCM16_FRAME_BYTES, SAMPLE_RATE, PCM16Buffer, PCMBufferOverflow
from app.audio.stable_prefix import RollingStablePrefix
from app.core.logging import get_logger
from app.core.types import Decoder, LanguageCode, ProcessingMode
from app.engine.base import TranscriptionRequest
from app.engine.scheduler import (
    InferenceScheduler,
    PartialOutstandingError,
    ServerBusyError,
    StalePartialError,
)
from app.schemas.protocol import (
    InputCommitEvent,
    ProtocolErrorCode,
    ProtocolErrorEvent,
    SessionReadyEvent,
    SessionStartEvent,
    SpeechStartedEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
)
from app.vad.base import (
    VADCapacityError,
    VADClosedError,
    VADInferenceError,
    VADProvider,
)

_LOGGER = get_logger(__name__)


async def _serve_websocket(
    websocket: WebSocket,
    scheduler: InferenceScheduler | None,
    config: WebSocketConfig,
    registry: _SessionRegistry,
) -> None:
    if not websocket_admitted(websocket):
        metrics = getattr(websocket.app.state, "metrics", None)
        if metrics is not None:
            metrics.record_rejection("INVALID_SESSION")
        await websocket.close(code=1008)
        return
    await websocket.accept()
    if scheduler is None or not scheduler.running:
        await _send_error(
            websocket, "SERVICE_UNAVAILABLE", "transcription service is not ready", retryable=True
        )
        await _close(websocket, 1013)
        return
    if not await registry.acquire():
        await _send_error(
            websocket, "SERVER_BUSY", "maximum concurrent sessions reached", retryable=True
        )
        await _close(websocket, 1013)
        return
    metrics = getattr(websocket.app.state, "metrics", None)
    send_lock = asyncio.Lock()
    session: _LiveSession | None = None
    connected_at = time.monotonic()
    try:
        if metrics is not None:
            metrics.session_started()
        while True:
            elapsed = time.monotonic() - connected_at
            if elapsed >= config.max_session_seconds:
                await _locked_error(
                    websocket, send_lock, "SESSION_LIMIT", "maximum session duration reached"
                )
                await _close(websocket, 1000)
                return
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=min(config.idle_timeout_seconds, config.max_session_seconds - elapsed),
                )
            except TimeoutError:
                await _locked_error(
                    websocket, send_lock, "IDLE_TIMEOUT", "session was idle for too long"
                )
                await _close(websocket, 1001)
                return
            if message["type"] == "websocket.disconnect":
                return

            binary, text = message.get("bytes"), message.get("text")
            if binary is not None:
                if session is None:
                    await _protocol_failure(
                        websocket,
                        send_lock,
                        "SESSION_REQUIRED",
                        "session.start must be the first event",
                    )
                    return
                if len(binary) > config.max_frame_bytes:
                    await _protocol_failure(
                        websocket,
                        send_lock,
                        "FRAME_TOO_LARGE",
                        "binary frame exceeds the configured limit",
                        close_code=1009,
                    )
                    return
                if len(binary) != PCM16_FRAME_BYTES:
                    await _protocol_failure(
                        websocket,
                        send_lock,
                        "INVALID_FRAME_SIZE",
                        f"binary frames must contain exactly {PCM16_FRAME_BYTES} bytes",
                    )
                    return
                if not await _handle_audio_frame(websocket, send_lock, scheduler, session, binary):
                    return
                continue
            if text is None:
                await _protocol_failure(
                    websocket,
                    send_lock,
                    "MALFORMED_EVENT",
                    "websocket event must contain text or binary data",
                )
                return
            payload = _parse_json_object(text)
            if payload is None:
                await _protocol_failure(
                    websocket,
                    send_lock,
                    "MALFORMED_EVENT",
                    "text events must be valid JSON objects",
                )
                return

            if session is None:
                if payload.get("type") != "session.start":
                    await _protocol_failure(
                        websocket,
                        send_lock,
                        "SESSION_REQUIRED",
                        "session.start must be the first event",
                    )
                    return
                try:
                    start = SessionStartEvent.model_validate_json(text)
                except ValidationError:
                    await _protocol_failure(
                        websocket,
                        send_lock,
                        "INVALID_SESSION",
                        "session.start contains invalid or unsupported fields",
                    )
                    return
                provider = getattr(websocket.app.state, "vad_provider", None) if start.vad else None
                try:
                    session = _new_session(start, config, provider)
                except VADCapacityError:
                    await _locked_error(
                        websocket,
                        send_lock,
                        "SERVER_BUSY",
                        "voice activity detection capacity is exhausted",
                        retryable=True,
                    )
                    await _close(websocket, 1013)
                    return
                except (VADInferenceError, VADClosedError):
                    await _locked_error(
                        websocket,
                        send_lock,
                        "INFERENCE_ERROR",
                        "voice activity detection is unavailable",
                        retryable=True,
                    )
                    await _close(websocket, 1011)
                    return
                async with send_lock:
                    await websocket.send_json(
                        SessionReadyEvent(session_id=session.session_id).model_dump(mode="json")
                    )
                continue
            if payload.get("type") == "session.start":
                await _protocol_failure(
                    websocket,
                    send_lock,
                    "SESSION_ALREADY_STARTED",
                    "session.start can only be sent once",
                )
                return
            try:
                InputCommitEvent.model_validate_json(text)
            except ValidationError:
                await _protocol_failure(
                    websocket,
                    send_lock,
                    "UNKNOWN_EVENT",
                    "only input.commit is valid after session.start",
                )
                return
            if session.buffer.empty:
                await _locked_error(
                    websocket, send_lock, "EMPTY_UTTERANCE", "input.commit requires buffered audio"
                )
                continue
            session.endpoint.commit()
            if not await _finalize(websocket, send_lock, scheduler, session):
                return
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        raise
    except Exception:
        if websocket.client_state is not WebSocketState.DISCONNECTED:
            await _locked_error(
                websocket, send_lock, "INTERNAL_ERROR", "realtime session failed", retryable=True
            )
            await _close(websocket, 1011)
    finally:
        if session is not None:
            if session.partial_task is not None:
                session.partial_task.cancel()
            if session.vad is not None:
                try:
                    session.vad.close()
                except Exception:
                    _LOGGER.exception("realtime_vad_close_failed", session_id=session.session_id)
        # Release the bounded slot before optional bookkeeping: a metrics backend that
        # raises must not permanently shrink session capacity.
        await registry.release()
        if metrics is not None:
            metrics.session_ended()


def _record_vad_metric(websocket: WebSocket, method: str, *args: object) -> None:
    metrics = getattr(websocket.app.state, "metrics", None)
    if metrics is None:
        return
    try:
        getattr(metrics, method)(*args)
    except Exception:
        _LOGGER.exception("realtime_vad_metrics_failed", metric=method)
        try:
            metrics.record_telemetry_failure()
        except Exception:
            _LOGGER.exception("realtime_metrics_failure_counter_failed")


def _new_session(
    start: SessionStartEvent,
    config: WebSocketConfig,
    provider: VADProvider | None,
) -> _LiveSession:
    partial_decoder, final_decoder = {
        ProcessingMode.LATENCY: (Decoder.CTC, Decoder.CTC),
        ProcessingMode.HYBRID: (Decoder.CTC, Decoder.RNNT),
        ProcessingMode.ACCURACY: (Decoder.CTC, Decoder.RNNT),
    }[start.mode]
    cadence_ms = {
        ProcessingMode.LATENCY: config.partial_latency_ms,
        ProcessingMode.HYBRID: config.partial_hybrid_ms,
        ProcessingMode.ACCURACY: config.partial_accuracy_ms,
    }[start.mode]
    vad = None
    vad_provider_name = None
    vad_threshold = None
    if start.vad:
        if provider is None:
            raise VADClosedError("VAD provider is unavailable")
        vad = provider.new_stream(16_000)
        vad_provider_name = provider.name
        vad_threshold = (
            provider.default_threshold if config.vad_threshold is None else config.vad_threshold
        )
    return _LiveSession(
        session_id=uuid4().hex,
        start=start,
        partial_decoder=partial_decoder,
        final_decoder=final_decoder,
        buffer=PCM16Buffer(config.max_utterance_ms),
        endpoint=EndpointDetector(
            EndpointConfig(
                speech_start_ms=config.speech_start_ms,
                speech_end_ms=config.speech_end_ms,
                max_utterance_ms=config.max_utterance_ms,
            )
        ),
        vad=vad,
        vad_provider_name=vad_provider_name,
        vad_threshold=vad_threshold,
        cadence=AdaptivePartialCadence(
            PartialCadenceConfig(
                initial_ms=cadence_ms,
                minimum_ms=config.partial_minimum_ms,
                maximum_ms=config.partial_maximum_ms,
            )
        ),
        stable_prefix=RollingStablePrefix(config.partial_history),
    )


async def _handle_audio_frame(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    scheduler: InferenceScheduler,
    session: _LiveSession,
    frame: bytes,
) -> bool:
    if session.start.vad:
        vad = session.vad
        threshold = session.vad_threshold
        if vad is None or threshold is None:
            raise VADClosedError("VAD stream is unavailable")
        try:
            score = await vad.score(frame)
        except VADCapacityError:
            await _locked_error(
                websocket,
                send_lock,
                "SERVER_BUSY",
                "voice activity detection capacity is exhausted",
                retryable=True,
            )
            await _close(websocket, 1013)
            return False
        except (VADInferenceError, VADClosedError):
            await _locked_error(
                websocket,
                send_lock,
                "INFERENCE_ERROR",
                "voice activity detection failed",
                retryable=True,
            )
            await _close(websocket, 1011)
            return False
        was_active = session.endpoint.active
        is_speech = score >= threshold
        if session.vad_provider_name is not None:
            _record_vad_metric(
                websocket, "record_vad_decision", session.vad_provider_name, is_speech
            )
        event = session.endpoint.process(is_speech)
        if event is not EndpointEvent.NONE:
            _record_vad_metric(websocket, "record_vad_endpoint_event", "native")
        keep_frame = was_active or is_speech
    else:
        event, keep_frame = EndpointEvent.NONE, True
    if keep_frame:
        try:
            session.buffer.append(frame)
        except PCMBufferOverflow:
            await _locked_error(
                websocket,
                send_lock,
                "UTTERANCE_TOO_LONG",
                "utterance exceeded the configured audio limit",
            )
            await _close(websocket, 1009)
            return False
    elif not session.endpoint.active:
        session.buffer.clear()

    if event is EndpointEvent.SPEECH_STARTED:
        async with send_lock:
            await websocket.send_json(SpeechStartedEvent().model_dump(mode="json"))
    if event in (EndpointEvent.UTTERANCE_ENDED, EndpointEvent.UTTERANCE_LIMIT):
        return await _finalize(websocket, send_lock, scheduler, session)
    if (
        not session.start.vad
        and session.buffer.duration_ms >= session.endpoint.config.max_utterance_ms
    ):
        return await _finalize(websocket, send_lock, scheduler, session)

    partial_done = session.partial_task is None or session.partial_task.done()
    partial_active = session.endpoint.active if session.start.vad else not session.buffer.empty
    if partial_active and partial_done and session.cadence.due(session.buffer.duration_ms):
        audio_ms = session.buffer.duration_ms
        request = TranscriptionRequest(
            audio=session.buffer.to_float32(),
            sample_rate=SAMPLE_RATE,
            language=session.start.language.value,
            decoder=session.partial_decoder.value,
        )
        session.cadence.mark_submitted(audio_ms)
        session.partial_task = asyncio.create_task(
            _emit_partial(
                websocket, send_lock, scheduler, session, request, session.epoch, audio_ms
            ),
            name=f"asr-partial-{session.session_id}",
        )
    return True


async def _emit_partial(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    scheduler: InferenceScheduler,
    session: _LiveSession,
    request: TranscriptionRequest,
    epoch: int,
    audio_ms: int,
) -> None:
    started = time.perf_counter()
    try:
        result = await scheduler.submit_partial(session.session_id, request)
    except (StalePartialError, PartialOutstandingError, asyncio.CancelledError):
        return
    except ServerBusyError:
        # Partials are best effort: surface the pressure, slow the cadence and keep the
        # session usable so the utterance can still reach a final.
        session.cadence.observe(False, audio_ms)
        await _locked_error(
            websocket, send_lock, "SERVER_BUSY", "inference queue is full", retryable=True
        )
        return
    except Exception:
        session.cadence.observe(False, audio_ms)
        await _locked_error(
            websocket, send_lock, "INFERENCE_ERROR", "partial transcription failed", retryable=True
        )
        return
    elapsed_seconds = time.perf_counter() - started
    try:
        if epoch == session.epoch:
            changed = result.text != session.last_partial
            session.last_partial = result.text
            session.cadence.observe(changed, audio_ms)
            session.revision += 1
            stable = session.stable_prefix.add(result.text)
            event = TranscriptPartialEvent(
                text=result.text,
                revision=session.revision,
                is_stable=bool(result.text) and stable == result.text,
            )
            async with send_lock:
                if epoch == session.epoch:
                    try:
                        await websocket.send_json(event.model_dump(mode="json"))
                    except (RuntimeError, WebSocketDisconnect):
                        # The peer is gone. Teardown belongs to the receive loop, and this
                        # task is detached, so a lost partial must not raise into the loop.
                        pass
    finally:
        # Delivery is attempted before any metrics call so a metrics defect cannot cost a
        # partial, and the latency is recorded even when the partial was preempted or the
        # send failed: the inference work happened either way.
        metrics = getattr(websocket.app.state, "metrics", None)
        if metrics is not None:
            mode = session.start.mode
            queue_wait = max(0.0, elapsed_seconds - result.inference_ms / 1_000)
            try:
                metrics.record_queue_wait(mode, queue_wait)
                metrics.record_partial_latency(session.start.language, mode, elapsed_seconds)
            except Exception:
                metrics.record_telemetry_failure()
                _LOGGER.exception("realtime_metrics_failed", session_id=session.session_id)


async def _finalize(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    scheduler: InferenceScheduler,
    session: _LiveSession,
) -> bool:
    endpoint_at = time.perf_counter()
    session.epoch += 1
    request = TranscriptionRequest(
        audio=session.buffer.to_float32(),
        sample_rate=SAMPLE_RATE,
        language=session.start.language.value,
        decoder=session.final_decoder.value,
    )
    session.buffer.clear()
    session.endpoint.reset()
    if session.vad is not None:
        session.vad.reset()
    try:
        result = await scheduler.submit_final(session.session_id, request)
    except ServerBusyError:
        await _locked_error(
            websocket, send_lock, "SERVER_BUSY", "inference queue is full", retryable=True
        )
        await _close(websocket, 1013)
        return False
    except Exception:
        await _locked_error(
            websocket, send_lock, "INFERENCE_ERROR", "final transcription failed", retryable=True
        )
        await _close(websocket, 1011)
        return False
    elapsed_seconds = time.perf_counter() - endpoint_at
    event = TranscriptFinalEvent(
        text=result.text,
        language=LanguageCode(result.language),
        decoder=Decoder(result.decoder),
        audio_duration_ms=result.audio_duration_ms,
        endpoint_to_final_ms=elapsed_seconds * 1_000.0,
    )
    try:
        try:
            async with send_lock:
                await websocket.send_json(event.model_dump(mode="json"))
        except (RuntimeError, WebSocketDisconnect):
            # The peer left after inference finished. The receive loop observes the
            # disconnect on its next read, so raising here would only relabel a client
            # departure as a server fault.
            pass
        session.cadence.reset()
        session.stable_prefix.reset()
        session.last_partial = ""
        session.partial_task = None
    finally:
        # The transcript reaches the wire before any metrics call, so a metrics defect
        # cannot swallow a completed result; accounting still runs when the client
        # vanishes mid-send, or the counters describe only deliveries that succeeded.
        metrics = getattr(websocket.app.state, "metrics", None)
        if metrics is not None:
            audio_seconds = result.audio_duration_ms / 1_000
            mode = session.start.mode
            language = session.start.language
            queue_wait = max(0.0, elapsed_seconds - result.inference_ms / 1_000)
            try:
                metrics.record_transcription(language, mode)
                metrics.record_audio_seconds(language, mode, audio_seconds)
                metrics.record_queue_wait(mode, queue_wait)
                metrics.record_final_latency(language, mode, elapsed_seconds)
                if audio_seconds > 0:
                    metrics.record_realtime_factor(language, mode, elapsed_seconds / audio_seconds)
            except Exception:
                metrics.record_telemetry_failure()
                _LOGGER.exception("realtime_metrics_failed", session_id=session.session_id)
    return True


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


async def _protocol_failure(
    websocket: WebSocket,
    lock: asyncio.Lock,
    code: ProtocolErrorCode,
    message: str,
    *,
    close_code: int = 1002,
) -> None:
    await _locked_error(websocket, lock, code, message)
    await _close(websocket, close_code)


async def _locked_error(
    websocket: WebSocket,
    lock: asyncio.Lock,
    code: ProtocolErrorCode,
    message: str,
    retryable: bool = False,
) -> None:
    async with lock:
        await _send_error(websocket, code, message, retryable=retryable)


async def _send_error(
    websocket: WebSocket,
    code: ProtocolErrorCode,
    message: str,
    *,
    retryable: bool = False,
) -> None:
    metrics = getattr(websocket.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_protocol_failure(code)
    if websocket.application_state is WebSocketState.CONNECTED:
        await websocket.send_json(
            ProtocolErrorEvent(code=code, message=message, retryable=retryable).model_dump(
                mode="json"
            )
        )


async def _close(websocket: WebSocket, code: int) -> None:
    if websocket.application_state is WebSocketState.CONNECTED:
        try:
            await websocket.close(code=code)
        except (RuntimeError, WebSocketDisconnect):
            pass

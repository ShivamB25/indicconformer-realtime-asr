"""Connection lifecycle and event dispatch for OpenAI realtime transcription."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from dataclasses import dataclass
from typing import Any, Literal, NoReturn
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketState

from app.api.auth import websocket_admitted
from app.api.realtime import ConnectionRegistry
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.types import ProcessingMode
from app.engine.base import TranscriptionRequest
from app.engine.scheduler import ServerBusyError
from app.openai_compat import MODEL_ID, OpenAIError, validate_model
from app.openai_compat.realtime import (
    CLIENT_EVENT_ADAPTER,
    AudioAppendEvent,
    AudioClearedEvent,
    AudioClearEvent,
    AudioCommitEvent,
    AudioCommittedEvent,
    DurationUsage,
    ErrorDetail,
    ErrorEvent,
    RealtimeSessionState,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionDeltaEvent,
    TranscriptionFailedEvent,
    TurnSnapshot,
)
from app.vad.base import VADCapacityError, VADError, VADInferenceError, VADProvider, VADStream

MAX_EVENT = 15 * 1024 * 1024
_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeConfig:
    max_event_bytes: int = MAX_EVENT
    max_session_audio_bytes: int = 50 * 1024 * 1024
    max_session_seconds: float = 3600.0
    idle_timeout_seconds: float = 60.0
    max_pending_turns: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.max_event_bytes <= MAX_EVENT:
            raise ValueError("event limit must be 1 byte to 15 MiB")
        if self.max_session_audio_bytes < 2:
            raise ValueError("audio limit is too small")
        if self.max_session_seconds <= 0 or self.idle_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_pending_turns <= 0:
            raise ValueError("max_pending_turns must be positive")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


async def _send(ws: WebSocket, lock: asyncio.Lock, event: BaseModel) -> None:
    async with lock:
        await ws.send_json(event.model_dump(mode="json"))


async def _error(
    ws: WebSocket,
    lock: asyncio.Lock,
    code: str,
    message: str,
    causal: str | None,
    param: str | None = None,
    kind: Literal["invalid_request_error", "server_error"] = "invalid_request_error",
) -> None:
    metric_code = {
        "audio_limit_exceeded": "INVALID_AUDIO",
        "event_too_large": "FRAME_TOO_LARGE",
        "idle_timeout": "IDLE_TIMEOUT",
        "internal_error": "INTERNAL_ERROR",
        "server_busy": "SERVER_BUSY",
        "service_unavailable": "SERVICE_UNAVAILABLE",
        "session_expired": "SESSION_LIMIT",
        "transcription_failed": "INFERENCE_ERROR",
        "vad_capacity_exceeded": "SERVER_BUSY",
        "vad_inference_failed": "INFERENCE_ERROR",
    }.get(code, "BAD_REQUEST")
    _record_metric(ws, "record_protocol_failure", metric_code)
    await _send(
        ws,
        lock,
        ErrorEvent(
            event_id=_id("event"),
            error=ErrorDetail(type=kind, code=code, message=message, param=param, event_id=causal),
        ),
    )


def _validation(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_input=False)
    )


def _record_metric(ws: WebSocket, method: str, *args: object) -> None:
    metrics = getattr(ws.app.state, "metrics", None)
    if metrics is None:
        return
    try:
        getattr(metrics, method)(*args)
    except Exception as exc:
        try:
            metrics.record_telemetry_failure()
        except Exception:
            pass
        _LOGGER.error(
            "openai_realtime_metrics_failed",
            metric=method,
            exception_type=type(exc).__name__,
        )


async def _infer(
    ws: WebSocket, lock: asyncio.Lock, scheduler: Any, session_id: str, turn: TurnSnapshot
) -> None:
    started = time.perf_counter()
    try:
        audio = await asyncio.to_thread(turn.to_engine_audio)
        result = await scheduler.submit_final(
            f"{session_id}:{turn.item_id}",
            TranscriptionRequest(
                audio=audio, sample_rate=16000, language=turn.language.value, decoder="rnnt"
            ),
        )
        await _send(
            ws,
            lock,
            TranscriptionDeltaEvent(event_id=_id("event"), item_id=turn.item_id, delta=result.text),
        )
        await _send(
            ws,
            lock,
            TranscriptionCompletedEvent(
                event_id=_id("event"),
                item_id=turn.item_id,
                transcript=result.text,
                usage=DurationUsage(seconds=result.audio_duration_ms / 1000),
            ),
        )
        elapsed = time.perf_counter() - started
        _record_metric(ws, "record_transcription", turn.language, ProcessingMode.ACCURACY)
        _record_metric(
            ws,
            "record_audio_seconds",
            turn.language,
            ProcessingMode.ACCURACY,
            result.audio_duration_ms / 1_000,
        )
        _record_metric(ws, "record_final_latency", turn.language, ProcessingMode.ACCURACY, elapsed)
    except asyncio.CancelledError:
        raise
    except ServerBusyError as exc:
        _LOGGER.warning("openai_realtime_inference_busy", exception_type=type(exc).__name__)
        _record_metric(ws, "record_protocol_failure", "SERVER_BUSY")
        if ws.application_state is WebSocketState.CONNECTED:
            await _send(
                ws,
                lock,
                TranscriptionFailedEvent(
                    event_id=_id("event"),
                    item_id=turn.item_id,
                    error=ErrorDetail(
                        type="server_error",
                        code="server_busy",
                        message="inference queue is full",
                        event_id=turn.client_event_id,
                    ),
                ),
            )
    except Exception as exc:
        _LOGGER.error("openai_realtime_inference_failed", exception_type=type(exc).__name__)
        _record_metric(ws, "record_protocol_failure", "INFERENCE_ERROR")
        if ws.application_state is WebSocketState.CONNECTED:
            await _send(
                ws,
                lock,
                TranscriptionFailedEvent(
                    event_id=_id("event"),
                    item_id=turn.item_id,
                    error=ErrorDetail(
                        type="server_error",
                        code="transcription_failed",
                        message="transcription failed",
                        event_id=turn.client_event_id,
                    ),
                ),
            )


async def _wait_for_turn_slot(tasks: set[asyncio.Task[None]], limit: int) -> None:
    """Backpressure the receive loop before conversion/inference task creation."""

    while len(tasks) >= limit:
        done, _ = await asyncio.wait(tuple(tasks), return_when=asyncio.FIRST_COMPLETED)
        tasks.difference_update(done)
        await asyncio.gather(*done, return_exceptions=True)


def _schedule(
    tasks: set[asyncio.Task[None]],
    ws: WebSocket,
    lock: asyncio.Lock,
    scheduler: Any,
    session_id: str,
    turn: TurnSnapshot,
) -> None:
    task = asyncio.create_task(_infer(ws, lock, scheduler, session_id, turn))
    tasks.add(task)


async def _vad_failure(
    ws: WebSocket,
    lock: asyncio.Lock,
    exc: VADError,
    causal: str | None,
) -> NoReturn:
    if isinstance(exc, VADCapacityError):
        code = "vad_capacity_exceeded"
        message = "voice activity detection capacity was exceeded"
        close_code = 1013
    else:
        code = "vad_inference_failed"
        message = "voice activity detection failed"
        close_code = 1011
    await _error(ws, lock, code, message, causal, kind="server_error")
    await ws.close(code=close_code)
    raise _VADConnectionClosed


async def _reset_vad_stream(
    ws: WebSocket,
    lock: asyncio.Lock,
    stream: VADStream,
    causal: str | None,
) -> None:
    try:
        stream.reset()
    except Exception as exc:
        _LOGGER.error("openai_realtime_vad_reset_failed", exception_type=type(exc).__name__)
        failure = exc if isinstance(exc, VADError) else VADInferenceError("VAD stream reset failed")
        await _vad_failure(ws, lock, failure, causal)


class _VADConnectionClosed(Exception):
    """The VAD error event and terminal close frame have already been sent."""


async def _commit(
    ws: WebSocket,
    lock: asyncio.Lock,
    scheduler: Any,
    tasks: set[asyncio.Task[None]],
    state: RealtimeSessionState,
    vad_stream: VADStream | None,
    causal: str | None,
    item_id: str,
    max_pending_turns: int,
    size: int | None = None,
) -> bool:
    await _wait_for_turn_slot(tasks, max_pending_turns)
    try:
        turn = state.snapshot(item_id=item_id, client_event_id=causal, byte_count=size)
    except ValueError as exc:
        await _error(ws, lock, "invalid_audio_buffer", str(exc), causal, "audio")
        return False
    if vad_stream is not None:
        await _reset_vad_stream(ws, lock, vad_stream, causal)
    await _send(
        ws,
        lock,
        AudioCommittedEvent(
            event_id=_id("event"), previous_item_id=turn.previous_item_id, item_id=turn.item_id
        ),
    )
    _schedule(tasks, ws, lock, scheduler, state.session_id, turn)
    return True


async def _vad(
    ws: WebSocket,
    lock: asyncio.Lock,
    scheduler: Any,
    tasks: set[asyncio.Task[None]],
    state: RealtimeSessionState,
    vad_stream: VADStream | None,
    causal: str | None,
    max_pending_turns: int,
) -> None:
    config = state.turn_detection
    if config is None:
        return
    if vad_stream is None:
        await _vad_failure(
            ws,
            lock,
            VADInferenceError("configured VAD stream is unavailable"),
            causal,
        )
    stop_frames = (config.silence_duration_ms + 19) // 20
    while (pending := state.next_vad_frame()) is not None:
        frame, absolute_end = pending
        try:
            score = await vad_stream.score(frame)
        except VADCapacityError as exc:
            await _vad_failure(ws, lock, exc, causal)
        except VADError as exc:
            await _vad_failure(ws, lock, exc, causal)
        speech = score >= config.threshold
        provider_name = getattr(getattr(ws.app.state, "vad_provider", None), "name", "unknown")
        _record_metric(ws, "record_vad_decision", provider_name, speech)
        if not state.speech_active:
            state.speech_run_frames = state.speech_run_frames + 1 if speech else 0
            if state.speech_run_frames < 3:
                state.trim_pre_speech(config.prefix_padding_ms)
                continue
            state.speech_active = True
            state.pending_item_id = _id("item")
            await _send(
                ws,
                lock,
                SpeechStartedEvent(
                    event_id=_id("event"),
                    audio_start_ms=max(0, absolute_end - 60 - config.prefix_padding_ms),
                    item_id=state.pending_item_id,
                ),
            )
            continue
        state.silence_run_frames = 0 if speech else state.silence_run_frames + 1
        if state.silence_run_frames < stop_frames:
            continue
        item = state.pending_item_id or _id("item")
        _record_metric(ws, "record_vad_endpoint_event", "openai")
        await _send(
            ws,
            lock,
            SpeechStoppedEvent(event_id=_id("event"), audio_end_ms=absolute_end, item_id=item),
        )
        if not await _commit(
            ws,
            lock,
            scheduler,
            tasks,
            state,
            vad_stream,
            causal,
            item,
            max_pending_turns,
            state.vad_scan_bytes,
        ):
            state.reset_vad()
            await _reset_vad_stream(ws, lock, vad_stream, causal)
            return


def _configure_vad_stream(
    provider: VADProvider | None,
    current: VADStream | None,
    state: RealtimeSessionState,
) -> VADStream | None:
    if state.turn_detection is None:
        if current is not None:
            current.close()
        return None
    if current is None:
        if provider is None:
            raise VADInferenceError("VAD provider is unavailable")
        return provider.new_stream(24_000)
    current.reset()
    return current


async def _handle(
    ws: WebSocket,
    lock: asyncio.Lock,
    scheduler: Any,
    tasks: set[asyncio.Task[None]],
    state: RealtimeSessionState,
    vad_provider: Any,
    vad_stream: VADStream | None,
    event: object,
    config: OpenAIRealtimeConfig,
) -> VADStream | None:
    if isinstance(event, SessionUpdateEvent):
        try:
            audio_patch = event.session.audio
            input_patch = None if audio_patch is None else audio_patch.input
            transcription_patch = None if input_patch is None else input_patch.transcription
            canonical_model = None
            if transcription_patch is not None and "model" in transcription_patch.model_fields_set:
                if transcription_patch.model is None:
                    raise ValueError("transcription model cannot be null")
                canonical_model = validate_model(transcription_patch.model)
            state.apply_update(event.session, canonical_model)
        except OpenAIError as exc:
            await _error(
                ws,
                lock,
                exc.code or "invalid_request",
                exc.message,
                event.event_id,
                exc.param,
                "server_error" if exc.type == "server_error" else "invalid_request_error",
            )
            return vad_stream
        except ValueError as exc:
            await _error(ws, lock, "invalid_session", str(exc), event.event_id, "session")
            return vad_stream
        try:
            vad_stream = _configure_vad_stream(vad_provider, vad_stream, state)
        except VADCapacityError as exc:
            await _vad_failure(ws, lock, exc, event.event_id)
        except VADError as exc:
            await _vad_failure(ws, lock, exc, event.event_id)
        except Exception as exc:
            _LOGGER.error(
                "openai_realtime_vad_configuration_failed",
                exception_type=type(exc).__name__,
            )
            await _vad_failure(
                ws, lock, VADInferenceError("VAD stream configuration failed"), event.event_id
            )
        await _send(
            ws, lock, SessionUpdatedEvent(event_id=_id("event"), session=state.session_payload())
        )
        return vad_stream
    if isinstance(event, AudioAppendEvent):
        if not state.configured:
            await _error(
                ws,
                lock,
                "session_not_configured",
                "session.update with exactly one language is required before audio",
                event.event_id,
            )
            return vad_stream
        try:
            data = base64.b64decode(event.audio, validate=True)
        except (binascii.Error, ValueError):
            await _error(
                ws,
                lock,
                "invalid_base64",
                "audio must be strict RFC 4648 base64",
                event.event_id,
                "audio",
            )
            return vad_stream
        try:
            state.append(data)
        except OverflowError as exc:
            await _error(ws, lock, "audio_limit_exceeded", str(exc), event.event_id, "audio")
            return vad_stream
        await _vad(
            ws, lock, scheduler, tasks, state, vad_stream, event.event_id, config.max_pending_turns
        )
        return vad_stream
    if isinstance(event, AudioClearEvent):
        state.clear()
        if vad_stream is not None:
            await _reset_vad_stream(ws, lock, vad_stream, event.event_id)
        await _send(ws, lock, AudioClearedEvent(event_id=_id("event")))
        return vad_stream
    assert isinstance(event, AudioCommitEvent)
    item = state.pending_item_id or _id("item")
    if state.speech_active:
        await _send(
            ws,
            lock,
            SpeechStoppedEvent(
                event_id=_id("event"),
                audio_end_ms=state.total_audio_bytes * 1000 // 48000,
                item_id=item,
            ),
        )
    await _commit(
        ws,
        lock,
        scheduler,
        tasks,
        state,
        vad_stream,
        event.event_id,
        item,
        config.max_pending_turns,
    )
    return vad_stream


async def _run_session(
    ws: WebSocket,
    lock: asyncio.Lock,
    scheduler: Any,
    config: OpenAIRealtimeConfig,
) -> None:
    tasks: set[asyncio.Task[None]] = set()
    vad_stream: VADStream | None = None
    metric_started = False
    deadline = float("inf")
    try:
        settings: Settings = ws.app.state.settings
        now = int(time.time())
        state = RealtimeSessionState(
            session_id=_id("sess"),
            expires_at=now + int(config.max_session_seconds),
            max_audio_bytes=min(settings.max_upload_bytes, config.max_session_audio_bytes),
            max_audio_seconds=min(settings.max_audio_seconds, config.max_session_seconds),
            model=MODEL_ID,
        )
        vad_provider = getattr(ws.app.state, "vad_provider", None)
        deadline = asyncio.get_running_loop().time() + config.max_session_seconds
        _record_metric(ws, "session_started")
        metric_started = True
        async with asyncio.timeout_at(deadline):
            await _send(
                ws,
                lock,
                SessionCreatedEvent(event_id=_id("event"), session=state.session_payload()),
            )
            while True:
                try:
                    message = await asyncio.wait_for(
                        ws.receive(), timeout=config.idle_timeout_seconds
                    )
                except TimeoutError:
                    await _error(ws, lock, "idle_timeout", "session was idle for too long", None)
                    await ws.close(code=1001)
                    return
                if message["type"] == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is None:
                    await _error(ws, lock, "invalid_event", "events must be JSON text frames", None)
                    continue
                if len(text.encode()) > config.max_event_bytes:
                    await _error(
                        ws, lock, "event_too_large", "event exceeds the 15 MiB limit", None
                    )
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await _error(ws, lock, "invalid_json", "event must contain valid JSON", None)
                    continue
                causal = payload.get("event_id") if isinstance(payload, dict) else None
                causal = causal if isinstance(causal, str) else None
                try:
                    event = CLIENT_EVENT_ADAPTER.validate_python(payload)
                except ValidationError as exc:
                    await _error(ws, lock, "invalid_event", _validation(exc), causal)
                    continue
                vad_stream = await _handle(
                    ws,
                    lock,
                    scheduler,
                    tasks,
                    state,
                    vad_provider,
                    vad_stream,
                    event,
                    config,
                )
    except TimeoutError as exc:
        if asyncio.get_running_loop().time() >= deadline:
            await _error(ws, lock, "session_expired", "maximum session duration reached", None)
            if ws.application_state is WebSocketState.CONNECTED:
                await ws.close(code=1000)
            return
        _LOGGER.error("openai_realtime_session_failed", exception_type=type(exc).__name__)
        if ws.application_state is WebSocketState.CONNECTED:
            await _error(
                ws,
                lock,
                "internal_error",
                "realtime session failed",
                None,
                kind="server_error",
            )
            await ws.close(code=1011)
        return
    except (WebSocketDisconnect, _VADConnectionClosed):
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _LOGGER.error("openai_realtime_session_failed", exception_type=type(exc).__name__)
        if ws.application_state is WebSocketState.CONNECTED:
            await _error(
                ws,
                lock,
                "internal_error",
                "realtime session failed",
                None,
                kind="server_error",
            )
            await ws.close(code=1011)
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if vad_stream is not None:
            try:
                vad_stream.close()
            except Exception as exc:
                _LOGGER.error(
                    "openai_realtime_vad_close_failed",
                    exception_type=type(exc).__name__,
                )
        if metric_started:
            _record_metric(ws, "session_ended")


async def _serve(
    ws: WebSocket,
    scheduler: Any,
    config: OpenAIRealtimeConfig,
    registry: ConnectionRegistry,
) -> None:
    if not websocket_admitted(ws):
        _record_metric(ws, "record_rejection", "INVALID_SESSION")
        await ws.close(code=1008)
        return
    await ws.accept()
    lock = asyncio.Lock()
    if scheduler is None or not scheduler.running:
        await _error(
            ws,
            lock,
            "service_unavailable",
            "transcription service is not ready",
            None,
            kind="server_error",
        )
        await ws.close(code=1013)
        return
    if not await registry.acquire():
        await _error(
            ws,
            lock,
            "server_busy",
            "maximum concurrent realtime sessions reached",
            None,
            kind="server_error",
        )
        await ws.close(code=1013)
        return
    try:
        await _run_session(ws, lock, scheduler, config)
    finally:
        await registry.release()

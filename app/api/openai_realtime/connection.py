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
from app.core.config import Settings
from app.engine.base import TranscriptionRequest
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


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeConfig:
    max_event_bytes: int = MAX_EVENT
    max_session_audio_bytes: int = 50 * 1024 * 1024
    max_session_seconds: float = 3600.0
    idle_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_event_bytes <= MAX_EVENT:
            raise ValueError("event limit must be 1 byte to 15 MiB")
        if self.max_session_audio_bytes < 2:
            raise ValueError("audio limit is too small")
        if self.max_session_seconds <= 0 or self.idle_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")


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


async def _infer(
    ws: WebSocket, lock: asyncio.Lock, scheduler: Any, session_id: str, turn: TurnSnapshot
) -> None:
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
    except asyncio.CancelledError:
        raise
    except Exception:
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
    task.add_done_callback(tasks.discard)


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
    size: int | None = None,
) -> bool:
    try:
        turn = state.snapshot(item_id=item_id, client_event_id=causal, byte_count=size)
    except ValueError as exc:
        await _error(ws, lock, "invalid_audio_buffer", str(exc), causal, "audio")
        return False
    if vad_stream is not None:
        vad_stream.reset()
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
        ws.app.state.metrics.record_vad_decision(ws.app.state.vad_provider.name, speech)
        if not state.speech_active:
            state.speech_run_frames = state.speech_run_frames + 1 if speech else 0
            if state.speech_run_frames < 3:
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
        ws.app.state.metrics.record_vad_endpoint_event("openai")
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
            state.vad_scan_bytes,
        ):
            state.reset_vad()
            vad_stream.reset()
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
) -> VADStream | None:
    if isinstance(event, SessionUpdateEvent):
        try:
            state.apply_update(
                event.session, validate_model(event.session.audio.input.transcription.model)
            )
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
        await _vad(ws, lock, scheduler, tasks, state, vad_stream, event.event_id)
        return vad_stream
    if isinstance(event, AudioClearEvent):
        state.clear()
        if vad_stream is not None:
            vad_stream.reset()
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
    await _commit(ws, lock, scheduler, tasks, state, vad_stream, event.event_id, item)
    return vad_stream


async def _serve(ws: WebSocket, scheduler: Any, config: OpenAIRealtimeConfig) -> None:
    if not websocket_admitted(ws):
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
    settings: Settings = ws.app.state.settings
    now = int(time.time())
    state = RealtimeSessionState(
        session_id=_id("sess"),
        expires_at=now + int(config.max_session_seconds),
        max_audio_bytes=min(settings.max_upload_bytes, config.max_session_audio_bytes),
        max_audio_seconds=min(settings.max_audio_seconds, config.max_session_seconds),
        model=MODEL_ID,
    )
    tasks: set[asyncio.Task[None]] = set()
    vad_provider = getattr(ws.app.state, "vad_provider", None)
    vad_stream: VADStream | None = None
    started = time.monotonic()
    await _send(
        ws, lock, SessionCreatedEvent(event_id=_id("event"), session=state.session_payload())
    )
    try:
        while True:
            elapsed = time.monotonic() - started
            try:
                message = await asyncio.wait_for(
                    ws.receive(),
                    timeout=min(config.idle_timeout_seconds, config.max_session_seconds - elapsed),
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
                await _error(ws, lock, "event_too_large", "event exceeds the 15 MiB limit", None)
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
                ws, lock, scheduler, tasks, state, vad_provider, vad_stream, event
            )
    except (TimeoutError, WebSocketDisconnect, _VADConnectionClosed):
        return
    finally:
        if vad_stream is not None:
            vad_stream.close()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.openai_realtime import OpenAIRealtimeConfig, create_openai_realtime_router
from app.api.realtime import ConnectionRegistry
from app.api.websocket import WebSocketConfig, create_websocket_router
from app.engine.base import TranscriptionRequest
from app.openai_compat.constants import is_openai_route
from tests.support.asgi import (
    SchedulerDouble,
    busy_scheduler,
    failing_scheduler,
    scheduler_app,
    settings_for_tests,
)
from tests.support.openai_realtime import OPENAI_TRANSCRIPTION_PATH, OpenAIRealtimeDriver
from tests.support.realtime import REALTIME_PATH, RealtimeDriver


def pcm24(duration_ms: int, amplitude: float = 0.4) -> bytes:
    samples = np.full(24_000 * duration_ms // 1_000, round(amplitude * 32_767), dtype="<i2")
    return samples.tobytes()


class ReverseScheduler(SchedulerDouble):
    def __init__(self) -> None:
        super().__init__()
        self._release = asyncio.Event()

    async def submit_final(self, session_id: str, request: TranscriptionRequest):  # type: ignore[no-untyped-def]
        self.finals.append(request)
        index = len(self.finals)
        if index == 1:
            await self._release.wait()
        else:
            self._release.set()
        result = self._result(request, index - 1)
        return type(result)(
            text=f"turn-{index}",
            language=result.language,
            decoder=result.decoder,
            audio_duration_ms=result.audio_duration_ms,
            inference_ms=result.inference_ms,
        )


def test_standard_realtime_path_is_the_only_openai_realtime_prefix() -> None:
    assert is_openai_route(OPENAI_TRANSCRIPTION_PATH)
    assert not is_openai_route(REALTIME_PATH)
    assert not is_openai_route("/v1/realtime/transcription_sessions")


def test_manual_commit_is_acknowledged_before_its_correlated_final() -> None:
    scheduler = SchedulerDouble(text="namaste")
    with TestClient(scheduler_app(scheduler)) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            created = rt.expect("session.created")
            assert created["session"]["type"] == "transcription"
            assert created["session"]["audio"]["input"]["format"] == {
                "type": "audio/pcm",
                "rate": 24000,
            }
            rt.update(turn_detection=None)
            updated = rt.expect("session.updated")
            assert updated["session"]["audio"]["input"]["transcription"]["languages"] == ["hi"]
            audio = pcm24(40)
            rt.append(audio[:37])
            rt.append(audio[37:])
            rt.commit()
            committed = rt.expect("input_audio_buffer.committed")
            delta = rt.expect("conversation.item.input_audio_transcription.delta")
            completed = rt.expect("conversation.item.input_audio_transcription.completed")
            assert delta["item_id"] == completed["item_id"] == committed["item_id"]
            assert completed["transcript"] == "namaste"
            assert completed["usage"] == {"type": "duration", "seconds": 0.04}
            assert scheduler.finals[0].sample_rate == 16000


def test_clear_invalid_base64_empty_commit_and_bounds_are_recoverable() -> None:
    with TestClient(scheduler_app(SchedulerDouble(), max_upload_bytes=2)) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=None)
            rt.expect("session.updated")
            rt.send_json(
                {"type": "input_audio_buffer.append", "event_id": "bad-b64", "audio": "%%%"}
            )
            error = rt.expect_error("invalid_base64")
            assert error["error"]["event_id"] == "bad-b64"
            rt.append(b"\0\0\0\0", "too-large")
            rt.expect_error("audio_limit_exceeded")
            rt.clear()
            rt.expect("input_audio_buffer.cleared")
            rt.commit("empty")
            error = rt.expect_error("invalid_audio_buffer")
            assert error["error"]["event_id"] == "empty"
            rt.clear("still-open")
            rt.expect("input_audio_buffer.cleared")


def test_server_vad_uses_one_item_for_started_stopped_commit_and_final() -> None:
    with TestClient(scheduler_app(SchedulerDouble())) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(
                turn_detection={
                    "type": "server_vad",
                    "threshold": 0.1,
                    "prefix_padding_ms": 0,
                    "silence_duration_ms": 100,
                }
            )
            rt.expect("session.updated")
            rt.append(pcm24(60) + pcm24(100, 0.0))
            started = rt.expect("input_audio_buffer.speech_started")
            stopped = rt.expect("input_audio_buffer.speech_stopped")
            committed = rt.expect("input_audio_buffer.committed")
            delta = rt.expect("conversation.item.input_audio_transcription.delta")
            completed = rt.expect("conversation.item.input_audio_transcription.completed")
            assert {
                started["item_id"],
                stopped["item_id"],
                committed["item_id"],
                delta["item_id"],
                completed["item_id"],
            } == {started["item_id"]}


def test_later_commits_remain_receivable_and_can_finish_first() -> None:
    scheduler = ReverseScheduler()
    with TestClient(scheduler_app(scheduler)) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=None)
            rt.expect("session.updated")
            rt.append(pcm24(20), "a1")
            rt.commit("c1")
            first = rt.expect("input_audio_buffer.committed")
            rt.append(pcm24(20), "a2")
            rt.commit("c2")
            second = rt.expect("input_audio_buffer.committed")
            assert second["previous_item_id"] == first["item_id"]
            events = [rt.next_event() for _ in range(4)]
            by_item = {
                item_id: [event["type"] for event in events if event["item_id"] == item_id]
                for item_id in (first["item_id"], second["item_id"])
            }
            expected = [
                "conversation.item.input_audio_transcription.delta",
                "conversation.item.input_audio_transcription.completed",
            ]
            assert by_item[first["item_id"]] == expected
            assert by_item[second["item_id"]] == expected
            completion_order = [
                event["item_id"] for event in events if event["type"].endswith(".completed")
            ]
            assert completion_order == [second["item_id"], first["item_id"]]


def test_native_route_still_starts_with_session_start_and_binary_pcm() -> None:
    with TestClient(scheduler_app(SchedulerDouble())) as client:
        with client.websocket_connect(REALTIME_PATH) as socket:
            native = RealtimeDriver(socket)
            native.send_start(language="hi", vad=False)
            native.expect("session.ready")
            native.send_frame(bytes(640))
            native.send_commit()
            native.expect("transcript.final")


def _realtime_app(
    scheduler: Any,
    *,
    registry: ConnectionRegistry | None = None,
    openai_config: OpenAIRealtimeConfig | None = None,
    include_native: bool = False,
    metrics: object | None = None,
) -> FastAPI:
    app = FastAPI()
    shared = registry or ConnectionRegistry(8)
    app.state.scheduler = scheduler
    app.state.settings = settings_for_tests()
    app.state.vad_provider = None
    if metrics is not None:
        app.state.metrics = metrics
    app.state.realtime_connections = shared
    app.include_router(create_openai_realtime_router(scheduler, openai_config, shared))
    if include_native:
        app.include_router(
            create_websocket_router(scheduler, WebSocketConfig(max_sessions=shared.limit), shared)
        )
    return app


def test_initial_patch_still_requires_explicit_language() -> None:
    with TestClient(_realtime_app(SchedulerDouble())) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.send_json(
                {
                    "type": "session.update",
                    "event_id": "missing-language",
                    "session": {},
                }
            )
            error = rt.expect_error("invalid_session")
            assert error["error"]["event_id"] == "missing-language"
            rt.send_json(
                {
                    "type": "session.update",
                    "event_id": "language-only",
                    "session": {
                        "audio": {
                            "input": {
                                "transcription": {"language": "hi"},
                                "turn_detection": None,
                            }
                        }
                    },
                }
            )
            updated = rt.expect("session.updated")
            assert updated["session"]["audio"]["input"]["transcription"]["languages"] == ["hi"]


def test_session_update_merges_partial_nested_patches() -> None:
    with TestClient(_realtime_app(SchedulerDouble())) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=None)
            configured = rt.expect("session.updated")["session"]
            rt.send_json(
                {
                    "type": "session.update",
                    "event_id": "patch-language",
                    "session": {"audio": {"input": {"transcription": {"language": "ta"}}}},
                }
            )
            patched = rt.expect("session.updated")["session"]

    configured_input = configured["audio"]["input"]
    patched_input = patched["audio"]["input"]
    assert patched_input["transcription"]["model"] == configured_input["transcription"]["model"]
    assert patched_input["transcription"]["languages"] == ["ta"]
    assert patched_input["format"] == configured_input["format"]
    assert patched_input["turn_detection"] is None


def test_hard_session_lifetime_is_not_reported_as_idle_timeout() -> None:
    config = OpenAIRealtimeConfig(max_session_seconds=0.05, idle_timeout_seconds=5.0)
    with TestClient(_realtime_app(SchedulerDouble(), openai_config=config)) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            error = rt.expect_error("session_expired", timeout=2.0)
            assert error["error"]["message"] == "maximum session duration reached"
            rt.expect_close(1000)


def test_both_protocols_share_one_application_connection_cap() -> None:
    registry = ConnectionRegistry(1)
    app = _realtime_app(SchedulerDouble(), registry=registry, include_native=True)
    with TestClient(app) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as openai_socket:
            OpenAIRealtimeDriver(openai_socket).expect("session.created")
            assert registry.active == 1
            with client.websocket_connect(REALTIME_PATH) as native_socket:
                native = RealtimeDriver(native_socket)
                native.expect_error("SERVER_BUSY")
                native.expect_close(1013)
            assert registry.active == 1
        assert registry.active == 0
        with client.websocket_connect(REALTIME_PATH) as native_socket:
            native = RealtimeDriver(native_socket)
            native.send_start(vad=False)
            native.expect("session.ready")
            assert registry.active == 1
    assert registry.active == 0


class _BlockingScheduler(SchedulerDouble):
    def __init__(self) -> None:
        super().__init__()
        self.release_event = threading.Event()
        self.started_event = threading.Event()
        self.active = 0
        self.max_active = 0

    async def submit_final(self, session_id: str, request: TranscriptionRequest):  # type: ignore[no-untyped-def]
        self.finals.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started_event.set()
        try:
            await asyncio.to_thread(self.release_event.wait)
            return self._result(request, len(self.finals) - 1)
        finally:
            self.active -= 1


def test_pending_turns_backpressure_before_spawning_more_work() -> None:
    scheduler = _BlockingScheduler()
    config = OpenAIRealtimeConfig(max_pending_turns=1)
    with TestClient(_realtime_app(scheduler, openai_config=config)) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=None)
            rt.expect("session.updated")
            rt.append(pcm24(20), "first-audio")
            rt.commit("first-commit")
            rt.expect("input_audio_buffer.committed")
            assert scheduler.started_event.wait(1.0)

            rt.append(pcm24(20), "second-audio")
            rt.commit("second-commit")
            try:
                rt.expect_silence(0.05)
                assert len(scheduler.finals) == 1
                assert scheduler.max_active == 1
            finally:
                scheduler.release_event.set()
            observed = rt.collect_until("input_audio_buffer.committed", timeout=2.0)
            assert RealtimeDriver.events_of(observed, "input_audio_buffer.committed")
            rt.collect_until("conversation.item.input_audio_transcription.completed", timeout=2.0)
    assert len(scheduler.finals) == 2
    assert scheduler.max_active == 1


@pytest.mark.parametrize(
    ("scheduler", "expected_code"),
    [
        (busy_scheduler(), "server_busy"),
        (failing_scheduler(), "transcription_failed"),
    ],
)
def test_engine_busy_is_distinct_from_engine_failure(
    scheduler: SchedulerDouble, expected_code: str
) -> None:
    with TestClient(_realtime_app(scheduler)) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=None)
            rt.expect("session.updated")
            rt.append(pcm24(20))
            rt.commit()
            rt.expect("input_audio_buffer.committed")
            failed = rt.expect("conversation.item.input_audio_transcription.failed")
            assert failed["error"]["code"] == expected_code


class _MetricsSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def _record(self, method: str, *args: object) -> None:
        self.calls.append((method, args))

    def session_started(self) -> None:
        self._record("session_started")

    def session_ended(self) -> None:
        self._record("session_ended")

    def record_protocol_failure(self, code: str) -> None:
        self._record("record_protocol_failure", code)

    def record_transcription(self, *args: object) -> None:
        self._record("record_transcription", *args)

    def record_audio_seconds(self, *args: object) -> None:
        self._record("record_audio_seconds", *args)

    def record_final_latency(self, *args: object) -> None:
        self._record("record_final_latency", *args)

    def record_telemetry_failure(self) -> None:
        self._record("record_telemetry_failure")


def test_openai_realtime_records_session_success_and_failure_metrics() -> None:
    success_metrics = _MetricsSpy()
    with TestClient(_realtime_app(SchedulerDouble(), metrics=success_metrics)) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=None)
            rt.expect("session.updated")
            rt.append(pcm24(20))
            rt.commit()
            rt.expect("input_audio_buffer.committed")
            rt.expect("conversation.item.input_audio_transcription.delta")
            rt.expect("conversation.item.input_audio_transcription.completed")
    methods = [method for method, _ in success_metrics.calls]
    assert methods.count("session_started") == 1
    assert methods.count("session_ended") == 1
    assert methods.count("record_transcription") == 1
    assert methods.count("record_audio_seconds") == 1
    assert methods.count("record_final_latency") == 1

    failure_metrics = _MetricsSpy()
    with TestClient(_realtime_app(failing_scheduler(), metrics=failure_metrics)) as client:
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=None)
            rt.expect("session.updated")
            rt.append(pcm24(20))
            rt.commit()
            rt.expect("input_audio_buffer.committed")
            rt.expect("conversation.item.input_audio_transcription.failed")
    assert (
        "record_protocol_failure",
        ("INFERENCE_ERROR",),
    ) in failure_metrics.calls
    assert [method for method, _ in failure_metrics.calls].count("session_ended") == 1

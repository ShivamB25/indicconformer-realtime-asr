from __future__ import annotations

import asyncio

import numpy as np
from fastapi.testclient import TestClient

from app.engine.base import TranscriptionRequest
from tests.support.asgi import SchedulerDouble, scheduler_app
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

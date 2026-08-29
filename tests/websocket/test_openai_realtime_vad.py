from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine.base import TranscriptionRequest
from app.vad.base import VADCapacityError, VADInferenceError
from tests.support.asgi import SchedulerDouble, scheduler_app
from tests.support.openai_realtime import OPENAI_TRANSCRIPTION_PATH, OpenAIRealtimeDriver


def pcm24(duration_ms: int, amplitude: float = 0.4) -> bytes:
    samples = np.full(24_000 * duration_ms // 1_000, round(amplitude * 32_767), dtype="<i2")
    return samples.tobytes()


class VADStreamDouble:
    def __init__(
        self,
        scorer: Callable[[bytes], float] | None = None,
        error: Exception | None = None,
        on_close: Callable[[], None] | None = None,
        reset_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._scorer = scorer or (lambda frame: 0.0)
        self._error = error
        self._on_close = on_close
        self._reset_error = reset_error
        self._close_error = close_error
        self.frames: list[bytes] = []
        self.resets = 0
        self.closes = 0
        self.closed = False

    async def score(self, frame: bytes) -> float:
        self.frames.append(frame)
        if self._error is not None:
            raise self._error
        return self._scorer(frame)

    def reset(self) -> None:
        self.resets += 1
        if self._reset_error is not None:
            raise self._reset_error

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.closes += 1
        if self._on_close is not None:
            self._on_close()
        if self._close_error is not None:
            raise self._close_error


class VADProviderDouble:
    name = "energy"
    default_threshold = 0.5

    def __init__(
        self,
        stream_factory: Callable[[Callable[[], None]], VADStreamDouble] | None = None,
        new_stream_error: Exception | None = None,
    ) -> None:
        self._stream_factory = stream_factory
        self._new_stream_error = new_stream_error
        self.streams: list[VADStreamDouble] = []
        self.rates: list[int] = []
        self._active = 0

    @property
    def active_streams(self) -> int:
        return self._active

    def new_stream(self, input_sample_rate: int) -> VADStreamDouble:
        self.rates.append(input_sample_rate)
        if self._new_stream_error is not None:
            raise self._new_stream_error
        self._active += 1

        def release() -> None:
            self._active -= 1

        stream = (
            self._stream_factory(release)
            if self._stream_factory
            else VADStreamDouble(on_close=release)
        )
        self.streams.append(stream)
        return stream


SERVER_VAD = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 0,
    "silence_duration_ms": 100,
}


def test_fragmented_appends_feed_identical_exact_24khz_frames() -> None:
    provider = VADProviderDouble()
    app = scheduler_app(SchedulerDouble())
    audio = pcm24(60) + b"partial-frame-tail"
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            whole = OpenAIRealtimeDriver(socket)
            whole.expect("session.created")
            whole.update(turn_detection=SERVER_VAD)
            whole.expect("session.updated")
            whole.append(audio)
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            fragmented = OpenAIRealtimeDriver(socket)
            fragmented.expect("session.created")
            fragmented.update(turn_detection=SERVER_VAD)
            fragmented.expect("session.updated")
            offsets = (7, 958, 961, 1_919, len(audio))
            start = 0
            for index, end in enumerate(offsets):
                fragmented.append(audio[start:end], f"fragment-{index}")
                start = end
    assert provider.rates == [24_000, 24_000]
    assert provider.streams[0].frames == provider.streams[1].frames
    assert provider.streams[0].frames == [
        audio[index : index + 960] for index in range(0, 2_880, 960)
    ]


def test_server_vad_threshold_is_inclusive() -> None:
    scores = iter([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    provider = VADProviderDouble(
        lambda release: VADStreamDouble(lambda frame: next(scores), on_close=release)
    )
    app = scheduler_app(SchedulerDouble())
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            realtime = OpenAIRealtimeDriver(socket)
            realtime.expect("session.created")
            realtime.update(turn_detection=SERVER_VAD)
            realtime.expect("session.updated")
            realtime.append(pcm24(160))
            started = realtime.expect("input_audio_buffer.speech_started")
            stopped = realtime.expect("input_audio_buffer.speech_stopped")
            committed = realtime.expect("input_audio_buffer.committed")
    assert started["item_id"] == stopped["item_id"] == committed["item_id"]


def test_vad_resets_on_update_clear_manual_and_automatic_commit_and_disable() -> None:
    def classify(frame: bytes) -> float:
        return 1.0 if any(frame) else 0.0

    provider = VADProviderDouble(lambda release: VADStreamDouble(classify, on_close=release))
    app = scheduler_app(SchedulerDouble())
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            realtime = OpenAIRealtimeDriver(socket)
            realtime.expect("session.created")
            realtime.update(turn_detection=SERVER_VAD, event_id="enable")
            realtime.expect("session.updated")
            stream = provider.streams[0]
            realtime.update(turn_detection=SERVER_VAD, event_id="update")
            realtime.expect("session.updated")
            realtime.append(pcm24(20), "before-clear")
            realtime.clear()
            realtime.expect("input_audio_buffer.cleared")
            realtime.append(pcm24(20), "manual-audio")
            realtime.commit("manual")
            realtime.expect("input_audio_buffer.committed")
            realtime.expect("conversation.item.input_audio_transcription.delta")
            realtime.expect("conversation.item.input_audio_transcription.completed")
            realtime.append(pcm24(60) + pcm24(100, 0.0), "automatic-audio")
            realtime.expect("input_audio_buffer.speech_started")
            realtime.expect("input_audio_buffer.speech_stopped")
            realtime.expect("input_audio_buffer.committed")
            realtime.expect("conversation.item.input_audio_transcription.delta")
            realtime.expect("conversation.item.input_audio_transcription.completed")
            realtime.update(turn_detection=None, event_id="disable")
            realtime.expect("session.updated")
            assert stream.resets == 4
            assert stream.closes == 1
            assert provider.active_streams == 0
            realtime.update(turn_detection=SERVER_VAD, event_id="re-enable")
            realtime.expect("session.updated")
            assert len(provider.streams) == 2
    assert provider.streams[1].closes == 1
    assert provider.active_streams == 0


def test_multiple_automatic_turns_preserve_per_item_ordering() -> None:
    def classify(frame: bytes) -> float:
        return 1.0 if any(frame) else 0.0

    provider = VADProviderDouble(lambda release: VADStreamDouble(classify, on_close=release))
    app = scheduler_app(SchedulerDouble())
    one_turn = pcm24(60) + pcm24(100, 0.0)
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            realtime = OpenAIRealtimeDriver(socket)
            realtime.expect("session.created")
            realtime.update(turn_detection=SERVER_VAD)
            realtime.expect("session.updated")
            realtime.append(one_turn + one_turn)
            events = []
            completed = 0
            while completed < 2:
                event = realtime.next_event()
                events.append(event)
                completed += event["type"].endswith(".completed")
    committed = [event for event in events if event["type"] == "input_audio_buffer.committed"]
    assert len(committed) == 2
    for turn in committed:
        correlated = [event["type"] for event in events if event.get("item_id") == turn["item_id"]]
        assert correlated == [
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
        ]
    assert provider.streams[0].resets == 2


def test_connections_own_isolated_streams_and_release_each_lease() -> None:
    provider = VADProviderDouble()
    app = scheduler_app(SchedulerDouble())
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as first_socket:
            first = OpenAIRealtimeDriver(first_socket)
            first.expect("session.created")
            first.update(turn_detection=SERVER_VAD, event_id="first-update")
            first.expect("session.updated")
            first.append(pcm24(20, 0.1), "first-audio")
            with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as second_socket:
                second = OpenAIRealtimeDriver(second_socket)
                second.expect("session.created")
                second.update(turn_detection=SERVER_VAD, event_id="second-update")
                second.expect("session.updated")
                second.append(pcm24(20, 0.2), "second-audio")
                assert provider.active_streams == 2
                assert provider.streams[0].frames != provider.streams[1].frames
            assert provider.active_streams == 1
            assert provider.streams[1].closes == 1
        assert provider.active_streams == 0


@pytest.mark.parametrize(
    ("failure", "expected_code", "close_code"),
    [
        (VADCapacityError("deadline"), "vad_capacity_exceeded", 1013),
        (VADInferenceError("classifier"), "vad_inference_failed", 1011),
    ],
)
def test_vad_score_failures_emit_server_error_then_close(
    failure: Exception, expected_code: str, close_code: int
) -> None:
    provider = VADProviderDouble(lambda release: VADStreamDouble(error=failure, on_close=release))
    app = scheduler_app(SchedulerDouble())
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            realtime = OpenAIRealtimeDriver(socket)
            realtime.expect("session.created")
            realtime.update(turn_detection=SERVER_VAD)
            realtime.expect("session.updated")
            realtime.append(pcm24(20), "faulting-append")
            error = realtime.expect_error(expected_code)
            assert error["error"]["type"] == "server_error"
            assert error["error"]["event_id"] == "faulting-append"
            realtime.expect_close(close_code)
    assert provider.active_streams == 0


def test_vad_stream_capacity_failure_closes_1013_without_fallback() -> None:
    provider = VADProviderDouble(new_stream_error=VADCapacityError("stream limit"))
    app = scheduler_app(SchedulerDouble())
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            realtime = OpenAIRealtimeDriver(socket)
            realtime.expect("session.created")
            realtime.update(turn_detection=SERVER_VAD, event_id="capacity-update")
            error = realtime.expect_error("vad_capacity_exceeded")
            assert error["error"]["type"] == "server_error"
            assert error["error"]["event_id"] == "capacity-update"
            realtime.expect_close(1013)
    assert provider.streams == []


def test_prefix_padding_bounds_committed_audio_and_preserves_absolute_timestamps() -> None:
    def classify(frame: bytes) -> float:
        return 1.0 if any(frame) else 0.0

    provider = VADProviderDouble(lambda release: VADStreamDouble(classify, on_close=release))
    scheduler = SchedulerDouble()
    app = scheduler_app(scheduler)
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(
                turn_detection={
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 40,
                    "silence_duration_ms": 100,
                }
            )
            rt.expect("session.updated")
            rt.append(pcm24(400, 0.0) + pcm24(60) + pcm24(100, 0.0))
            started = rt.expect("input_audio_buffer.speech_started")
            stopped = rt.expect("input_audio_buffer.speech_stopped")
            rt.expect("input_audio_buffer.committed")
            rt.expect("conversation.item.input_audio_transcription.delta")
            completed = rt.expect("conversation.item.input_audio_transcription.completed")

    assert started["audio_start_ms"] == 360
    assert stopped["audio_end_ms"] == 560
    assert completed["usage"]["seconds"] == 0.2
    assert scheduler.finals[0].audio.size == 3_200


@pytest.mark.parametrize("action", ["update", "clear", "manual", "automatic"])
def test_every_vad_reset_failure_emits_server_error_and_closes_1011(action: str) -> None:
    def classify(frame: bytes) -> float:
        return 1.0 if any(frame) else 0.0

    provider = VADProviderDouble(
        lambda release: VADStreamDouble(
            classify,
            on_close=release,
            reset_error=VADInferenceError("reset fault must stay private"),
        )
    )
    app = scheduler_app(SchedulerDouble())
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=SERVER_VAD, event_id="enable")
            rt.expect("session.updated")
            if action == "update":
                rt.update(turn_detection=SERVER_VAD, event_id="reset-update")
                causal = "reset-update"
            elif action == "clear":
                rt.clear("reset-clear")
                causal = "reset-clear"
            elif action == "manual":
                rt.append(pcm24(20), "audio")
                rt.commit("reset-manual")
                causal = "reset-manual"
            else:
                rt.append(pcm24(60) + pcm24(100, 0.0), "reset-automatic")
                rt.expect("input_audio_buffer.speech_started")
                rt.expect("input_audio_buffer.speech_stopped")
                causal = "reset-automatic"
            error = rt.expect_error("vad_inference_failed")
            assert error["error"]["type"] == "server_error"
            assert error["error"]["event_id"] == causal
            rt.expect_close(1011)


class _CancellableScheduler(SchedulerDouble):
    def __init__(self) -> None:
        super().__init__()
        self.started_event = threading.Event()
        self.cancelled_event = threading.Event()

    async def submit_final(self, session_id: str, request: TranscriptionRequest):  # type: ignore[no-untyped-def]
        self.finals.append(request)
        self.started_event.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled_event.set()


def test_inference_tasks_are_cancelled_and_gathered_even_when_vad_close_raises() -> None:
    provider = VADProviderDouble(
        lambda release: VADStreamDouble(
            scorer=lambda _: 1.0,
            on_close=release,
            close_error=RuntimeError("private close fault"),
        )
    )
    scheduler = _CancellableScheduler()
    app = scheduler_app(scheduler)
    with TestClient(app) as client:
        app.state.vad_provider = provider
        with client.websocket_connect(OPENAI_TRANSCRIPTION_PATH) as socket:
            rt = OpenAIRealtimeDriver(socket)
            rt.expect("session.created")
            rt.update(turn_detection=SERVER_VAD)
            rt.expect("session.updated")
            rt.append(pcm24(60))
            rt.expect("input_audio_buffer.speech_started")
            rt.commit()
            rt.expect("input_audio_buffer.speech_stopped")
            rt.expect("input_audio_buffer.committed")
            assert scheduler.started_event.wait(1.0)
    assert scheduler.cancelled_event.wait(1.0)
    assert provider.active_streams == 0

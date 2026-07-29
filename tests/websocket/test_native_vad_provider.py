"""Native realtime integration tests for provider-backed streaming VAD."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.websocket import WebSocketConfig
from app.vad.base import VADCapacityError, VADClosedError, VADInferenceError
from tests.support.asgi import SchedulerDouble, realtime_only_app
from tests.support.audio import silence_frame, speech_frame
from tests.support.realtime import REALTIME_PATH, RealtimeDriver


@dataclass(slots=True)
class ScriptedVADStream:
    scores: list[float] = field(default_factory=list)
    failure: Exception | None = None
    scored_frames: list[bytes] = field(default_factory=list)
    resets: int = 0
    closes: int = 0

    async def score(self, frame: bytes) -> float:
        self.scored_frames.append(frame)
        if self.failure is not None:
            raise self.failure
        if not self.scores:
            return 0.0
        return self.scores.pop(0)

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closes += 1


@dataclass(slots=True)
class ScriptedVADProvider:
    stream_factory: Callable[[], ScriptedVADStream] = ScriptedVADStream
    default_threshold: float = 0.5
    acquisition_error: Exception | None = None
    streams: list[ScriptedVADStream] = field(default_factory=list)
    acquisitions: int = 0

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def active_streams(self) -> int:
        return sum(stream.closes == 0 for stream in self.streams)

    async def startup(self) -> None:
        return None

    def new_stream(self, input_sample_rate: int) -> ScriptedVADStream:
        assert input_sample_rate == 16_000
        self.acquisitions += 1
        if self.acquisition_error is not None:
            raise self.acquisition_error
        stream = self.stream_factory()
        self.streams.append(stream)
        return stream

    async def close(self) -> None:
        return None


def app_with_provider(
    provider: ScriptedVADProvider,
    scheduler: SchedulerDouble,
    config: WebSocketConfig | None = None,
) -> FastAPI:
    app = realtime_only_app(scheduler, config)
    app.state.vad_provider = provider
    return app


def test_vad_preserves_speech_order_and_all_onset_and_trailing_frames() -> None:
    scores = [1.0] * 3 + [0.0] * 30
    provider = ScriptedVADProvider(stream_factory=lambda: ScriptedVADStream(scores.copy()))
    scheduler = SchedulerDouble()
    app = app_with_provider(provider, scheduler)
    frames = [speech_frame()] * 3 + [silence_frame()] * 30

    with TestClient(app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(vad=True)
        realtime.expect("session.ready")
        realtime.send_frames(frames)
        observed = realtime.collect_until("transcript.final")

    assert RealtimeDriver.event_types(observed)[0] == "speech.started"
    assert RealtimeDriver.event_types(observed)[-1] == "transcript.final"
    assert scheduler.finals[0].audio.size == len(frames) * 320
    assert provider.streams[0].scored_frames == frames
    assert provider.streams[0].resets == 1
    assert provider.streams[0].closes == 1


def test_vad_disabled_bypasses_the_provider_entirely() -> None:
    provider = ScriptedVADProvider(
        acquisition_error=AssertionError("VAD-disabled sessions must not acquire a stream")
    )
    scheduler = SchedulerDouble()
    app = app_with_provider(provider, scheduler)

    with TestClient(app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(vad=False)
        realtime.expect("session.ready")
        realtime.send_frame(silence_frame())
        realtime.send_commit()
        assert realtime.expect("transcript.final")["audio_duration_ms"] == 20

    assert provider.acquisitions == 0
    assert provider.streams == []


def test_concurrent_sessions_have_independent_vad_stream_state() -> None:
    scripts = iter(([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]))
    provider = ScriptedVADProvider(stream_factory=lambda: ScriptedVADStream(list(next(scripts))))
    app = app_with_provider(provider, SchedulerDouble())

    with TestClient(app) as client:
        with (
            client.websocket_connect(REALTIME_PATH) as first_socket,
            client.websocket_connect(REALTIME_PATH) as second_socket,
        ):
            first = RealtimeDriver(first_socket)
            second = RealtimeDriver(second_socket)
            first.send_start(vad=True)
            second.send_start(vad=True)
            first.expect("session.ready")
            second.expect("session.ready")

            first.send_frames([speech_frame()] * 3)
            second.send_frames([speech_frame()] * 3)
            assert first.expect("speech.started") == {"type": "speech.started"}
            second.expect_silence()

    assert len(provider.streams) == 2
    assert provider.streams[0] is not provider.streams[1]
    assert [len(stream.scored_frames) for stream in provider.streams] == [3, 3]
    assert [stream.closes for stream in provider.streams] == [1, 1]


def test_manual_and_utterance_limit_finals_reset_and_release_their_streams() -> None:
    scripts = iter(([1.0], [1.0] * 10))
    provider = ScriptedVADProvider(stream_factory=lambda: ScriptedVADStream(list(next(scripts))))
    scheduler = SchedulerDouble()
    app = app_with_provider(
        provider,
        scheduler,
        WebSocketConfig(max_utterance_ms=200, partial_hybrid_ms=200),
    )

    with TestClient(app) as client:
        with client.websocket_connect(REALTIME_PATH) as socket:
            realtime = RealtimeDriver(socket)
            realtime.send_start(vad=True)
            realtime.expect("session.ready")
            realtime.send_frame(speech_frame())
            realtime.send_commit()
            realtime.expect("transcript.final")

        with client.websocket_connect(REALTIME_PATH) as socket:
            realtime = RealtimeDriver(socket)
            realtime.send_start(vad=True)
            realtime.expect("session.ready")
            realtime.send_frames([speech_frame()] * 10)
            observed = realtime.collect_until("transcript.final")
            assert RealtimeDriver.events_of(observed, "speech.started")

    assert len(scheduler.finals) == 2
    assert [stream.resets for stream in provider.streams] == [1, 1]
    assert [stream.closes for stream in provider.streams] == [1, 1]


def test_provider_default_threshold_is_used_unless_config_explicitly_overrides_it() -> None:
    default_provider = ScriptedVADProvider(
        default_threshold=0.75,
        stream_factory=lambda: ScriptedVADStream([0.6, 0.6, 0.6]),
    )
    default_app = app_with_provider(default_provider, SchedulerDouble())
    with TestClient(default_app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(vad=True)
        realtime.expect("session.ready")
        realtime.send_frames([speech_frame()] * 3)
        realtime.expect_silence()

    override_provider = ScriptedVADProvider(
        default_threshold=0.75,
        stream_factory=lambda: ScriptedVADStream([0.6, 0.6, 0.6]),
    )
    override_app = app_with_provider(
        override_provider,
        SchedulerDouble(),
        WebSocketConfig(vad_threshold=0.5),
    )
    with TestClient(override_app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(vad=True)
        realtime.expect("session.ready")
        realtime.send_frames([speech_frame()] * 3)
        realtime.expect("speech.started")


@pytest.mark.parametrize("failure_at", ["acquire", "score"])
def test_vad_capacity_is_retryable_and_closes_with_try_again_later(failure_at: str) -> None:
    error = VADCapacityError("full")
    if failure_at == "acquire":
        provider = ScriptedVADProvider(acquisition_error=error)
    else:
        provider = ScriptedVADProvider(stream_factory=lambda: ScriptedVADStream(failure=error))
    scheduler = SchedulerDouble()
    app = app_with_provider(provider, scheduler)

    with TestClient(app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(vad=True)
        if failure_at == "score":
            realtime.expect("session.ready")
            realtime.send_frame(speech_frame())
        failure = realtime.expect_error("SERVER_BUSY")
        assert failure["retryable"] is True
        realtime.expect_close(1013)

    assert scheduler.partials == []
    assert scheduler.finals == []


@pytest.mark.parametrize(
    "failure",
    [VADInferenceError("classifier failed"), VADClosedError("stream closed")],
)
def test_vad_runtime_failure_stops_before_any_scheduler_submission(failure: Exception) -> None:
    provider = ScriptedVADProvider(stream_factory=lambda: ScriptedVADStream(failure=failure))
    scheduler = SchedulerDouble()
    app = app_with_provider(provider, scheduler)

    with TestClient(app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(vad=True)
        realtime.expect("session.ready")
        realtime.send_frame(speech_frame())
        error = realtime.expect_error("INFERENCE_ERROR")
        assert error["retryable"] is True
        realtime.expect_close(1011)

    assert scheduler.partials == []
    assert scheduler.finals == []
    assert len(provider.streams[0].scored_frames) == 1
    assert provider.streams[0].closes == 1

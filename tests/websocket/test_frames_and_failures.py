"""Binary frame constraints and realtime failure paths.

A conforming client sends exactly one 20 ms mono PCM16 frame per binary message:
320 samples, which is 640 bytes. The tests below pin both halves of that contract
(what a frame must be, and what the server does when it is not) plus the error
code, retryable flag, and close code for every failure the session can report.

Inference is always a scheduler double, so a failure here means the realtime
protocol changed, never that a model behaved differently.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.websocket import WebSocketConfig
from app.api.websocket import connection as websocket_connection
from app.audio.pcm import PCM16_FRAME_BYTES, SAMPLES_PER_FRAME
from app.engine.scheduler import PartialOutstandingError, StalePartialError
from tests.support.asgi import (
    SchedulerDouble,
    busy_scheduler,
    failing_scheduler,
    mock_engine_app,
    realtime_only_app,
)
from tests.support.audio import silence_frame, speech_frame, speech_frames
from tests.support.realtime import REALTIME_PATH, RealtimeDriver


class _LogSpy:
    def __init__(self) -> None:
        self.errors: list[tuple[str, dict[str, object]]] = []

    def error(self, event: str, **fields: object) -> None:
        self.errors.append((event, fields))


# A partial needs one cadence interval of audio; 400 ms is the hybrid default.
FRAMES_TO_FIRST_PARTIAL = 400 // 20


class TestFrameSize:
    def test_the_frame_contract_is_exactly_one_twenty_millisecond_frame(self) -> None:
        assert SAMPLES_PER_FRAME == 320
        assert PCM16_FRAME_BYTES == 640
        assert len(speech_frame()) == PCM16_FRAME_BYTES

    def test_an_exact_frame_is_accepted(self) -> None:
        scheduler = SchedulerDouble()
        with TestClient(realtime_only_app(scheduler)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frame(silence_frame())
                realtime.expect_silence()
                realtime.send_commit()
                assert realtime.expect("transcript.final")["audio_duration_ms"] == 20

    @pytest.mark.parametrize(
        ("case", "size"),
        [
            ("sample_count_mistaken_for_bytes", 320),
            ("one_byte_short", PCM16_FRAME_BYTES - 1),
            ("half_frame", PCM16_FRAME_BYTES // 2),
            ("single_byte", 1),
            ("empty_message", 0),
            ("odd_length", 639),
        ],
    )
    def test_a_short_frame_is_refused_and_closes_the_socket(self, case: str, size: int) -> None:
        del case
        scheduler = SchedulerDouble()
        with TestClient(realtime_only_app(scheduler)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frame(b"\x00" * size)
                error = realtime.expect_error("INVALID_FRAME_SIZE")
                assert error["retryable"] is False
                assert str(PCM16_FRAME_BYTES) in error["message"]
                realtime.expect_close(1002)

        assert scheduler.finals == []
        assert scheduler.partials == []

    @pytest.mark.parametrize("size", [PCM16_FRAME_BYTES + 1, PCM16_FRAME_BYTES * 2, 4_096])
    def test_a_frame_beyond_the_byte_limit_is_refused_as_too_large(self, size: int) -> None:
        with TestClient(realtime_only_app(SchedulerDouble())) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frame(b"\x00" * size)
                error = realtime.expect_error("FRAME_TOO_LARGE")
                assert error["retryable"] is False
                realtime.expect_close(1009)

    def test_a_permitted_larger_message_still_has_to_be_one_frame(self) -> None:
        """Raising the byte ceiling must not relax the exact-frame requirement."""

        config = WebSocketConfig(max_frame_bytes=PCM16_FRAME_BYTES * 4)
        with TestClient(realtime_only_app(SchedulerDouble(), config)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frame(speech_frame() * 2)
                realtime.expect_error("INVALID_FRAME_SIZE")
                realtime.expect_close(1002)

    def test_frames_are_never_reassembled_across_messages(self) -> None:
        """Two half frames are two protocol violations, not one whole frame."""

        with TestClient(realtime_only_app(SchedulerDouble())) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                half = PCM16_FRAME_BYTES // 2
                realtime.send_frame(b"\x00" * half)
                realtime.expect_error("INVALID_FRAME_SIZE")
                realtime.expect_close(1002)


class TestExplicitCommitFailures:
    def test_a_commit_without_audio_is_refused_without_closing(self) -> None:
        scheduler = SchedulerDouble()
        with TestClient(realtime_only_app(scheduler)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_commit()
                error = realtime.expect_error("EMPTY_UTTERANCE")
                assert error["retryable"] is False

                realtime.send_frame(speech_frame())
                realtime.send_commit()
                assert realtime.expect("transcript.final")["audio_duration_ms"] == 20

        assert len(scheduler.finals) == 1

    def test_repeated_empty_commits_keep_the_session_usable(self) -> None:
        with TestClient(realtime_only_app(SchedulerDouble())) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                for _ in range(3):
                    realtime.send_commit()
                    realtime.expect_error("EMPTY_UTTERANCE")
                realtime.expect_silence()


class TestFinalInferenceFailures:
    def test_a_full_queue_is_retryable_and_closes_with_try_again_later(self) -> None:
        with TestClient(realtime_only_app(busy_scheduler())) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frame(speech_frame())
                realtime.send_commit()
                error = realtime.expect_error("SERVER_BUSY")
                assert error["retryable"] is True
                realtime.expect_close(1013)

    def test_an_engine_failure_is_retryable_and_closes_as_an_internal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logger = _LogSpy()
        monkeypatch.setattr(websocket_connection, "_LOGGER", logger)
        with TestClient(realtime_only_app(failing_scheduler())) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frame(speech_frame())
                realtime.send_commit()
                error = realtime.expect_error("INFERENCE_ERROR")
                assert error["retryable"] is True
                realtime.expect_close(1011)

        assert logger.errors == [("realtime_final_failed", {"exception_type": "RuntimeError"})]

    def test_no_transcript_is_ever_emitted_alongside_a_failure(self) -> None:
        with TestClient(realtime_only_app(failing_scheduler())) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frame(speech_frame())
                realtime.send_commit()
                observed = realtime.collect_until("error")

        assert RealtimeDriver.events_of(observed, "transcript.final") == []
        assert RealtimeDriver.events_of(observed, "transcript.partial") == []


class TestPartialInferenceFailures:
    def test_a_failing_partial_reports_an_error_but_preserves_the_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logger = _LogSpy()
        monkeypatch.setattr(websocket_connection, "_LOGGER", logger)
        scheduler = SchedulerDouble(partial_error=RuntimeError("partial exploded"))
        with TestClient(realtime_only_app(scheduler)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frames([speech_frame()] * FRAMES_TO_FIRST_PARTIAL)
                error = realtime.expect_error("INFERENCE_ERROR")
                assert error["retryable"] is True
                realtime.send_commit()
                assert realtime.expect("transcript.final")["decoder"] == "rnnt"

        assert len(scheduler.partials) == 1
        assert len(scheduler.finals) == 1
        assert logger.errors == [("realtime_partial_failed", {"exception_type": "RuntimeError"})]

    def test_a_busy_partial_reports_an_error_but_preserves_the_session(self) -> None:
        scheduler = SchedulerDouble(partial_error=busy_scheduler().partial_error)
        with TestClient(realtime_only_app(scheduler)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frames([speech_frame()] * FRAMES_TO_FIRST_PARTIAL)
                assert realtime.expect_error("SERVER_BUSY")["retryable"] is True
                realtime.send_commit()
                assert realtime.expect("transcript.final")["decoder"] == "rnnt"

        assert len(scheduler.finals) == 1

    @pytest.mark.parametrize(
        ("case", "error"),
        [
            ("superseded", StalePartialError("superseded")),
            ("already_running", PartialOutstandingError("outstanding")),
        ],
    )
    def test_a_discarded_partial_is_silent_and_does_not_disturb_the_session(
        self, case: str, error: Exception
    ) -> None:
        del case
        scheduler = SchedulerDouble(partial_error=error)
        with TestClient(realtime_only_app(scheduler)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frames([speech_frame()] * FRAMES_TO_FIRST_PARTIAL)
                realtime.expect_silence()
                realtime.send_commit()
                final = realtime.expect("transcript.final")

        assert final["audio_duration_ms"] == 400
        assert len(scheduler.partials) == 1


class TestUtteranceLimit:
    def test_reaching_the_utterance_limit_finalizes_and_keeps_the_session_open(self) -> None:
        scheduler = SchedulerDouble()
        config = WebSocketConfig(max_utterance_ms=200)
        with TestClient(realtime_only_app(scheduler, config)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frames([speech_frame()] * 10)
                first = realtime.expect("transcript.final")

                realtime.send_frames([speech_frame()] * 10)
                second = realtime.expect("transcript.final")

        assert first["audio_duration_ms"] == 200
        assert second["audio_duration_ms"] == 200
        assert len(scheduler.finals) == 2

    def test_the_limit_is_enforced_with_voice_activity_detection_too(self) -> None:
        scheduler = SchedulerDouble()
        config = WebSocketConfig(max_utterance_ms=200, speech_end_ms=600)
        with TestClient(realtime_only_app(scheduler, config)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=True)
                realtime.expect("session.ready")
                realtime.send_frames(speech_frames(400))
                observed = realtime.collect_until("transcript.final")

        assert RealtimeDriver.events_of(observed, "speech.started") != []
        finals = RealtimeDriver.events_of(observed, "transcript.final")
        assert finals[0]["audio_duration_ms"] == 200

    def test_audio_is_never_buffered_past_the_limit(self) -> None:
        scheduler = SchedulerDouble()
        config = WebSocketConfig(max_utterance_ms=200)
        with TestClient(realtime_only_app(scheduler, config)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(vad=False)
                realtime.expect("session.ready")
                realtime.send_frames([speech_frame()] * 30)
                realtime.collect_until("transcript.final")
                realtime.drain()

        assert [request.audio.size for request in scheduler.finals] == [3_200, 3_200, 3_200]


class TestIdleAndDisconnect:
    def test_an_idle_session_is_closed_with_its_own_code(self) -> None:
        config = WebSocketConfig(idle_timeout_seconds=0.05)
        with TestClient(realtime_only_app(SchedulerDouble(), config)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                error = realtime.expect_error("IDLE_TIMEOUT", timeout=5.0)
                assert error["retryable"] is False
                realtime.expect_close(1001)

    def test_the_hard_session_lifetime_is_not_reported_as_idle(self) -> None:
        config = WebSocketConfig(max_session_seconds=0.05, idle_timeout_seconds=5.0)
        with TestClient(realtime_only_app(SchedulerDouble(), config)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.expect_error("SESSION_LIMIT", timeout=2.0)
                realtime.expect_close(1000)

    def test_a_client_disconnect_releases_the_session_slot(self) -> None:
        config = WebSocketConfig(max_sessions=1)
        app = realtime_only_app(SchedulerDouble(), config)
        with TestClient(app) as client:
            for _ in range(3):
                with client.websocket_connect(REALTIME_PATH) as socket:
                    realtime = RealtimeDriver(socket)
                    realtime.send_start(vad=False)
                    realtime.expect("session.ready")


class TestServiceState:
    def test_a_session_is_refused_while_the_application_is_not_ready(self) -> None:
        app = mock_engine_app()
        client = TestClient(app)

        with client.websocket_connect(REALTIME_PATH) as socket:
            realtime = RealtimeDriver(socket)
            error = realtime.expect_error("SERVICE_UNAVAILABLE")
            assert error["retryable"] is True
            realtime.expect_close(1013)

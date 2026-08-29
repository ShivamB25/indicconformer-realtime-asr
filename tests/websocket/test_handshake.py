"""Realtime handshake: who may talk, when, and in what shape.

Only the wire protocol is used here. Sessions are opened against the real
application factory with an injected MockEngine, or against a router-only app
when the point of the test is that no scheduler was ever bound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.websocket import WebSocketConfig
from app.core.types import SUPPORTED_LANGUAGE_CODES
from tests.support.asgi import SchedulerDouble, mock_engine_app, realtime_only_app
from tests.support.realtime import REALTIME_PATH, RealtimeDriver, start_payload


class TestSessionStart:
    def test_a_valid_start_is_acknowledged_with_a_session_id(self) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(language="hi")
                ready = realtime.expect("session.ready")
                assert set(ready) == {"type", "session_id"}
                assert len(ready["session_id"]) == 32
                assert int(ready["session_id"], 16) >= 0

    def test_session_ids_are_not_reused_across_connections(self) -> None:
        identifiers = set()
        with TestClient(mock_engine_app()) as client:
            for _ in range(3):
                with client.websocket_connect(REALTIME_PATH) as socket:
                    realtime = RealtimeDriver(socket)
                    realtime.send_start()
                    identifiers.add(realtime.expect("session.ready")["session_id"])
        assert len(identifiers) == 3

    def test_optional_fields_may_be_omitted(self) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_json({"type": "session.start", "language": "ta"})
                realtime.expect("session.ready")
                realtime.expect_silence()

    def test_a_fully_specified_start_is_accepted(self) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_json(
                    {
                        "type": "session.start",
                        "language": "mr",
                        "format": "pcm_s16le",
                        "sample_rate": 16_000,
                        "channels": 1,
                        "mode": "accuracy",
                        "vad": False,
                    }
                )
                realtime.expect("session.ready")

    @pytest.mark.parametrize("language", sorted(SUPPORTED_LANGUAGE_CODES))
    def test_every_supported_language_opens_a_session(self, language: str) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start(language=language)
                realtime.expect("session.ready")

    @pytest.mark.parametrize(
        "payload",
        [
            start_payload(language="en"),
            start_payload(language="HI"),
            start_payload(language="hi-IN"),
            start_payload(language=""),
            start_payload(language=None),
            {"type": "session.start"},
            start_payload(sample_rate=8_000),
            start_payload(sample_rate=44_100),
            start_payload(sample_rate="16000"),
            start_payload(channels=2),
            start_payload(channels=0),
            start_payload(format="opus"),
            start_payload(format="wav"),
            start_payload(format="pcm_s16be"),
            start_payload(mode="fast"),
            start_payload(mode="LATENCY"),
            start_payload(mode=""),
            start_payload(vad="true"),
            start_payload(vad=1),
            start_payload(sample_rate=16_000.0),
            start_payload(unexpected="field"),
        ],
        ids=[
            "unsupported_language",
            "uppercase_language",
            "locale_language",
            "empty_language",
            "null_language",
            "missing_language",
            "sample_rate_8k",
            "sample_rate_44k",
            "sample_rate_as_string",
            "stereo",
            "zero_channels",
            "opus_format",
            "wav_format",
            "big_endian_pcm",
            "unknown_mode",
            "uppercase_mode",
            "empty_mode",
            "vad_as_string",
            "vad_as_int",
            "sample_rate_as_float",
            "unknown_field",
        ],
    )
    def test_an_unsupported_start_is_refused_and_the_socket_closes(
        self, payload: dict[str, Any]
    ) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_json(payload)
                error = realtime.expect_error("INVALID_SESSION")
                assert error["retryable"] is False
                assert error["message"]
                realtime.expect_close(1002)

    def test_a_second_start_is_refused(self) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start()
                realtime.expect("session.ready")
                realtime.send_start(language="bn")
                realtime.expect_error("SESSION_ALREADY_STARTED")
                realtime.expect_close(1002)


class TestFirstEventOrdering:
    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "input.commit"},
            {"type": "speech.started"},
            {"type": "transcript.final", "text": "spoofed"},
            {},
            {"language": "hi"},
        ],
        ids=["commit", "server_event", "spoofed_final", "empty_object", "no_type"],
    )
    def test_control_events_before_session_start_are_refused(self, payload: dict[str, Any]) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_json(payload)
                realtime.expect_error("SESSION_REQUIRED")
                realtime.expect_close(1002)

    @pytest.mark.parametrize(
        "text",
        ["", "not json", "[]", '"session.start"', "42", "null", '{"type": "session.start"'],
        ids=["empty", "plain_text", "array", "string", "number", "null", "truncated_object"],
    )
    def test_text_that_is_not_a_json_object_is_refused(self, text: str) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_text(text)
                realtime.expect_error("MALFORMED_EVENT")
                realtime.expect_close(1002)

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "input.flush"},
            {"type": "input.commit", "extra": 1},
            {"type": "session.stop"},
            {"type": ""},
        ],
        ids=["unknown_type", "commit_with_extra_field", "session_stop", "empty_type"],
    )
    def test_only_input_commit_is_accepted_after_start(self, payload: dict[str, Any]) -> None:
        with TestClient(mock_engine_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start()
                realtime.expect("session.ready")
                realtime.send_json(payload)
                realtime.expect_error("UNKNOWN_EVENT")
                realtime.expect_close(1002)


class TestAdmission:
    TOKEN = "test-service-api-key-that-is-long-enough"

    @classmethod
    def secured_app(cls, tmp_path: Path, **settings: Any) -> Any:
        key_file = tmp_path / "api.key"
        key_file.write_text(cls.TOKEN, encoding="utf-8")
        return mock_engine_app(
            api_key_file=key_file,
            **settings,
        )

    @pytest.mark.parametrize(
        "authorization",
        [None, "Basic credentials", "Bearer", "Bearer  malformed", "Bearer wrong-token"],
    )
    def test_missing_malformed_or_wrong_bearer_is_rejected_before_accept(
        self, tmp_path: Path, authorization: str | None
    ) -> None:
        headers = {} if authorization is None else {"authorization": authorization}
        with TestClient(self.secured_app(tmp_path)) as client:
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(REALTIME_PATH, headers=headers):
                    pass
        assert caught.value.code == 1008

    def test_present_origin_must_be_allowlisted(self, tmp_path: Path) -> None:
        app = self.secured_app(tmp_path, websocket_allowed_origins=("https://speech.example",))
        headers = {
            "authorization": f"Bearer {self.TOKEN}",
            "origin": "https://evil.example",
        }
        with TestClient(app) as client:
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(REALTIME_PATH, headers=headers):
                    pass
        assert caught.value.code == 1008

    def test_valid_bearer_and_allowlisted_origin_are_accepted(self, tmp_path: Path) -> None:
        app = self.secured_app(tmp_path, websocket_allowed_origins=("https://speech.example",))
        headers = {
            "authorization": f"Bearer {self.TOKEN}",
            "origin": "https://speech.example",
        }
        with TestClient(app) as client:
            with client.websocket_connect(REALTIME_PATH, headers=headers) as socket:
                realtime = RealtimeDriver(socket)
                realtime.send_start()
                realtime.expect("session.ready")


class TestAvailability:
    def test_a_connection_without_a_scheduler_is_refused(self) -> None:
        with TestClient(realtime_only_app()) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                error = realtime.expect_error("SERVICE_UNAVAILABLE")
                assert error["retryable"] is True
                realtime.expect_close(1013)

    def test_a_stopped_scheduler_is_treated_as_unavailable(self) -> None:
        scheduler = SchedulerDouble(running=False)
        with TestClient(realtime_only_app(scheduler)) as client:
            with client.websocket_connect(REALTIME_PATH) as socket:
                realtime = RealtimeDriver(socket)
                realtime.expect_error("SERVICE_UNAVAILABLE")
                realtime.expect_close(1013)
        assert scheduler.finals == []

    def test_the_session_limit_rejects_the_next_connection(self) -> None:
        app = realtime_only_app(SchedulerDouble(), WebSocketConfig(max_sessions=1))
        with TestClient(app) as client:
            with client.websocket_connect(REALTIME_PATH) as first:
                RealtimeDriver(first).send_start()
                RealtimeDriver(first).expect("session.ready")
                with client.websocket_connect(REALTIME_PATH) as second:
                    realtime = RealtimeDriver(second)
                    error = realtime.expect_error("SERVER_BUSY")
                    assert error["retryable"] is True
                    realtime.expect_close(1013)

    def test_a_released_session_slot_is_reusable(self) -> None:
        app = realtime_only_app(SchedulerDouble(), WebSocketConfig(max_sessions=1))
        with TestClient(app) as client:
            for _ in range(3):
                with client.websocket_connect(REALTIME_PATH) as socket:
                    realtime = RealtimeDriver(socket)
                    realtime.send_start()
                    realtime.expect("session.ready")

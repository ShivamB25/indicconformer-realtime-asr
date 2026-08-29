"""End-to-end realtime protocol tests using only the deterministic MockEngine."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.engine.mock import MockEngine
from app.main import create_app
from tests.support.audio import silence_frame, speech_frame
from tests.support.realtime import REALTIME_PATH, RealtimeDriver


@pytest.mark.parametrize(
    ("mode", "final_decoder"),
    [("latency", "ctc"), ("hybrid", "rnnt"), ("accuracy", "rnnt")],
)
def test_explicit_commit_emits_mode_specific_final(mode: str, final_decoder: str) -> None:
    app = create_app(Settings(environment="test"), engine=MockEngine())
    with TestClient(app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(language="bn", mode=mode, vad=False)
        ready = realtime.expect("session.ready")
        assert ready["session_id"]
        realtime.send_frame(silence_frame())
        realtime.send_commit()
        final = realtime.expect("transcript.final")
        assert final["language"] == "bn"
        assert final["decoder"] == final_decoder
        assert final["audio_duration_ms"] == 20
        assert final["endpoint_to_final_ms"] >= 0
        assert "mock transcript" in final["text"]


def test_vad_emits_speech_started_before_commit() -> None:
    app = create_app(Settings(environment="test"), engine=MockEngine())
    with TestClient(app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(language="hi", mode="hybrid", vad=True)
        realtime.expect("session.ready")
        realtime.send_frames([speech_frame(), speech_frame(), speech_frame()])
        assert realtime.expect("speech.started") == {"type": "speech.started"}
        realtime.send_commit()
        assert realtime.expect("transcript.final")["decoder"] == "rnnt"


def test_invalid_transport_contract_closes_cleanly() -> None:
    app = create_app(Settings(environment="test"), engine=MockEngine())
    with TestClient(app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_start(sample_rate=8_000)
        error = realtime.expect("error")
        assert error["code"] == "INVALID_SESSION"
        assert error["retryable"] is False
        assert realtime.next_message().code == 1002


def test_binary_audio_before_session_start_is_rejected() -> None:
    app = create_app(Settings(environment="test"), engine=MockEngine())
    with TestClient(app) as client, client.websocket_connect(REALTIME_PATH) as socket:
        realtime = RealtimeDriver(socket)
        realtime.send_frame(silence_frame())
        assert realtime.expect("error")["code"] == "SESSION_REQUIRED"
        assert realtime.next_message().code == 1002

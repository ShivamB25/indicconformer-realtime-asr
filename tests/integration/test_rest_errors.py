"""REST rejection and failure contracts for POST /v1/transcribe.

Every case goes through the real application over HTTP with a deterministic
engine or scheduler double, so the assertions describe the status code and body a
client actually receives rather than an internal exception type.

Response models are validated in JSON mode: the schemas are strict, so wire
values such as ``"hi"`` are only accepted when validated as JSON rather than as
already-constructed Python objects.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.schemas.rest import ErrorResponse, TranscriptionResponse
from tests.support.asgi import (
    SchedulerDouble,
    busy_scheduler,
    failing_scheduler,
    mock_engine_app,
    scheduler_app,
)
from tests.support.audio import speech_frame
from tests.support.wav import pcm_bytes_for_ms, wav_bytes

TRANSCRIBE = "/v1/transcribe"


def upload(payload: bytes, name: str = "sample.pcm") -> dict[str, tuple[str, bytes, str]]:
    return {"audio": (name, payload, "application/octet-stream")}


def post(client: TestClient, /, audio: bytes = b"", **form: Any) -> Any:
    return client.post(TRANSCRIBE, data=form, files=upload(audio or speech_frame()))


def assert_validation_error(response: Any, field: str) -> None:
    assert response.status_code == 422
    payload = ErrorResponse.model_validate_json(response.text)
    assert field in payload.error
    assert payload.request_id is None
    assert "detail" not in response.json()


class TestFormValidation:
    @pytest.mark.parametrize("language", ["en", "HI", "hi-IN", "xx", "zzz", "hindi"])
    def test_an_unsupported_language_is_a_validation_error(self, language: str) -> None:
        with TestClient(scheduler_app(SchedulerDouble())) as client:
            response = post(client, language=language)

        assert_validation_error(response, "body.language")

    def test_a_missing_language_is_a_validation_error(self) -> None:
        with TestClient(scheduler_app(SchedulerDouble())) as client:
            response = client.post(TRANSCRIBE, files=upload(speech_frame()))

        assert_validation_error(response, "body.language")

    @pytest.mark.parametrize("mode", ["fast", "HYBRID", "balanced", "latency ", "hybrid,latency"])
    def test_an_unsupported_mode_is_a_validation_error(self, mode: str) -> None:
        with TestClient(scheduler_app(SchedulerDouble())) as client:
            response = post(client, language="hi", mode=mode)

        assert_validation_error(response, "body.mode")

    def test_an_omitted_mode_falls_back_to_the_hybrid_default(self) -> None:
        with TestClient(scheduler_app(SchedulerDouble())) as client:
            response = post(client, language="hi")

        assert response.status_code == 200, response.text
        payload = TranscriptionResponse.model_validate_json(response.text)
        assert payload.mode == "hybrid"
        assert payload.decoder == "rnnt"

    def test_a_missing_audio_file_is_a_validation_error(self) -> None:
        with TestClient(scheduler_app(SchedulerDouble())) as client:
            response = client.post(TRANSCRIBE, data={"language": "hi"})

        assert_validation_error(response, "body.audio")

    def test_a_rejected_request_never_reaches_inference(self) -> None:
        scheduler = SchedulerDouble()
        with TestClient(scheduler_app(scheduler)) as client:
            assert post(client, language="en").status_code == 422

        assert scheduler.finals == []


class TestAudioDecoding:
    @pytest.mark.parametrize(
        ("case", "payload", "message"),
        [
            ("empty", b"", "audio is empty"),
            ("odd_length", b"\x00\x01\x02", "pcm_s16le audio must contain complete samples"),
            ("truncated_riff", b"RIFF", "invalid WAV container"),
            ("riff_garbage", b"RIFF____WAVEjunk", "invalid WAV container"),
        ],
    )
    def test_undecodable_audio_is_a_bad_request(
        self, case: str, payload: bytes, message: str
    ) -> None:
        del case
        scheduler = SchedulerDouble()
        with TestClient(scheduler_app(scheduler)) as client:
            response = client.post(TRANSCRIBE, data={"language": "hi"}, files=upload(payload))

        assert response.status_code == 400
        assert ErrorResponse.model_validate_json(response.text).error == message
        assert scheduler.finals == []

    @pytest.mark.parametrize(
        ("channels", "sample_rate", "sample_width", "message"),
        [
            (2, 16_000, 2, "audio must be mono"),
            (1, 8_000, 2, "audio sample rate must be 16000 Hz"),
            (1, 44_100, 2, "audio sample rate must be 16000 Hz"),
            (1, 16_000, 1, "audio must be signed 16-bit PCM"),
            (1, 16_000, 4, "audio must be signed 16-bit PCM"),
        ],
    )
    def test_wav_containers_outside_the_pcm_contract_are_refused(
        self, channels: int, sample_rate: int, sample_width: int, message: str
    ) -> None:
        container = wav_bytes(
            b"\x00" * (channels * sample_width * 320),
            channels=channels,
            sample_rate=sample_rate,
            sample_width=sample_width,
        )
        with TestClient(scheduler_app(SchedulerDouble())) as client:
            response = client.post(
                TRANSCRIBE, data={"language": "hi"}, files=upload(container, "sample.wav")
            )

        assert response.status_code == 400
        assert ErrorResponse.model_validate_json(response.text).error == message

    def test_a_conforming_wav_container_is_accepted(self) -> None:
        container = wav_bytes(pcm_bytes_for_ms(100))
        with TestClient(scheduler_app(SchedulerDouble())) as client:
            response = client.post(
                TRANSCRIBE, data={"language": "hi"}, files=upload(container, "sample.wav")
            )

        assert response.status_code == 200, response.text
        assert TranscriptionResponse.model_validate_json(response.text).audio_duration_ms == 100

    def test_headerless_pcm_of_the_same_audio_agrees_with_the_container(self) -> None:
        pcm = pcm_bytes_for_ms(200)
        with TestClient(scheduler_app(SchedulerDouble())) as client:
            headerless = client.post(TRANSCRIBE, data={"language": "hi"}, files=upload(pcm))
            contained = client.post(
                TRANSCRIBE,
                data={"language": "hi"},
                files=upload(wav_bytes(pcm), "sample.wav"),
            )

        assert headerless.json()["audio_duration_ms"] == 200
        assert contained.json()["audio_duration_ms"] == 200


class TestSizeAndDurationLimits:
    def test_an_upload_beyond_the_byte_limit_is_refused(self) -> None:
        scheduler = SchedulerDouble()
        with TestClient(scheduler_app(scheduler, max_upload_bytes=640)) as client:
            response = client.post(
                TRANSCRIBE, data={"language": "hi"}, files=upload(pcm_bytes_for_ms(40))
            )

        assert response.status_code == 413
        assert ErrorResponse.model_validate_json(response.text).error == (
            "audio upload exceeds configured limit"
        )
        assert scheduler.finals == []

    def test_an_upload_exactly_at_the_byte_limit_is_accepted(self) -> None:
        with TestClient(scheduler_app(SchedulerDouble(), max_upload_bytes=640)) as client:
            response = client.post(
                TRANSCRIBE, data={"language": "hi"}, files=upload(pcm_bytes_for_ms(20))
            )

        assert response.status_code == 200, response.text

    def test_audio_longer_than_the_duration_limit_is_refused(self) -> None:
        scheduler = SchedulerDouble()
        with TestClient(scheduler_app(scheduler, max_audio_seconds=1)) as client:
            response = client.post(
                TRANSCRIBE, data={"language": "hi"}, files=upload(pcm_bytes_for_ms(2_000))
            )

        assert response.status_code == 413
        assert ErrorResponse.model_validate_json(response.text).error == (
            "audio duration exceeds configured limit"
        )
        assert scheduler.finals == []

    def test_audio_exactly_at_the_duration_limit_is_accepted(self) -> None:
        with TestClient(scheduler_app(SchedulerDouble(), max_audio_seconds=1)) as client:
            response = client.post(
                TRANSCRIBE, data={"language": "hi"}, files=upload(pcm_bytes_for_ms(1_000))
            )

        assert response.status_code == 200, response.text
        assert response.json()["audio_duration_ms"] == 1_000


class TestServiceAvailability:
    def test_a_request_before_startup_is_refused(self) -> None:
        scheduler = SchedulerDouble()
        client = TestClient(scheduler_app(scheduler))

        response = client.post(TRANSCRIBE, data={"language": "hi"}, files=upload(speech_frame()))

        assert response.status_code == 503
        assert ErrorResponse.model_validate_json(response.text).error == "service is not ready"
        assert scheduler.finals == []

    def test_a_full_inference_queue_is_reported_as_unavailable(self) -> None:
        with TestClient(scheduler_app(busy_scheduler())) as client:
            response = post(client, language="hi")

        assert response.status_code == 503
        assert ErrorResponse.model_validate_json(response.text).error == (
            "transcription is unavailable"
        )

    def test_an_engine_failure_is_reported_as_unavailable(self) -> None:
        with TestClient(scheduler_app(failing_scheduler())) as client:
            response = post(client, language="hi")

        assert response.status_code == 503
        assert ErrorResponse.model_validate_json(response.text).error == (
            "transcription is unavailable"
        )

    def test_inference_slower_than_the_request_timeout_is_refused(self) -> None:
        scheduler = SchedulerDouble(delay_seconds=5.0)
        with TestClient(scheduler_app(scheduler, request_timeout_seconds=0.05)) as client:
            response = post(client, language="hi")

        assert response.status_code == 503
        assert ErrorResponse.model_validate_json(response.text).error == "transcription timed out"

    def test_a_failure_body_is_an_error_response_not_a_transcript(self) -> None:
        with TestClient(scheduler_app(failing_scheduler())) as client:
            body = post(client, language="hi").json()

        assert set(body) == {"error", "request_id"}
        assert "text" not in body


class TestEngineOnlyFallback:
    """Without a scheduler the endpoint must still serve requests off the loop."""

    def test_the_engine_serves_the_request_when_no_scheduler_is_bound(self) -> None:
        app = mock_engine_app()
        with TestClient(app) as client:
            app.state.scheduler = None
            response = post(client, language="ta", mode="latency")

        assert response.status_code == 200, response.text
        payload = TranscriptionResponse.model_validate_json(response.text)
        assert payload.language == "ta"
        assert payload.decoder == "ctc"
        assert payload.audio_duration_ms == 20

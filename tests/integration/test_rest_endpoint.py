"""End-to-end REST transcription contracts through the real application."""

import pytest
from fastapi.testclient import TestClient

from app.api import rest as rest_api
from app.core.types import SUPPORTED_LANGUAGE_CODES
from tests.support.asgi import SchedulerDouble, scheduler_app
from tests.support.audio import speech_frame


@pytest.mark.parametrize(
    ("mode", "decoder"),
    [("latency", "ctc"), ("hybrid", "rnnt"), ("accuracy", "rnnt")],
)
def test_mode_selects_the_server_decoder(mode: str, decoder: str) -> None:
    scheduler = SchedulerDouble()
    with TestClient(scheduler_app(scheduler)) as client:
        response = client.post(
            "/v1/transcribe",
            data={"language": "hi", "mode": mode},
            files={"audio": ("sample.pcm", speech_frame(), "application/octet-stream")},
        )

    assert response.status_code == 200, response.text
    assert response.json()["mode"] == mode
    assert response.json()["decoder"] == decoder
    assert scheduler.finals[0].decoder == decoder


@pytest.mark.parametrize("language", sorted(SUPPORTED_LANGUAGE_CODES))
def test_every_supported_language_reaches_inference(language: str) -> None:
    scheduler = SchedulerDouble()
    with TestClient(scheduler_app(scheduler)) as client:
        response = client.post(
            "/v1/transcribe",
            data={"language": language},
            files={"audio": ("sample.pcm", speech_frame(), "application/octet-stream")},
        )

    assert response.status_code == 200, response.text
    assert response.json()["language"] == language
    assert scheduler.finals[0].language == language


def test_client_decoder_override_is_rejected() -> None:
    scheduler = SchedulerDouble()
    with TestClient(scheduler_app(scheduler)) as client:
        response = client.post(
            "/v1/transcribe",
            data={"language": "hi", "mode": "latency", "decoder": "rnnt"},
            files={"audio": ("sample.pcm", speech_frame(), "application/octet-stream")},
        )

    assert response.status_code == 400
    assert response.json()["error"] == (
        "decoder is selected by the server from mode and cannot be requested"
    )
    assert scheduler.finals == []


def test_successful_rest_request_updates_runtime_metrics() -> None:
    with TestClient(scheduler_app(SchedulerDouble())) as client:
        response = client.post(
            "/v1/transcribe",
            data={"language": "hi", "mode": "latency"},
            files={"audio": ("sample.pcm", speech_frame(), "application/octet-stream")},
        )
        metrics = client.get("/metrics")

    assert response.status_code == 200
    sample = next(
        line
        for line in metrics.text.splitlines()
        if line.startswith('asr_transcriptions_total{language="hi",mode="latency"}')
    )
    assert float(sample.rsplit(" ", 1)[1]) > 0
    assert 'asr_errors_total{code="telemetry_error"}' not in metrics.text


def test_metrics_failure_cannot_discard_a_completed_transcription() -> None:
    class FailingMetrics:
        def record_transcription(self, *_: object) -> None:
            raise ValueError("collector failed")

        def record_telemetry_failure(self) -> None:
            pass

    scheduler = SchedulerDouble()
    app = scheduler_app(scheduler)
    with TestClient(app) as client:
        app.state.metrics = FailingMetrics()
        response = client.post(
            "/v1/transcribe",
            data={"language": "hi", "mode": "latency"},
            files={"audio": ("sample.pcm", speech_frame(), "application/octet-stream")},
        )

    assert response.status_code == 200, response.text
    assert response.json()["decoder"] == "ctc"
    assert len(scheduler.finals) == 1


def test_metrics_failure_is_counted_without_discarding_the_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = scheduler_app(SchedulerDouble())
    with TestClient(app) as client:

        def fail(*_: object) -> None:
            raise ValueError("collector failed")

        monkeypatch.setattr(app.state.metrics, "record_transcription", fail)
        response = client.post(
            "/v1/transcribe",
            data={"language": "hi", "mode": "latency"},
            files={"audio": ("sample.pcm", speech_frame(), "application/octet-stream")},
        )
        metrics = client.get("/metrics")

    assert response.status_code == 200, response.text
    assert response.json()["text"]
    assert 'asr_errors_total{code="telemetry_error"} 1.0' in metrics.text


def test_metrics_and_logging_failures_cannot_discard_a_completed_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingMetrics:
        def record_transcription(self, *_: object) -> None:
            raise ValueError("collector failed")

        def record_telemetry_failure(self) -> None:
            pass

    class FailingLogger:
        def exception(self, *_: object, **__: object) -> None:
            raise RuntimeError("logger failed")

    scheduler = SchedulerDouble()
    app = scheduler_app(scheduler)
    monkeypatch.setattr(rest_api, "_LOGGER", FailingLogger())
    with TestClient(app) as client:
        app.state.metrics = FailingMetrics()
        response = client.post(
            "/v1/transcribe",
            data={"language": "hi", "mode": "latency"},
            files={"audio": ("sample.pcm", speech_frame(), "application/octet-stream")},
        )

    assert response.status_code == 200, response.text
    assert response.json()["text"]
    assert len(scheduler.finals) == 1

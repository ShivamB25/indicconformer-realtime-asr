"""Focused contracts for the OpenAI-compatible REST surface."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from app.openai_compat import MODEL_ALIAS, MODEL_ID
from app.openai_compat.schemas import OpenAIErrorEnvelope
from tests.support.asgi import SchedulerDouble, scheduler_app
from tests.support.wav import pcm_bytes_for_ms, wav_bytes

TRANSCRIPTIONS = "/v1/audio/transcriptions"


def audio_file(duration_ms: int = 200) -> dict[str, tuple[str, bytes, str]]:
    payload = wav_bytes(pcm_bytes_for_ms(duration_ms))
    return {"file": ("sample.wav", payload, "audio/wav")}


def transcription_form(**overrides: Any) -> dict[str, Any]:
    return {"model": MODEL_ID, "language": "hi", **overrides}


def assert_openai_error(response: Any, *, param: str, code: str) -> None:
    payload = OpenAIErrorEnvelope.model_validate_json(response.text)
    assert payload.error.param == param
    assert payload.error.code == code
    assert payload.error.type == "invalid_request_error"
    assert response.headers["x-request-id"]


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"response_format": "json"},
        {"stream": "false"},
        {"temperature": "0"},
    ],
)
def test_json_transcription_uses_final_rnnt_policy(extra: dict[str, str]) -> None:
    scheduler = SchedulerDouble(text="namaste")
    with TestClient(scheduler_app(scheduler)) as client:
        response = client.post(
            TRANSCRIPTIONS,
            data=transcription_form(**extra),
            files=audio_file(),
        )

    assert response.status_code == 200, response.text
    assert response.json() == {"text": "namaste"}
    assert response.headers["x-request-id"]
    assert len(scheduler.finals) == 1
    assert scheduler.finals[0].decoder == "rnnt"
    assert scheduler.finals[0].language == "hi"
    assert scheduler.finals[0].sample_rate == 16_000
    assert scheduler.finals[0].audio.ndim == 1


def test_text_response_is_plain_python_text() -> None:
    scheduler = SchedulerDouble(text="plain transcript")
    with TestClient(scheduler_app(scheduler)) as client:
        response = client.post(
            TRANSCRIPTIONS,
            data=transcription_form(response_format="text"),
            files=audio_file(),
        )

    assert response.status_code == 200, response.text
    assert response.text == "plain transcript"
    assert response.headers["content-type"].startswith("text/plain")


@pytest.mark.parametrize(
    ("override", "param", "code"),
    [
        ({"model": "whisper-1"}, "model", "model_not_found"),
        ({"language": "en"}, "language", "unsupported_language"),
        ({"response_format": "verbose_json"}, "response_format", "unsupported_value"),
        ({"stream": "true"}, "stream", "unsupported_value"),
        ({"temperature": "0.1"}, "temperature", "unsupported_value"),
        ({"prompt": "words"}, "prompt", "unsupported_parameter"),
        (
            {"timestamp_granularities[]": "word"},
            "timestamp_granularities[]",
            "unsupported_parameter",
        ),
        ({"mystery": "value"}, "mystery", "unknown_parameter"),
    ],
)
def test_unsupported_requests_never_reach_scheduler(
    override: dict[str, str], param: str, code: str
) -> None:
    scheduler = SchedulerDouble()
    with TestClient(scheduler_app(scheduler)) as client:
        response = client.post(
            TRANSCRIPTIONS,
            data=transcription_form(**override),
            files=audio_file(),
        )

    assert response.status_code in {400, 404}
    assert_openai_error(response, param=param, code=code)
    assert scheduler.finals == []


@pytest.mark.parametrize("missing", ["file", "model", "language"])
def test_required_multipart_fields_use_openai_errors(missing: str) -> None:
    scheduler = SchedulerDouble()
    data = transcription_form()
    files = audio_file()
    if missing == "file":
        files = {}
    else:
        del data[missing]
    with TestClient(scheduler_app(scheduler)) as client:
        response = client.post(TRANSCRIPTIONS, data=data, files=files)

    assert response.status_code == 400
    assert_openai_error(response, param=missing, code="missing_required_parameter")
    assert scheduler.finals == []


def test_malformed_media_is_rejected_before_inference() -> None:
    scheduler = SchedulerDouble()
    with TestClient(scheduler_app(scheduler)) as client:
        response = client.post(
            TRANSCRIPTIONS,
            data=transcription_form(),
            files={"file": ("broken.webm", b"not an audio container", "audio/webm")},
        )

    assert response.status_code == 400
    assert_openai_error(response, param="file", code="invalid_audio")
    assert scheduler.finals == []


def test_decoded_duration_is_bounded_before_inference() -> None:
    scheduler = SchedulerDouble()
    with TestClient(scheduler_app(scheduler, max_audio_seconds=1)) as client:
        response = client.post(
            TRANSCRIPTIONS,
            data=transcription_form(),
            files=audio_file(1_020),
        )

    assert response.status_code == 413
    assert_openai_error(response, param="file", code="audio_too_long")
    assert scheduler.finals == []


def test_model_list_and_retrieval_use_standard_objects() -> None:
    with TestClient(scheduler_app(SchedulerDouble())) as client:
        listing = client.get("/v1/models")
        canonical = client.get(f"/v1/models/{MODEL_ID}")
        alias = client.get(f"/v1/models/{MODEL_ALIAS}")

    assert listing.status_code == 200
    assert listing.json()["object"] == "list"
    assert listing.json()["data"] == [canonical.json()]
    assert canonical.json() == {
        "id": MODEL_ID,
        "object": "model",
        "created": 0,
        "owned_by": "ai4bharat",
    }
    assert alias.json() == canonical.json()
    assert listing.headers["x-request-id"]
    assert canonical.headers["x-request-id"]


def test_unmodified_openai_client_transcribes_and_lists_models() -> None:
    scheduler = SchedulerDouble(text="sdk transcript")
    with TestClient(scheduler_app(scheduler)) as client:
        sdk = OpenAI(
            api_key="local-test-key",
            base_url="http://testserver/v1",
            http_client=client,  # type: ignore[arg-type]  # httpx2 sync-client compatible
        )
        transcript = sdk.audio.transcriptions.create(
            file=("sample.wav", wav_bytes(pcm_bytes_for_ms(200)), "audio/wav"),
            model=MODEL_ID,
            language="hi",
        )
        plain_text = sdk.audio.transcriptions.create(
            file=("sample.wav", wav_bytes(pcm_bytes_for_ms(200)), "audio/wav"),
            model=MODEL_ALIAS,
            language="hi",
            response_format="text",
        )
        models = sdk.models.list()
        retrieved = sdk.models.retrieve(MODEL_ID)

    assert transcript.text == "sdk transcript"
    assert plain_text == "sdk transcript"
    assert [model.id for model in models.data] == [MODEL_ID]
    assert retrieved.id == MODEL_ID

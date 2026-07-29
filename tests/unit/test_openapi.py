"""Generated OpenAPI contracts that keep Swagger useful for real clients."""

from __future__ import annotations

from typing import Any

from app.main import create_app
from tests.support.asgi import settings_for_tests


def schema() -> dict[str, Any]:
    return create_app(settings_for_tests()).openapi()


def test_service_metadata_and_tag_guidance_are_published() -> None:
    document = schema()

    assert document["info"]["title"] == "AI4Bharat IndicConformer ASR"
    assert "Try a transcription in Swagger" in document["info"]["description"]
    assert [tag["name"] for tag in document["tags"]] == [
        "transcription",
        "openai",
        "health",
    ]
    assert {entry["url"] for entry in document["x-websocket-endpoints"]} == {
        "/v1/realtime",
        "/v1/realtime/transcription_sessions",
    }


def test_swagger_authorize_controls_only_inference_http_operations() -> None:
    document = schema()
    expected: list[dict[str, list[Any]]] = [{"BearerAuth": []}, {}]

    bearer = document["components"]["securitySchemes"]["BearerAuth"]
    assert bearer["type"] == "http"
    assert bearer["scheme"] == "bearer"
    for path in (
        "/v1/transcribe",
        "/v1/audio/transcriptions",
        "/v1/models",
        "/v1/models/{model}",
    ):
        operations = document["paths"][path]
        assert all(operation["security"] == expected for operation in operations.values())
    assert "security" not in document["paths"]["/health/live"]["get"]
    assert "security" not in document["paths"]["/health/ready"]["get"]


def test_native_transcription_is_a_complete_swagger_upload_form() -> None:
    document = schema()
    operation = document["paths"]["/v1/transcribe"]["post"]
    reference = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    body = document["components"]["schemas"][reference.rsplit("/", 1)[1]]

    assert operation["summary"] == "Transcribe mono 16 kHz PCM audio"
    assert body["required"] == ["audio", "language"]
    assert set(body["properties"]) == {"audio", "language", "mode"}
    assert body["properties"]["audio"]["contentMediaType"] == "application/octet-stream"
    assert "16 kHz" in body["properties"]["audio"]["description"]
    assert body["properties"]["mode"]["default"] == "hybrid"
    assert "401" in operation["responses"]


def test_openai_transcription_is_a_complete_swagger_upload_form() -> None:
    document = schema()
    operation = document["paths"]["/v1/audio/transcriptions"]["post"]
    body = operation["requestBody"]["content"]["multipart/form-data"]["schema"]

    assert body["required"] == ["file", "model", "language"]
    assert set(body["properties"]) == {
        "file",
        "model",
        "language",
        "response_format",
        "stream",
        "temperature",
    }
    assert body["properties"]["file"]["format"] == "binary"
    assert body["properties"]["model"]["default"].startswith("ai4bharat/")
    assert body["properties"]["response_format"]["enum"] == ["json", "text"]
    assert "text/plain" in operation["responses"]["200"]["content"]


def test_health_operations_explain_probe_semantics() -> None:
    document = schema()
    live = document["paths"]["/health/live"]["get"]
    ready = document["paths"]["/health/ready"]["get"]

    assert live["summary"] == "Check process liveness"
    assert "never loads a model" in live["description"]
    assert ready["summary"] == "Check model and scheduler readiness"
    assert ready["responses"]["503"]["description"] == (
        "One or more startup components are not ready"
    )

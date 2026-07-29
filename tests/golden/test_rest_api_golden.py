"""Golden HTTP surface: the paths, methods, status codes, and body field names.

The fixture records the service's public shape only. It is checked against the
generated OpenAPI document and against live responses served by the MockEngine,
so renaming a route or a response field breaks this test rather than silently
breaking clients. Routes are read from the schema rather than from the router
objects, because the latter are a private FastAPI detail.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.readiness import CheckStatus
from app.schemas.rest import ErrorResponse, ReadyResponse, TranscriptionResponse
from tests.support.asgi import SchedulerDouble, scheduler_app
from tests.support.audio import speech_frame
from tests.support.golden import load_golden
from tests.support.realtime import REALTIME_PATH, RealtimeDriver

GOLDEN = load_golden("rest_api.json")
ENDPOINTS: dict[str, Any] = GOLDEN["endpoints"]
HTTP_PATHS = sorted(path for path, spec in ENDPOINTS.items() if "method" in spec)
WEBSOCKET_PATHS = sorted(
    path for path, spec in ENDPOINTS.items() if spec.get("transport") == "websocket"
)
UPLOAD = {"audio": ("sample.pcm", speech_frame(), "application/octet-stream")}


@pytest.fixture(name="client")
def client_fixture() -> Iterator[TestClient]:
    with TestClient(scheduler_app(SchedulerDouble())) as client:
        yield client


@pytest.fixture(name="schema", scope="module")
def schema_fixture() -> dict[str, Any]:
    document: dict[str, Any] = scheduler_app(SchedulerDouble()).openapi()
    return document


class TestPublishedSurface:
    def test_the_recorded_paths_are_not_empty(self) -> None:
        assert HTTP_PATHS == ["/health/live", "/health/ready", "/v1/transcribe"]
        assert WEBSOCKET_PATHS == [REALTIME_PATH]

    @pytest.mark.parametrize("path", HTTP_PATHS)
    def test_every_recorded_path_is_published_with_its_method(
        self, path: str, schema: dict[str, Any]
    ) -> None:
        method = str(ENDPOINTS[path]["method"]).lower()

        assert path in schema["paths"], f"{path} is not published"
        assert method in schema["paths"][path]

    @pytest.mark.parametrize("path", HTTP_PATHS)
    def test_every_recorded_success_status_is_declared(
        self, path: str, schema: dict[str, Any]
    ) -> None:
        method = str(ENDPOINTS[path]["method"]).lower()
        declared = {int(code) for code in schema["paths"][path][method]["responses"]}

        assert ENDPOINTS[path]["success_status"] in declared

    def test_no_undocumented_transcription_path_appears(self, schema: dict[str, Any]) -> None:
        versioned = sorted(path for path in schema["paths"] if path.startswith("/v1"))

        assert versioned == ["/v1/transcribe"]

    def test_the_realtime_path_is_not_an_http_endpoint(
        self, client: TestClient, schema: dict[str, Any]
    ) -> None:
        assert REALTIME_PATH not in schema["paths"]
        assert client.get(REALTIME_PATH).status_code in {404, 405}

    def test_the_realtime_path_accepts_a_websocket_session(self, client: TestClient) -> None:
        with client.websocket_connect(REALTIME_PATH) as socket:
            realtime = RealtimeDriver(socket)
            realtime.send_start(language="hi")
            assert realtime.expect("session.ready")["session_id"]


class TestLiveBodies:
    def test_the_liveness_body_matches_the_golden_fields(self, client: TestClient) -> None:
        spec = ENDPOINTS["/health/live"]
        response = client.get("/health/live")

        assert response.status_code == spec["success_status"]
        assert sorted(response.json()) == sorted(spec["fields"])
        assert response.json()["status"] == spec["status_value"]

    def test_the_readiness_body_matches_the_golden_fields(self, client: TestClient) -> None:
        spec = ENDPOINTS["/health/ready"]
        response = client.get("/health/ready")

        assert response.status_code == spec["success_status"]
        body = response.json()
        assert sorted(body) == sorted(spec["fields"])
        assert body["status"] in spec["status_values"]
        assert sorted(body["checks"]) == sorted(spec["check_names"])
        assert set(body["checks"].values()) <= set(spec["check_values"])
        ReadyResponse.model_validate_json(response.text)

    def test_the_recorded_check_values_are_exactly_the_runtime_statuses(self) -> None:
        assert sorted(ENDPOINTS["/health/ready"]["check_values"]) == sorted(
            status.value for status in CheckStatus
        )

    def test_the_transcription_body_matches_the_golden_fields(self, client: TestClient) -> None:
        spec = ENDPOINTS["/v1/transcribe"]
        response = client.post("/v1/transcribe", data={"language": "hi"}, files=UPLOAD)

        assert response.status_code == spec["success_status"]
        assert sorted(response.json()) == sorted(spec["fields"])
        TranscriptionResponse.model_validate_json(response.text)

    def test_the_recorded_transcription_fields_are_the_schema_fields(self) -> None:
        assert sorted(ENDPOINTS["/v1/transcribe"]["fields"]) == sorted(
            TranscriptionResponse.model_fields
        )

    def test_the_error_body_matches_the_golden_fields(self, client: TestClient) -> None:
        response = client.post(
            "/v1/transcribe",
            data={"language": "hi"},
            files={"audio": ("empty.pcm", b"", "application/octet-stream")},
        )

        assert response.status_code == 400
        assert sorted(response.json()) == sorted(GOLDEN["error_fields"])
        ErrorResponse.model_validate_json(response.text)

    def test_the_recorded_error_fields_are_the_schema_fields(self) -> None:
        assert sorted(GOLDEN["error_fields"]) == sorted(ErrorResponse.model_fields)


class TestRecordedErrorStatuses:
    """Each recorded rejection status must be reachable through the real endpoint."""

    def test_a_bad_request_status_is_reachable(self, client: TestClient) -> None:
        response = client.post(
            "/v1/transcribe",
            data={"language": "hi"},
            files={"audio": ("odd.pcm", b"\x01", "application/octet-stream")},
        )

        assert response.status_code == 400
        assert 400 in ENDPOINTS["/v1/transcribe"]["error_statuses"]

    def test_a_payload_too_large_status_is_reachable(self) -> None:
        with TestClient(scheduler_app(SchedulerDouble(), max_upload_bytes=2)) as client:
            response = client.post("/v1/transcribe", data={"language": "hi"}, files=UPLOAD)

        assert response.status_code == 413
        assert 413 in ENDPOINTS["/v1/transcribe"]["error_statuses"]

    def test_a_validation_status_is_reachable(self, client: TestClient) -> None:
        response = client.post("/v1/transcribe", data={"language": "xx"}, files=UPLOAD)

        assert response.status_code == 422
        assert 422 in ENDPOINTS["/v1/transcribe"]["error_statuses"]

    def test_an_unavailable_status_is_reachable(self) -> None:
        client = TestClient(scheduler_app(SchedulerDouble()))

        response = client.post("/v1/transcribe", data={"language": "hi"}, files=UPLOAD)

        assert response.status_code == 503
        assert 503 in ENDPOINTS["/v1/transcribe"]["error_statuses"]


class TestUploadEncodings:
    def test_the_recorded_encodings_are_the_two_the_endpoint_decodes(self) -> None:
        assert GOLDEN["accepted_upload_encodings"] == ["pcm_s16le", "wav_pcm_s16le_mono_16k"]

    def test_headerless_pcm_is_accepted(self, client: TestClient) -> None:
        response = client.post("/v1/transcribe", data={"language": "hi"}, files=UPLOAD)

        assert response.status_code == 200
        assert response.json()["audio_duration_ms"] == 20

"""Contract tests for the REST request options and response bodies."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.readiness import CheckStatus, ReadinessTracker
from app.core.types import Decoder, LanguageCode, ProcessingMode
from app.schemas.rest import (
    ErrorResponse,
    LiveResponse,
    ReadyResponse,
    TranscriptionOptions,
    TranscriptionResponse,
)

ALL_LANGUAGES = [code.value for code in LanguageCode]


def error_types(exc: ValidationError) -> set[str]:
    return {error["type"] for error in exc.errors()}


def error_locations(exc: ValidationError) -> set[tuple[int | str, ...]]:
    return {error["loc"] for error in exc.errors()}


class TestTranscriptionOptions:
    def test_default_mode_is_hybrid(self) -> None:
        options = TranscriptionOptions.model_validate_json('{"language": "hi"}')
        assert options.mode is ProcessingMode.HYBRID

    @pytest.mark.parametrize("language", ALL_LANGUAGES)
    def test_every_language_is_accepted(self, language: str) -> None:
        options = TranscriptionOptions.model_validate_json(json.dumps({"language": language}))
        assert options.language is LanguageCode(language)

    def test_language_is_required(self) -> None:
        with pytest.raises(ValidationError) as caught:
            TranscriptionOptions.model_validate_json("{}")
        assert error_locations(caught.value) == {("language",)}
        assert error_types(caught.value) == {"missing"}

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("language", "en"),
            ("language", "HI"),
            ("language", ""),
            ("mode", "fast"),
            ("mode", "HYBRID"),
        ],
    )
    def test_values_outside_the_closed_sets_are_rejected(self, field: str, value: str) -> None:
        payload: dict[str, Any] = {"language": "hi"}
        payload[field] = value
        with pytest.raises(ValidationError) as caught:
            TranscriptionOptions.model_validate_json(json.dumps(payload))
        assert error_locations(caught.value) == {(field,)}
        assert error_types(caught.value) == {"enum"}

    @pytest.mark.parametrize("field", ["beam_size", "temperature", "prompt", "decoder", "decoders"])
    def test_unknown_options_are_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError) as caught:
            TranscriptionOptions.model_validate_json(json.dumps({"language": "hi", field: "x"}))
        assert error_locations(caught.value) == {(field,)}
        assert error_types(caught.value) == {"extra_forbidden"}

    def test_strict_python_mode_needs_enum_instances(self) -> None:
        """Form/query strings must be coerced before constructing options."""

        with pytest.raises(ValidationError) as caught:
            TranscriptionOptions.model_validate({"language": "hi"})
        assert error_types(caught.value) == {"is_instance_of"}

        options = TranscriptionOptions.model_validate(
            {
                "language": LanguageCode.HI,
                "mode": ProcessingMode.ACCURACY,
            }
        )
        assert options.mode is ProcessingMode.ACCURACY


class TestTranscriptionResponse:
    def valid_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": "sample text",
            "language": "hi",
            "mode": "hybrid",
            "decoder": "ctc",
            "audio_duration_ms": 1_000,
            "inference_ms": 12.5,
            "request_id": "req-1",
        }
        payload.update(overrides)
        return payload

    def test_round_trip_is_lossless(self) -> None:
        payload = self.valid_payload()
        response = TranscriptionResponse.model_validate_json(json.dumps(payload))
        assert json.loads(response.model_dump_json()) == payload

    def test_empty_transcript_is_representable(self) -> None:
        response = TranscriptionResponse.model_validate_json(
            json.dumps(self.valid_payload(text="", audio_duration_ms=0, inference_ms=0.0))
        )
        assert response.text == ""
        assert response.audio_duration_ms == 0

    @pytest.mark.parametrize(
        ("field", "value", "expected_type"),
        [
            ("audio_duration_ms", -1, "greater_than_equal"),
            ("audio_duration_ms", 1.5, "int_type"),
            ("inference_ms", -0.1, "greater_than_equal"),
            ("inference_ms", "12", "float_type"),
            ("request_id", 7, "string_type"),
            ("text", None, "string_type"),
        ],
    )
    def test_field_constraints(self, field: str, value: object, expected_type: str) -> None:
        with pytest.raises(ValidationError) as caught:
            TranscriptionResponse.model_validate_json(
                json.dumps(self.valid_payload(**{field: value}))
            )
        assert error_locations(caught.value) == {(field,)}
        assert error_types(caught.value) == {expected_type}

    @pytest.mark.parametrize(
        "field", ["text", "language", "mode", "decoder", "audio_duration_ms", "request_id"]
    )
    def test_every_documented_field_is_required(self, field: str) -> None:
        payload = self.valid_payload()
        del payload[field]
        with pytest.raises(ValidationError) as caught:
            TranscriptionResponse.model_validate_json(json.dumps(payload))
        assert error_locations(caught.value) == {(field,)}

    def test_response_never_leaks_extra_fields(self) -> None:
        with pytest.raises(ValidationError) as caught:
            TranscriptionResponse.model_validate_json(
                json.dumps(self.valid_payload(model_path="/srv/models/x"))
            )
        assert error_types(caught.value) == {"extra_forbidden"}

    def test_mode_and_decoder_are_reported_independently(self) -> None:
        response = TranscriptionResponse.model_validate_json(
            json.dumps(self.valid_payload(mode="accuracy", decoder="rnnt"))
        )
        assert response.mode is ProcessingMode.ACCURACY
        assert response.decoder is Decoder.RNNT


class TestErrorResponse:
    def test_request_id_is_optional(self) -> None:
        response = ErrorResponse.model_validate_json('{"error": "bad request"}')
        assert response.request_id is None
        assert json.loads(response.model_dump_json()) == {
            "error": "bad request",
            "request_id": None,
        }

    def test_error_message_is_required(self) -> None:
        with pytest.raises(ValidationError) as caught:
            ErrorResponse.model_validate_json('{"request_id": "r"}')
        assert error_locations(caught.value) == {("error",)}

    def test_error_body_cannot_carry_diagnostics(self) -> None:
        with pytest.raises(ValidationError) as caught:
            ErrorResponse.model_validate_json('{"error": "x", "traceback": "..."}')
        assert error_types(caught.value) == {"extra_forbidden"}


class TestHealthResponses:
    def test_live_response_defaults_to_live(self) -> None:
        assert json.loads(LiveResponse().model_dump_json()) == {"status": "live"}

    def test_live_response_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError) as caught:
            LiveResponse.model_validate_json('{"status": "live", "uptime_s": 3}')
        assert error_types(caught.value) == {"extra_forbidden"}

    def test_ready_response_round_trip(self) -> None:
        payload = {
            "status": "ready",
            "stage": "serving",
            "checks": {"engine": "ready", "scheduler": "ready"},
            "detail": None,
        }
        response = ReadyResponse.model_validate_json(json.dumps(payload))
        assert json.loads(response.model_dump_json()) == payload

    def test_ready_response_detail_is_optional(self) -> None:
        response = ReadyResponse.model_validate_json(
            '{"status": "starting", "stage": "loading_model", "checks": {}}'
        )
        assert response.detail is None
        assert response.checks == {}

    @pytest.mark.parametrize("value", [1, None, True, ["ready"]])
    def test_check_values_must_be_strings(self, value: object) -> None:
        with pytest.raises(ValidationError) as caught:
            ReadyResponse.model_validate_json(
                json.dumps({"status": "ready", "stage": "s", "checks": {"engine": value}})
            )
        assert error_locations(caught.value) == {("checks", "engine")}

    def test_a_readiness_snapshot_can_be_serialized_directly(self) -> None:
        """The health route must be able to publish a tracker snapshot as-is."""

        tracker = ReadinessTracker()
        tracker.update(
            stage="serving",
            engine=CheckStatus.READY,
            scheduler=CheckStatus.DISABLED,
        )
        snapshot = tracker.snapshot()

        response = ReadyResponse(
            status="ready" if snapshot.ready else "starting",
            stage=snapshot.stage,
            checks=snapshot.checks,
            detail=snapshot.detail,
        )
        assert json.loads(response.model_dump_json()) == {
            "status": "ready",
            "stage": "serving",
            "checks": {"engine": "ready", "scheduler": "disabled"},
            "detail": None,
        }

    @pytest.mark.parametrize("status", [status.value for status in CheckStatus])
    def test_every_check_status_is_representable(self, status: str) -> None:
        response = ReadyResponse.model_validate_json(
            json.dumps({"status": "starting", "stage": "boot", "checks": {"engine": status}})
        )
        assert response.checks == {"engine": status}

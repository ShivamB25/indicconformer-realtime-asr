"""Contract tests for the realtime control-event schemas.

The wire format is JSON text, so validation is exercised the way the server
must do it: on the raw bytes/str. ``strict=True`` makes JSON-mode and
Python-mode behaviour differ sharply, and that difference is pinned here
because it decides whether a real client can connect at all.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from app.core.types import Decoder, LanguageCode, ProcessingMode
from app.schemas.protocol import (
    ClientEvent,
    InputCommitEvent,
    ProtocolErrorEvent,
    ServerEvent,
    SessionReadyEvent,
    SessionStartEvent,
    SpeechStartedEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
)

CLIENT_EVENTS: TypeAdapter[Any] = TypeAdapter(ClientEvent)
SERVER_EVENTS: TypeAdapter[Any] = TypeAdapter(ServerEvent)

EXPECTED_LANGUAGE_CODES = (
    "as",
    "bn",
    "brx",
    "doi",
    "gu",
    "hi",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
)


def error_types(exc: ValidationError) -> set[str]:
    return {error["type"] for error in exc.errors()}


def error_locations(exc: ValidationError) -> set[tuple[int | str, ...]]:
    return {error["loc"] for error in exc.errors()}


def start_json(**fields: Any) -> str:
    payload: dict[str, Any] = {"type": "session.start", "language": "hi"}
    payload.update(fields)
    return json.dumps(payload)


class TestLanguageCoverage:
    def test_language_enum_is_exactly_the_twenty_two_supported_codes(self) -> None:
        assert tuple(code.value for code in LanguageCode) == EXPECTED_LANGUAGE_CODES
        assert len(EXPECTED_LANGUAGE_CODES) == 22

    @pytest.mark.parametrize("language", EXPECTED_LANGUAGE_CODES)
    def test_every_language_is_accepted_and_preserved(self, language: str) -> None:
        event = SessionStartEvent.model_validate_json(start_json(language=language))
        assert event.language is LanguageCode(language)
        assert json.loads(event.model_dump_json())["language"] == language

    @pytest.mark.parametrize("language", EXPECTED_LANGUAGE_CODES)
    def test_every_language_survives_a_final_transcript_round_trip(self, language: str) -> None:
        payload = {
            "type": "transcript.final",
            "text": "",
            "language": language,
            "decoder": "ctc",
            "audio_duration_ms": 0,
            "endpoint_to_final_ms": 0.0,
        }
        event = TranscriptFinalEvent.model_validate_json(json.dumps(payload))
        assert event.language is LanguageCode(language)
        assert json.loads(event.model_dump_json()) == payload

    @pytest.mark.parametrize(
        "language",
        ["en", "HI", "hi ", " hi", "hin", "", "zz", "hi-IN", "as_IN", "*"],
    )
    def test_unsupported_language_codes_are_rejected(self, language: str) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate_json(start_json(language=language))
        assert error_locations(caught.value) == {("language",)}
        assert error_types(caught.value) == {"enum"}

    def test_language_is_required(self) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate_json('{"type": "session.start"}')
        assert error_types(caught.value) == {"missing"}


class TestSessionStartAudioContract:
    def test_defaults_are_the_documented_pcm_contract(self) -> None:
        event = SessionStartEvent.model_validate_json('{"language": "hi"}')
        assert json.loads(event.model_dump_json()) == {
            "type": "session.start",
            "language": "hi",
            "format": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "mode": "hybrid",
            "vad": True,
        }

    @pytest.mark.parametrize("audio_format", ["wav", "pcm_s16be", "pcm_f32le", "opus", ""])
    def test_only_pcm_s16le_is_accepted(self, audio_format: str) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate_json(start_json(format=audio_format))
        assert error_locations(caught.value) == {("format",)}
        assert error_types(caught.value) == {"literal_error"}

    @pytest.mark.parametrize("sample_rate", [8000, 22050, 44100, 48000, 0, -16000, "16000"])
    def test_only_sixteen_kilohertz_is_accepted(self, sample_rate: object) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate_json(start_json(sample_rate=sample_rate))
        assert error_locations(caught.value) == {("sample_rate",)}
        expected = "literal_error" if isinstance(sample_rate, int) else "value_error"
        assert error_types(caught.value) == {expected}

    @pytest.mark.parametrize("channels", [0, 2, 6, -1, "1"])
    def test_only_mono_is_accepted(self, channels: object) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate_json(start_json(channels=channels))
        assert error_locations(caught.value) == {("channels",)}
        expected = "literal_error" if isinstance(channels, int) else "value_error"
        assert error_types(caught.value) == {expected}

    @pytest.mark.parametrize("mode", [mode.value for mode in ProcessingMode])
    def test_every_processing_mode_is_accepted(self, mode: str) -> None:
        event = SessionStartEvent.model_validate_json(start_json(mode=mode))
        assert event.mode is ProcessingMode(mode)

    def test_processing_mode_set_is_closed(self) -> None:
        assert [mode.value for mode in ProcessingMode] == ["latency", "hybrid", "accuracy"]

    @pytest.mark.parametrize("mode", ["fast", "LATENCY", "balanced", "", "accuracy "])
    def test_unknown_processing_modes_are_rejected(self, mode: str) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate_json(start_json(mode=mode))
        assert error_locations(caught.value) == {("mode",)}
        assert error_types(caught.value) == {"enum"}

    @pytest.mark.parametrize("vad", [1, 0, "true", None, "yes"])
    def test_vad_must_be_a_real_boolean(self, vad: object) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate_json(start_json(vad=vad))
        assert error_locations(caught.value) == {("vad",)}
        assert error_types(caught.value) == {"bool_type"}

    def test_vad_can_be_disabled_explicitly(self) -> None:
        assert SessionStartEvent.model_validate_json(start_json(vad=False)).vad is False

    @pytest.mark.parametrize("field", ["language_code", "sampleRate", "codec", "session_id"])
    def test_unknown_fields_are_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate_json(start_json(**{field: "x"}))
        assert error_locations(caught.value) == {(field,)}
        assert error_types(caught.value) == {"extra_forbidden"}


class TestStrictValidationMode:
    """Strict schemas only accept wire values through JSON-mode validation.

    A handler that does ``json.loads`` before ``model_validate`` rejects every
    valid client handshake, so the difference is asserted rather than assumed.
    """

    def test_json_mode_accepts_wire_strings(self) -> None:
        event = CLIENT_EVENTS.validate_json(
            '{"type": "session.start", "language": "ta", "mode": "accuracy"}'
        )
        assert isinstance(event, SessionStartEvent)
        assert event.language is LanguageCode.TA
        assert event.mode is ProcessingMode.ACCURACY

    def test_python_mode_requires_enum_instances(self) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionStartEvent.model_validate({"type": "session.start", "language": "hi"})
        assert error_types(caught.value) == {"is_instance_of"}

        event = SessionStartEvent.model_validate(
            {
                "type": "session.start",
                "language": LanguageCode.HI,
                "mode": ProcessingMode.LATENCY,
            }
        )
        assert event.language is LanguageCode.HI


class TestClientEventUnion:
    def test_session_start_is_selected_by_its_discriminator(self) -> None:
        event = CLIENT_EVENTS.validate_json(start_json())
        assert isinstance(event, SessionStartEvent)

    def test_input_commit_needs_no_other_field(self) -> None:
        event = CLIENT_EVENTS.validate_json('{"type": "input.commit"}')
        assert isinstance(event, InputCommitEvent)

    def test_input_commit_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError) as caught:
            CLIENT_EVENTS.validate_json('{"type": "input.commit", "flush": true}')
        assert error_types(caught.value) == {"extra_forbidden"}

    @pytest.mark.parametrize(
        "event_type",
        ["session.ready", "transcript.final", "session.stop", "", "SESSION.START"],
    )
    def test_server_and_unknown_event_types_are_not_client_events(self, event_type: str) -> None:
        with pytest.raises(ValidationError) as caught:
            CLIENT_EVENTS.validate_json(json.dumps({"type": event_type}))
        assert error_types(caught.value) == {"union_tag_invalid"}

    def test_missing_discriminator_is_reported_as_such(self) -> None:
        with pytest.raises(ValidationError) as caught:
            CLIENT_EVENTS.validate_json('{"language": "hi"}')
        assert error_types(caught.value) == {"union_tag_not_found"}

    @pytest.mark.parametrize("text", ["[]", '"session.start"', "17", "null"])
    def test_non_object_payloads_are_rejected(self, text: str) -> None:
        with pytest.raises(ValidationError) as caught:
            CLIENT_EVENTS.validate_json(text)
        assert error_types(caught.value) == {"dict_type"}

    @pytest.mark.parametrize("text", ["", "{", "{'type': 'input.commit'}", "not json"])
    def test_malformed_json_is_a_validation_error(self, text: str) -> None:
        with pytest.raises(ValidationError):
            CLIENT_EVENTS.validate_json(text)


class TestServerEvents:
    def test_session_ready_carries_a_session_id(self) -> None:
        event = SERVER_EVENTS.validate_json('{"type": "session.ready", "session_id": "abc123"}')
        assert isinstance(event, SessionReadyEvent)
        assert event.session_id == "abc123"

    def test_session_ready_requires_a_session_id(self) -> None:
        with pytest.raises(ValidationError) as caught:
            SessionReadyEvent.model_validate_json('{"type": "session.ready"}')
        assert error_types(caught.value) == {"missing"}

    def test_speech_started_is_a_bare_marker(self) -> None:
        event = SERVER_EVENTS.validate_json('{"type": "speech.started"}')
        assert isinstance(event, SpeechStartedEvent)
        assert json.loads(event.model_dump_json()) == {"type": "speech.started"}

    def test_speech_started_carries_no_payload(self) -> None:
        with pytest.raises(ValidationError) as caught:
            SpeechStartedEvent.model_validate_json('{"type": "speech.started", "audio_ms": 60}')
        assert error_types(caught.value) == {"extra_forbidden"}

    def test_partial_serializes_text_revision_and_stability(self) -> None:
        event = TranscriptPartialEvent(text="ka kha", revision=4, is_stable=True)
        assert json.loads(event.model_dump_json()) == {
            "type": "transcript.partial",
            "text": "ka kha",
            "revision": 4,
            "is_stable": True,
        }

    @pytest.mark.parametrize("revision", [-1, -100])
    def test_partial_revision_cannot_be_negative(self, revision: int) -> None:
        with pytest.raises(ValidationError) as caught:
            TranscriptPartialEvent.model_validate_json(
                json.dumps({"text": "x", "revision": revision, "is_stable": False})
            )
        assert error_locations(caught.value) == {("revision",)}
        assert error_types(caught.value) == {"greater_than_equal"}

    def test_partial_revision_must_be_an_integer(self) -> None:
        with pytest.raises(ValidationError) as caught:
            TranscriptPartialEvent.model_validate_json(
                '{"text": "x", "revision": 1.5, "is_stable": false}'
            )
        assert error_types(caught.value) == {"int_type"}

    def test_partial_stability_flag_is_required_and_boolean(self) -> None:
        with pytest.raises(ValidationError) as caught:
            TranscriptPartialEvent.model_validate_json('{"text": "x", "revision": 0}')
        assert error_locations(caught.value) == {("is_stable",)}

        with pytest.raises(ValidationError) as caught:
            TranscriptPartialEvent.model_validate_json(
                '{"text": "x", "revision": 0, "is_stable": 1}'
            )
        assert error_types(caught.value) == {"bool_type"}

    @pytest.mark.parametrize("decoder", [decoder.value for decoder in Decoder])
    def test_final_accepts_both_decoders(self, decoder: str) -> None:
        event = TranscriptFinalEvent.model_validate_json(
            json.dumps(
                {
                    "text": "t",
                    "language": "hi",
                    "decoder": decoder,
                    "audio_duration_ms": 20,
                    "endpoint_to_final_ms": 1.0,
                }
            )
        )
        assert event.decoder is Decoder(decoder)

    def test_decoder_set_is_closed(self) -> None:
        assert [decoder.value for decoder in Decoder] == ["ctc", "rnnt"]

    @pytest.mark.parametrize("decoder", ["greedy", "beam", "CTC", "rnn-t", ""])
    def test_final_rejects_unknown_decoders(self, decoder: str) -> None:
        with pytest.raises(ValidationError) as caught:
            TranscriptFinalEvent.model_validate_json(
                json.dumps(
                    {
                        "text": "t",
                        "language": "hi",
                        "decoder": decoder,
                        "audio_duration_ms": 20,
                        "endpoint_to_final_ms": 1.0,
                    }
                )
            )
        assert error_locations(caught.value) == {("decoder",)}
        assert error_types(caught.value) == {"enum"}

    @pytest.mark.parametrize(
        ("field", "value", "expected_type"),
        [
            ("audio_duration_ms", -1, "greater_than_equal"),
            ("audio_duration_ms", 20.5, "int_type"),
            ("endpoint_to_final_ms", -0.5, "greater_than_equal"),
            ("endpoint_to_final_ms", "12", "float_type"),
        ],
    )
    def test_final_timing_fields_are_bounded(
        self, field: str, value: object, expected_type: str
    ) -> None:
        payload: dict[str, Any] = {
            "text": "t",
            "language": "hi",
            "decoder": "ctc",
            "audio_duration_ms": 20,
            "endpoint_to_final_ms": 1.0,
        }
        payload[field] = value
        with pytest.raises(ValidationError) as caught:
            TranscriptFinalEvent.model_validate_json(json.dumps(payload))
        assert error_locations(caught.value) == {(field,)}
        assert error_types(caught.value) == {expected_type}

    def test_final_accepts_an_integral_endpoint_latency(self) -> None:
        event = TranscriptFinalEvent.model_validate_json(
            '{"text": "t", "language": "hi", "decoder": "ctc",'
            ' "audio_duration_ms": 20, "endpoint_to_final_ms": 2}'
        )
        assert event.endpoint_to_final_ms == pytest.approx(2.0)

    def test_error_event_defaults_to_not_retryable(self) -> None:
        event = ProtocolErrorEvent.model_validate_json(
            '{"code": "INVALID_FRAME_SIZE", "message": "bad frame"}'
        )
        assert event.retryable is False
        assert json.loads(event.model_dump_json()) == {
            "type": "error",
            "code": "INVALID_FRAME_SIZE",
            "message": "bad frame",
            "retryable": False,
        }

    def test_error_event_retryable_flag_round_trips(self) -> None:
        event = ProtocolErrorEvent.model_validate_json(
            '{"code": "SERVER_BUSY", "message": "queue full", "retryable": true}'
        )
        assert event.retryable is True

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "transcript.partial", "text": "x", "revision": 0, "is_stable": False},
            {"type": "session.ready", "session_id": "s"},
            {"type": "speech.started"},
            {"type": "error", "code": "C", "message": "m", "retryable": True},
            {
                "type": "transcript.final",
                "text": "t",
                "language": "hi",
                "decoder": "rnnt",
                "audio_duration_ms": 40,
                "endpoint_to_final_ms": 3.5,
            },
        ],
    )
    def test_every_server_event_round_trips_through_the_union(
        self, payload: dict[str, Any]
    ) -> None:
        event = SERVER_EVENTS.validate_json(json.dumps(payload))
        assert json.loads(SERVER_EVENTS.dump_json(event)) == payload

    def test_client_events_are_not_server_events(self) -> None:
        with pytest.raises(ValidationError) as caught:
            SERVER_EVENTS.validate_json('{"type": "input.commit"}')
        assert error_types(caught.value) == {"union_tag_invalid"}

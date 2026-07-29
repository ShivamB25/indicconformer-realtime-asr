"""Golden wire contract for the realtime protocol.

The fixture records field names, defaults, closed sets, and frame geometry, so a
rename, a new required field, a silently changed default, or an undocumented
error code fails here rather than in a client integration months later. It
contains no audio and no transcription output.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from app.audio.pcm import (
    BYTES_PER_SAMPLE,
    FRAME_DURATION_MS,
    PCM16_FRAME_BYTES,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)
from app.core.types import ProcessingMode
from app.schemas.protocol import (
    InputCommitEvent,
    ProtocolErrorEvent,
    SessionReadyEvent,
    SessionStartEvent,
    SpeechStartedEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
)
from tests.support.golden import load_golden

GOLDEN = load_golden("protocol.json")
CLIENT_MODELS: dict[str, type[BaseModel]] = {
    "session.start": SessionStartEvent,
    "input.commit": InputCommitEvent,
}
SERVER_MODELS: dict[str, type[BaseModel]] = {
    "session.ready": SessionReadyEvent,
    "speech.started": SpeechStartedEvent,
    "transcript.partial": TranscriptPartialEvent,
    "transcript.final": TranscriptFinalEvent,
    "error": ProtocolErrorEvent,
}
WEBSOCKET_SOURCE = Path(__file__).resolve().parents[2] / "app" / "api" / "websocket.py"


class TestEventInventory:
    def test_the_client_event_set_is_closed(self) -> None:
        assert set(GOLDEN["client_events"]) == set(CLIENT_MODELS)

    def test_the_server_event_set_is_closed(self) -> None:
        assert set(GOLDEN["server_events"]) == set(SERVER_MODELS)

    @pytest.mark.parametrize("event_type", sorted(CLIENT_MODELS))
    def test_client_event_fields_match_the_golden_shape(self, event_type: str) -> None:
        expected: dict[str, Any] = GOLDEN["client_events"][event_type]
        fields = CLIENT_MODELS[event_type].model_fields
        required = sorted(name for name, field in fields.items() if field.is_required())
        optional = sorted(name for name, field in fields.items() if not field.is_required())
        assert required == sorted(expected["required"])
        assert optional == sorted(expected["optional"])

    @pytest.mark.parametrize("event_type", sorted(CLIENT_MODELS))
    def test_client_event_defaults_match_the_golden_shape(self, event_type: str) -> None:
        expected = GOLDEN["client_events"][event_type]["defaults"]
        fields = CLIENT_MODELS[event_type].model_fields
        actual = {
            name: field.get_default() for name, field in fields.items() if not field.is_required()
        }
        assert {name: str(value) for name, value in actual.items()} == {
            name: str(value) for name, value in expected.items()
        }

    @pytest.mark.parametrize("event_type", sorted(SERVER_MODELS))
    def test_server_event_fields_match_the_golden_shape(self, event_type: str) -> None:
        expected = GOLDEN["server_events"][event_type]["fields"]
        assert sorted(SERVER_MODELS[event_type].model_fields) == sorted(expected)

    def test_the_type_discriminator_is_the_declared_event_name(self) -> None:
        for event_type, model in (CLIENT_MODELS | SERVER_MODELS).items():
            assert model.model_fields["type"].get_default() == event_type


class TestAudioGeometry:
    def test_the_frame_contract_matches_the_golden_values(self) -> None:
        audio = GOLDEN["audio"]
        assert audio["sample_rate"] == SAMPLE_RATE
        assert audio["frame_duration_ms"] == FRAME_DURATION_MS
        assert audio["frame_bytes"] == PCM16_FRAME_BYTES
        assert audio["samples_per_frame"] == SAMPLES_PER_FRAME
        assert audio["channels"] == 1
        assert audio["encoding"] == SessionStartEvent.model_fields["format"].get_default()

    def test_the_frame_values_are_internally_consistent(self) -> None:
        audio = GOLDEN["audio"]
        assert audio["samples_per_frame"] * BYTES_PER_SAMPLE == audio["frame_bytes"]
        assert (
            audio["sample_rate"] * audio["frame_duration_ms"] // 1_000
            == (audio["samples_per_frame"])
        )


class TestErrorCodes:
    def test_every_error_code_emitted_by_the_server_is_documented(self) -> None:
        source = WEBSOCKET_SOURCE.read_text(encoding="utf-8")
        emitted = set(re.findall(r'"([A-Z][A-Z_]{3,})"', source))
        assert emitted == set(GOLDEN["error_codes"])

    @pytest.mark.parametrize("code", sorted(GOLDEN["error_codes"]))
    def test_documented_close_codes_are_valid_websocket_codes(self, code: str) -> None:
        entry = GOLDEN["error_codes"][code]
        assert isinstance(entry["retryable"], bool)
        close_code = entry["close_code"]
        assert close_code is None or 1000 <= close_code <= 1015

    def test_retryable_codes_are_the_transient_ones(self) -> None:
        retryable = {code for code, entry in GOLDEN["error_codes"].items() if entry["retryable"]}
        assert retryable == {
            "INFERENCE_ERROR",
            "INTERNAL_ERROR",
            "SERVER_BUSY",
            "SERVICE_UNAVAILABLE",
        }


class TestModeContract:
    def test_the_processing_mode_set_is_closed(self) -> None:
        languages = load_golden("languages.json")
        assert set(languages["final_decoder_by_mode"]) == {mode.value for mode in ProcessingMode}
        assert languages["partial_decoder"] == "ctc"

    def test_the_declared_default_mode_is_the_model_default(self) -> None:
        default = SessionStartEvent.model_fields["mode"].get_default()
        assert str(default) == GOLDEN["client_events"]["session.start"]["defaults"]["mode"]

"""Strict client and server event schemas for OpenAI realtime transcription."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from app.core.types import LanguageCode


class StrictModel(BaseModel):
    """Protocol model which refuses coercion and unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


EventId = Annotated[str, Field(min_length=1, max_length=128)]
LanguageValue = Annotated[LanguageCode, Field(strict=False)]
ModelValue = Annotated[str, Field(min_length=1, max_length=256)]


class PCMFormat(StrictModel):
    type: Literal["audio/pcm"]
    rate: Literal[24_000]


class TranscriptionConfigPatch(StrictModel):
    model: ModelValue | None = None
    languages: Annotated[list[LanguageValue], Field(min_length=1, max_length=1)] | None = None
    language: LanguageValue | None = None

    @model_validator(mode="after")
    def one_language_spelling_at_most(self) -> TranscriptionConfigPatch:
        if {"language", "languages"}.issubset(self.model_fields_set):
            raise ValueError("language and languages cannot be used together")
        return self


class ServerVAD(StrictModel):
    type: Literal["server_vad"]
    threshold: Annotated[float, Field(strict=True, ge=0.01, le=1.0)] = 0.5
    prefix_padding_ms: Annotated[int, Field(strict=True, ge=0, le=5_000)] = 300
    silence_duration_ms: Annotated[int, Field(strict=True, ge=100, le=5_000)] = 500


class AudioInputConfigPatch(StrictModel):
    format: PCMFormat | None = None
    transcription: TranscriptionConfigPatch | None = None
    turn_detection: ServerVAD | None = None

    @model_validator(mode="after")
    def nonnullable_objects_cannot_be_cleared(self) -> AudioInputConfigPatch:
        for name in ("format", "transcription"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self


class AudioConfigPatch(StrictModel):
    input: AudioInputConfigPatch | None = None

    @model_validator(mode="after")
    def input_cannot_be_cleared(self) -> AudioConfigPatch:
        if "input" in self.model_fields_set and self.input is None:
            raise ValueError("input cannot be null")
        return self


class SessionUpdate(StrictModel):
    type: Literal["transcription"] | None = None
    audio: AudioConfigPatch | None = None

    @model_validator(mode="after")
    def session_objects_cannot_be_cleared(self) -> SessionUpdate:
        for name in ("type", "audio"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} cannot be null")
        return self


class SessionUpdateEvent(StrictModel):
    type: Literal["session.update"]
    event_id: EventId | None = None
    session: SessionUpdate


class AudioAppendEvent(StrictModel):
    type: Literal["input_audio_buffer.append"]
    event_id: EventId | None = None
    audio: Annotated[str, Field(min_length=1)]


class AudioCommitEvent(StrictModel):
    type: Literal["input_audio_buffer.commit"]
    event_id: EventId | None = None


class AudioClearEvent(StrictModel):
    type: Literal["input_audio_buffer.clear"]
    event_id: EventId | None = None


ClientEvent = SessionUpdateEvent | AudioAppendEvent | AudioCommitEvent | AudioClearEvent
CLIENT_EVENT_ADAPTER: TypeAdapter[ClientEvent] = TypeAdapter(ClientEvent)


class SessionTranscription(StrictModel):
    model: str
    languages: list[LanguageCode]


class SessionAudioInput(StrictModel):
    format: PCMFormat
    transcription: SessionTranscription
    turn_detection: ServerVAD | None


class SessionAudio(StrictModel):
    input: SessionAudioInput


class TranscriptionSession(StrictModel):
    id: str
    object: Literal["realtime.transcription_session"] = "realtime.transcription_session"
    type: Literal["transcription"] = "transcription"
    expires_at: int
    audio: SessionAudio
    include: list[Literal["item.input_audio_transcription.logprobs"]] = Field(
        default_factory=list, max_length=0
    )


class SessionCreatedEvent(StrictModel):
    type: Literal["session.created"] = "session.created"
    event_id: str
    session: TranscriptionSession


class SessionUpdatedEvent(StrictModel):
    type: Literal["session.updated"] = "session.updated"
    event_id: str
    session: TranscriptionSession


class AudioClearedEvent(StrictModel):
    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"
    event_id: str


class AudioCommittedEvent(StrictModel):
    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    event_id: str
    previous_item_id: str | None
    item_id: str


class SpeechStartedEvent(StrictModel):
    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"
    event_id: str
    audio_start_ms: int
    item_id: str


class SpeechStoppedEvent(StrictModel):
    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
    event_id: str
    audio_end_ms: int
    item_id: str


class ErrorDetail(StrictModel):
    type: Literal["invalid_request_error", "server_error"]
    code: str
    message: str
    param: str | None = None
    event_id: str | None = None


class ErrorEvent(StrictModel):
    type: Literal["error"] = "error"
    event_id: str
    error: ErrorDetail


class TranscriptionDeltaEvent(StrictModel):
    type: Literal["conversation.item.input_audio_transcription.delta"] = (
        "conversation.item.input_audio_transcription.delta"
    )
    event_id: str
    item_id: str
    content_index: Literal[0] = 0
    delta: str


class DurationUsage(StrictModel):
    type: Literal["duration"] = "duration"
    seconds: Annotated[float, Field(ge=0)]


class TranscriptionCompletedEvent(StrictModel):
    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    event_id: str
    item_id: str
    content_index: Literal[0] = 0
    transcript: str
    usage: DurationUsage


class TranscriptionFailedEvent(StrictModel):
    type: Literal["conversation.item.input_audio_transcription.failed"] = (
        "conversation.item.input_audio_transcription.failed"
    )
    event_id: str
    item_id: str
    content_index: Literal[0] = 0
    error: ErrorDetail

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


class PCMFormat(StrictModel):
    type: Literal["audio/pcm"]
    rate: Literal[24_000]


class TranscriptionConfig(StrictModel):
    model: Annotated[str, Field(min_length=1, max_length=256)]
    languages: Annotated[list[LanguageValue], Field(min_length=1, max_length=1)] | None = None
    language: LanguageValue | None = None

    @model_validator(mode="after")
    def exactly_one_language_spelling(self) -> TranscriptionConfig:
        if self.languages is not None and self.language is not None:
            raise ValueError("language and languages cannot be used together")
        if self.languages is None and self.language is None:
            raise ValueError("exactly one language is required")
        return self

    @property
    def selected_language(self) -> LanguageCode:
        if self.languages is not None:
            return self.languages[0]
        assert self.language is not None
        return self.language


class ServerVAD(StrictModel):
    type: Literal["server_vad"]
    threshold: Annotated[float, Field(strict=True, ge=0.01, le=1.0)] = 0.5
    prefix_padding_ms: Annotated[int, Field(strict=True, ge=0, le=5_000)] = 300
    silence_duration_ms: Annotated[int, Field(strict=True, ge=100, le=5_000)] = 500


class AudioInputConfig(StrictModel):
    format: PCMFormat
    transcription: TranscriptionConfig
    turn_detection: ServerVAD | None = None


class AudioConfig(StrictModel):
    input: AudioInputConfig


class SessionUpdate(StrictModel):
    type: Literal["transcription"]
    audio: AudioConfig


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

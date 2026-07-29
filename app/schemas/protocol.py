"""Strict text control events for the realtime protocol.

Audio frames are raw binary PCM and intentionally have no JSON model here.
WebSocket orchestration lives elsewhere.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.types import Decoder, LanguageCode, ProcessingMode


class ProtocolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SessionStartEvent(ProtocolEvent):
    type: Literal["session.start"] = "session.start"
    language: LanguageCode
    format: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    mode: ProcessingMode = ProcessingMode.HYBRID
    vad: bool = True

    @field_validator("sample_rate", "channels", mode="before")
    @classmethod
    def require_json_integers(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("transport dimensions must be JSON integers")
        return value


class InputCommitEvent(ProtocolEvent):
    type: Literal["input.commit"] = "input.commit"


ClientEvent = Annotated[
    SessionStartEvent | InputCommitEvent,
    Field(discriminator="type"),
]


class SessionReadyEvent(ProtocolEvent):
    type: Literal["session.ready"] = "session.ready"
    session_id: str


class SpeechStartedEvent(ProtocolEvent):
    type: Literal["speech.started"] = "speech.started"


class TranscriptPartialEvent(ProtocolEvent):
    type: Literal["transcript.partial"] = "transcript.partial"
    text: str
    revision: int = Field(ge=0)
    is_stable: bool


class TranscriptFinalEvent(ProtocolEvent):
    type: Literal["transcript.final"] = "transcript.final"
    text: str
    language: LanguageCode
    decoder: Decoder
    audio_duration_ms: int = Field(ge=0)
    endpoint_to_final_ms: float = Field(ge=0)


class ProtocolErrorEvent(ProtocolEvent):
    type: Literal["error"] = "error"
    code: str
    message: str
    retryable: bool = False


ServerEvent = Annotated[
    SessionReadyEvent
    | SpeechStartedEvent
    | TranscriptPartialEvent
    | TranscriptFinalEvent
    | ProtocolErrorEvent,
    Field(discriminator="type"),
]

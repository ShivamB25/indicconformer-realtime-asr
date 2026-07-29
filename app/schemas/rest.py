"""Typed REST request options and responses."""

from pydantic import BaseModel, ConfigDict, Field

from app.core.types import Decoder, LanguageCode, ProcessingMode


class TranscriptionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    language: LanguageCode
    mode: ProcessingMode = ProcessingMode.HYBRID


class TranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    language: LanguageCode
    mode: ProcessingMode
    decoder: Decoder
    audio_duration_ms: int = Field(ge=0)
    inference_ms: float = Field(ge=0)
    request_id: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    error: str
    request_id: str | None = None


class LiveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str = "live"


class ReadyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: str
    stage: str
    checks: dict[str, str]
    detail: str | None = None

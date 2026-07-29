"""Typed REST request options and responses."""

from pydantic import BaseModel, ConfigDict, Field

from app.core.types import Decoder, LanguageCode, ProcessingMode


class TranscriptionOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={"examples": [{"language": "hi", "mode": "hybrid"}]},
    )

    language: LanguageCode = Field(description="Supported transcription language code")
    mode: ProcessingMode = Field(
        default=ProcessingMode.HYBRID,
        description="Latency/quality policy that selects the server-side decoder",
    )


class TranscriptionResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [
                {
                    "text": "नमस्ते दुनिया",
                    "language": "hi",
                    "mode": "hybrid",
                    "decoder": "rnnt",
                    "audio_duration_ms": 1250,
                    "inference_ms": 84.2,
                    "request_id": "7139a78b-7ad0-4d6f-bcad-208d5119e00f",
                }
            ]
        },
    )

    text: str = Field(description="Final transcript")
    language: LanguageCode = Field(description="Language used for decoding")
    mode: ProcessingMode = Field(description="Requested processing mode")
    decoder: Decoder = Field(description="Final decoder selected by server policy")
    audio_duration_ms: int = Field(ge=0, description="Accepted audio duration")
    inference_ms: float = Field(ge=0, description="Model inference time")
    request_id: str = Field(description="Request correlation identifier")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [
                {
                    "error": "service is not ready",
                    "request_id": "7139a78b-7ad0-4d6f-bcad-208d5119e00f",
                }
            ]
        },
    )

    error: str = Field(description="Safe client-facing error message")
    request_id: str | None = Field(default=None, description="Request correlation identifier")


class LiveResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, json_schema_extra={"examples": [{"status": "live"}]}
    )

    status: str = Field(default="live", description="Process liveness status")


class ReadyResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [
                {
                    "status": "ready",
                    "stage": "ready",
                    "checks": {"engine": "ready", "scheduler": "ready"},
                    "detail": None,
                }
            ]
        },
    )

    status: str = Field(description="`ready` or `not_ready`")
    stage: str = Field(description="Current lifecycle stage")
    checks: dict[str, str] = Field(description="Per-component readiness state")
    detail: str | None = Field(default=None, description="Safe startup failure detail")

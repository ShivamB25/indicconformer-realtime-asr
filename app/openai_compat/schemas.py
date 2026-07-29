"""Strict response schemas shared by OpenAI-compatible HTTP surfaces."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIErrorDetail(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "message": "Language 'en' is not supported by this model",
                    "type": "invalid_request_error",
                    "param": "language",
                    "code": "unsupported_language",
                }
            ]
        },
    )

    message: str = Field(description="Safe client-facing error message")
    type: str = Field(description="OpenAI-compatible error category")
    param: str | None = Field(default=None, description="Invalid parameter, when applicable")
    code: str | None = Field(default=None, description="Machine-readable error code")


class OpenAIErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: OpenAIErrorDetail = Field(description="OpenAI-compatible error detail")


class TranscriptionJSONResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"text": "नमस्ते दुनिया"}]},
    )

    text: str = Field(description="Final transcript text")


class ModelObject(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "id": "ai4bharat/indic-conformer-600m-multilingual",
                    "object": "model",
                    "created": 0,
                    "owned_by": "ai4bharat",
                }
            ]
        },
    )

    id: str = Field(description="Canonical model identifier")
    object: Literal["model"] = "model"
    created: int = Field(description="Compatibility creation timestamp")
    owned_by: str = Field(description="Model owner")


class ModelList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[ModelObject] = Field(description="Available transcription models")

"""Strict response schemas shared by OpenAI-compatible HTTP surfaces."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class OpenAIErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: OpenAIErrorDetail


class TranscriptionJSONResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class ModelObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[ModelObject]

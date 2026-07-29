"""OpenAI-compatible transcription and model REST endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import partial
from time import perf_counter
from uuid import uuid4

import anyio
from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import FormData, UploadFile

from app.api.auth import require_http_api_key
from app.core.config import Settings
from app.core.types import Decoder, LanguageCode, ProcessingMode
from app.engine.base import TranscriptionRequest
from app.observability.metrics import MetricCode
from app.openai_compat import MODEL_ID, OpenAIError, validate_model
from app.openai_compat.audio import (
    TARGET_SAMPLE_RATE,
    AudioDurationExceeded,
    InvalidAudioError,
    decode_audio_file,
)
from app.openai_compat.constants import MODEL_CREATED, MODEL_OWNER
from app.openai_compat.schemas import ModelList, ModelObject, TranscriptionJSONResponse
from app.transcription import record_success, run_transcription

router = APIRouter(prefix="/v1", tags=["openai"])

_ALLOWED_FIELDS = frozenset(
    {"file", "model", "language", "response_format", "stream", "temperature"}
)
_UNSUPPORTED_FIELDS = frozenset(
    {
        "chunking_strategy",
        "diarization",
        "include",
        "keywords",
        "languages",
        "logprobs",
        "prompt",
        "timestamp_granularities",
        "timestamp_granularities[]",
    }
)


@dataclass(frozen=True, slots=True)
class TranscriptionForm:
    file: UploadFile
    language: LanguageCode
    response_format: str


def _begin_request(request: Request) -> str:
    request_id = str(uuid4())
    request.state.openai_request_id = request_id
    return request_id


def _invalid(message: str, param: str, code: str = "invalid_value") -> OpenAIError:
    return OpenAIError(message, param=param, code=code)


def _single_values(form: FormData) -> dict[str, str | UploadFile]:
    values: dict[str, str | UploadFile] = {}
    for name, value in form.multi_items():
        if name in values:
            raise _invalid(f"The parameter '{name}' may only be provided once", name)
        if name in _UNSUPPORTED_FIELDS:
            raise _invalid(
                f"The parameter '{name}' is not supported by this model",
                name,
                "unsupported_parameter",
            )
        if name not in _ALLOWED_FIELDS:
            raise _invalid(
                f"Unrecognized request argument supplied: {name}",
                name,
                "unknown_parameter",
            )
        values[name] = value
    return values


def _required_text(values: dict[str, str | UploadFile], name: str) -> str:
    value = values.get(name)
    if value is None:
        raise _invalid(f"Missing required parameter: '{name}'", name, "missing_required_parameter")
    if not isinstance(value, str) or not value:
        raise _invalid(f"The parameter '{name}' must be a non-empty string", name)
    return value


def _parse_zero_temperature(value: str | UploadFile | None) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise _invalid("The parameter 'temperature' must be a number", "temperature")
    try:
        temperature = Decimal(value)
    except InvalidOperation as exc:
        raise _invalid("The parameter 'temperature' must be a number", "temperature") from exc
    if not temperature.is_finite() or temperature != 0:
        raise _invalid(
            "Only temperature=0 is supported by this deterministic model",
            "temperature",
            "unsupported_value",
        )


def _parse_transcription_form(form: FormData) -> TranscriptionForm:
    values = _single_values(form)

    upload = values.get("file")
    if upload is None:
        raise _invalid("Missing required parameter: 'file'", "file", "missing_required_parameter")
    if not isinstance(upload, UploadFile):
        raise _invalid("The parameter 'file' must be an uploaded file", "file")

    validate_model(_required_text(values, "model"))
    language_value = _required_text(values, "language")
    try:
        language = LanguageCode(language_value)
    except ValueError as exc:
        raise _invalid(
            f"Language '{language_value}' is not supported by this model",
            "language",
            "unsupported_language",
        ) from exc

    response_format_value = values.get("response_format", "json")
    if not isinstance(response_format_value, str) or response_format_value not in {"json", "text"}:
        raise _invalid(
            "Only response_format values 'json' and 'text' are supported",
            "response_format",
            "unsupported_value",
        )

    stream = values.get("stream")
    if stream is not None:
        if not isinstance(stream, str) or stream.lower() not in {"false", "0"}:
            raise _invalid(
                "Streaming transcription is not supported; stream must be false",
                "stream",
                "unsupported_value",
            )
    _parse_zero_temperature(values.get("temperature"))
    return TranscriptionForm(upload, language, response_format_value)


def _model_object() -> ModelObject:
    return ModelObject(
        id=MODEL_ID,
        created=MODEL_CREATED,
        owned_by=MODEL_OWNER,
    )


@router.get("/models", response_model=ModelList)
async def list_models(request: Request) -> JSONResponse:
    request_id = _begin_request(request)
    require_http_api_key(request)
    payload = ModelList(data=[_model_object()])
    return JSONResponse(
        content=payload.model_dump(mode="json"),
        headers={"x-request-id": request_id},
    )


@router.get("/models/{model:path}", response_model=ModelObject)
async def retrieve_model(request: Request, model: str) -> JSONResponse:
    request_id = _begin_request(request)
    require_http_api_key(request)
    validate_model(model)
    return JSONResponse(
        content=_model_object().model_dump(mode="json"),
        headers={"x-request-id": request_id},
    )


@router.post(
    "/audio/transcriptions",
    response_model=TranscriptionJSONResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid OpenAI request"},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "Upload limit exceeded"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Transcription unavailable"},
    },
)
async def create_transcription(request: Request) -> Response:
    request_id = _begin_request(request)
    require_http_api_key(request)
    settings: Settings = request.app.state.settings
    metrics = request.app.state.metrics

    try:
        form = await request.form(
            max_files=2,
            max_fields=32,
            max_part_size=min(settings.max_upload_bytes + 1, 1024 * 1024),
        )
    except OpenAIError:
        raise
    except Exception as exc:
        raise _invalid(
            "The multipart request could not be parsed", "file", "invalid_multipart"
        ) from exc
    parsed = _parse_transcription_form(form)

    if not request.app.state.readiness.snapshot().ready:
        raise OpenAIError(
            "The transcription service is not ready",
            status_code=503,
            error_type="server_error",
            code="service_unavailable",
        )

    try:
        if parsed.file.size is not None and parsed.file.size > settings.max_upload_bytes:
            raise OpenAIError(
                "The audio upload exceeds the configured size limit",
                status_code=413,
                param="file",
                code="upload_too_large",
            )
        payload = await parsed.file.read(settings.max_upload_bytes + 1)
    finally:
        await parsed.file.close()
    if len(payload) > settings.max_upload_bytes:
        raise OpenAIError(
            "The audio upload exceeds the configured size limit",
            status_code=413,
            param="file",
            code="upload_too_large",
        )

    try:
        waveform = await anyio.to_thread.run_sync(
            partial(decode_audio_file, payload, max_audio_seconds=settings.max_audio_seconds)
        )
    except AudioDurationExceeded as exc:
        raise OpenAIError(
            str(exc),
            status_code=413,
            param="file",
            code="audio_too_long",
        ) from exc
    except InvalidAudioError as exc:
        raise OpenAIError(str(exc), param="file", code="invalid_audio") from exc

    engine_request = TranscriptionRequest(
        audio=waveform,
        sample_rate=TARGET_SAMPLE_RATE,
        language=parsed.language.value,
        decoder=Decoder.RNNT.value,
    )
    started = perf_counter()
    try:
        with anyio.fail_after(settings.request_timeout_seconds):
            result = await run_transcription(request, request_id, engine_request)
    except TimeoutError as exc:
        metrics.record_error(MetricCode.TIMEOUT)
        raise OpenAIError(
            "Transcription timed out",
            status_code=503,
            error_type="server_error",
            code="timeout",
        ) from exc
    except RuntimeError as exc:
        metrics.record_error(MetricCode.INFERENCE_ERROR)
        raise OpenAIError(
            "Transcription is unavailable",
            status_code=503,
            error_type="server_error",
            code="service_unavailable",
        ) from exc

    elapsed_seconds = perf_counter() - started
    record_success(
        metrics,
        request_id,
        parsed.language,
        ProcessingMode.ACCURACY,
        result,
        elapsed_seconds,
    )
    headers = {"x-request-id": request_id}
    if parsed.response_format == "text":
        return PlainTextResponse(result.text, headers=headers)
    response = TranscriptionJSONResponse(text=result.text)
    return JSONResponse(content=response.model_dump(mode="json"), headers=headers)

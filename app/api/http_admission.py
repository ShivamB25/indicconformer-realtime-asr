"""Pre-parser admission controls for inference HTTP requests."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.auth import require_http_api_key
from app.observability.metrics import MetricCode
from app.openai_compat.constants import is_openai_route
from app.openai_compat.errors import OpenAIError, openai_error_response
from app.schemas.rest import ErrorResponse

# Multipart boundaries and the small text fields are transport overhead rather than
# uploaded audio. Keeping this allowance fixed lets an audio file exactly at the
# configured limit through while placing a hard bound on all bytes parsed/spooled.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
_BODY_LIMIT_PATHS = frozenset({"/v1/audio/transcriptions", "/v1/transcribe"})
_AUTHENTICATED_HTTP_PATHS = frozenset({"/v1/audio/transcriptions", "/v1/models", "/v1/transcribe"})


class _BodyLimitExceeded(BaseException):
    """Private control-flow signal that multipart parsers cannot translate to 400."""


class HTTPAdmissionMiddleware:
    """Authenticate and bound inference HTTP bodies before request parsing.

    This is pure ASGI middleware rather than ``BaseHTTPMiddleware`` so request body
    backpressure remains intact. The receive wrapper stops requesting chunks as
    soon as the total body crosses the configured bound, including requests that
    use chunked transfer encoding.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not _is_authenticated_path(path):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        try:
            require_http_api_key(request)
        except HTTPException as exc:
            await _authentication_response(path, exc)(scope, receive, send)
            return

        if path not in _BODY_LIMIT_PATHS:
            await self.app(scope, receive, send)
            return

        settings = request.app.state.settings
        limit = settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        declared_length = _content_length(scope)
        if declared_length is not None and declared_length > limit:
            _record_upload_rejection(request)
            await _body_too_large_response(path)(scope, receive, send)
            return

        consumed = 0

        async def limited_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise _BodyLimitExceeded
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyLimitExceeded:
            _record_upload_rejection(request)
            await _body_too_large_response(path)(scope, receive, send)


def _is_authenticated_path(path: str) -> bool:
    return path in _AUTHENTICATED_HTTP_PATHS or path.startswith("/v1/models/")


def _content_length(scope: Scope) -> int | None:
    values = [
        value for name, value in scope.get("headers", ()) if name.lower() == b"content-length"
    ]
    if not values:
        return 0
    if len(values) != 1:
        return None
    try:
        value = values[0].decode("ascii")
        if not value or not value.isdecimal():
            return None
        return int(value)
    except (UnicodeDecodeError, ValueError):
        return None


def _record_upload_rejection(request: Request) -> None:
    metrics: Any = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record_rejection(MetricCode.UPLOAD_TOO_LARGE)


def _authentication_response(path: str, exc: HTTPException) -> JSONResponse:
    headers = dict(exc.headers or {})
    if is_openai_route(path):
        return openai_error_response(
            OpenAIError(
                str(exc.detail),
                status_code=status.HTTP_401_UNAUTHORIZED,
                error_type="authentication_error",
                code="invalid_api_key",
            ),
            request_id=str(uuid4()),
            headers=headers,
        )
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=ErrorResponse(error=str(exc.detail)).model_dump(mode="json"),
        headers=headers,
    )


def _body_too_large_response(path: str) -> JSONResponse:
    if is_openai_route(path):
        return openai_error_response(
            OpenAIError(
                "The request body exceeds the configured size limit",
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                param="file",
                code="upload_too_large",
            ),
            request_id=str(uuid4()),
        )
    return JSONResponse(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        content=ErrorResponse(error="request body exceeds configured limit").model_dump(
            mode="json"
        ),
    )

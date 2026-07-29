"""FastAPI application factory."""

from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.health import router as health_router
from app.api.openai import router as openai_router
from app.api.openai_realtime import router as openai_realtime_router
from app.api.rest import router as rest_router
from app.api.websocket import router as websocket_router
from app.core.config import Settings, get_settings
from app.core.lifespan import Scheduler, build_lifespan
from app.core.logging import configure_logging
from app.core.readiness import ReadinessTracker
from app.engine.base import Engine
from app.observability.metrics import MetricCode, install_metrics
from app.openai_compat.constants import is_openai_route
from app.openai_compat.errors import OpenAIError, openai_error_response
from app.schemas.rest import ErrorResponse


@lru_cache(maxsize=1)
def _configure_logging_once() -> None:
    configure_logging()


def _describe_validation_failure(exc: RequestValidationError) -> str:
    """Summarize field errors without echoing the rejected input back."""

    parts = [
        f"{'.'.join(str(item) for item in error['loc']) or 'request'}: {error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(parts) if parts else "request validation failed"


def _openai_request_id(request: Request) -> str:
    request_id = getattr(request.state, "openai_request_id", None)
    if isinstance(request_id, str):
        return request_id
    request_id = str(uuid4())
    request.state.openai_request_id = request_id
    return request_id


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    scheduler: Scheduler | None = None,
) -> FastAPI:
    """Construct the service without loading model runtimes or assets."""

    _configure_logging_once()
    runtime_settings = settings if settings is not None else get_settings()
    application = FastAPI(
        title="IndicConformer Realtime ASR",
        version="1.0.0",
        lifespan=build_lifespan(
            runtime_settings,
            engine=engine,
            scheduler=scheduler,
        ),
    )

    @application.exception_handler(OpenAIError)
    async def openai_error(request: Request, exc: OpenAIError) -> JSONResponse:
        if exc.status_code < 500:
            request.app.state.metrics.record_rejection(MetricCode.BAD_REQUEST)
        return openai_error_response(exc, request_id=_openai_request_id(request))

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if is_openai_route(request.url.path):
            error_type = (
                "authentication_error"
                if exc.status_code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}
                else "invalid_request_error"
            )
            code = {
                status.HTTP_401_UNAUTHORIZED: "invalid_api_key",
                status.HTTP_403_FORBIDDEN: "permission_denied",
                status.HTTP_404_NOT_FOUND: "not_found",
            }.get(exc.status_code)
            return openai_error_response(
                OpenAIError(
                    str(exc.detail),
                    status_code=exc.status_code,
                    error_type=error_type,
                    code=code,
                ),
                request_id=_openai_request_id(request),
                headers=exc.headers,
            )
        payload = ErrorResponse(error=str(exc.detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request.app.state.metrics.record_rejection(MetricCode.VALIDATION_ERROR)
        message = _describe_validation_failure(exc)
        if is_openai_route(request.url.path):
            return openai_error_response(
                OpenAIError(message, param="request", code="validation_error"),
                request_id=_openai_request_id(request),
            )
        payload = ErrorResponse(error=message)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=payload.model_dump(mode="json"),
        )

    application.state.settings = runtime_settings
    application.state.readiness = ReadinessTracker()
    application.state.engine = None
    application.state.scheduler = None
    install_metrics(application)
    application.include_router(health_router)
    application.include_router(rest_router)
    application.include_router(openai_router)
    application.include_router(openai_realtime_router)
    application.include_router(websocket_router)
    return application


app = create_app()

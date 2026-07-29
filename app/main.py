"""FastAPI application factory."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.rest import router as rest_router
from app.api.websocket import router as websocket_router
from app.core.config import Settings, get_settings
from app.core.lifespan import Scheduler, build_lifespan
from app.core.logging import configure_logging
from app.core.readiness import ReadinessTracker
from app.engine.base import Engine
from app.observability.metrics import MetricCode, install_metrics
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

    @application.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        payload = ErrorResponse(error=str(exc.detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request.app.state.metrics.record_rejection(MetricCode.VALIDATION_ERROR)
        payload = ErrorResponse(error=_describe_validation_failure(exc))
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
    application.include_router(websocket_router)
    return application


app = create_app()

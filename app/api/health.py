"""Inference-free Kubernetes health endpoints."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.core.readiness import CheckStatus, ReadinessTracker
from app.schemas.rest import LiveResponse, ReadyResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=LiveResponse,
    summary="Check process liveness",
    description=(
        "Returns `200` when the HTTP process and event loop are alive. "
        "It never loads a model or invokes inference."
    ),
)
async def live() -> LiveResponse:
    """Process liveness only; this endpoint never invokes inference."""

    return LiveResponse()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Check model and scheduler readiness",
    description=(
        "Returns `200` only when startup checks are ready or disabled; otherwise "
        "returns `503` with the current stage and component checks."
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadyResponse,
            "description": "One or more startup components are not ready",
        }
    },
)
async def ready(request: Request) -> ReadyResponse | JSONResponse:
    """Report staged engine and scheduler readiness without doing work."""

    tracker: ReadinessTracker = request.app.state.readiness
    snapshot = tracker.snapshot()
    checks = dict(snapshot.checks)

    engine = getattr(request.app.state, "engine", None)
    if engine is not None and not engine.readiness.ready:
        checks["engine"] = engine.readiness.state.value

    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None and not scheduler.running:
        checks["scheduler"] = CheckStatus.STOPPED

    is_ready = all(value in {CheckStatus.READY, CheckStatus.DISABLED} for value in checks.values())
    payload = ReadyResponse(
        status="ready" if is_ready else "not_ready",
        stage=snapshot.stage,
        checks=checks,
        detail=snapshot.detail,
    )
    if is_ready:
        return payload
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(mode="json"),
    )

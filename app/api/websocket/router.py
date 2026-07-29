"""FastAPI router assembly for the native realtime protocol."""

from fastapi import APIRouter, WebSocket

from app.api.websocket.connection import _serve_websocket
from app.api.websocket.state import WebSocketConfig, _SessionRegistry
from app.engine.scheduler import InferenceScheduler


def create_websocket_router(
    scheduler: InferenceScheduler | None = None, config: WebSocketConfig | None = None
) -> APIRouter:
    """Create a router, optionally binding a scheduler for isolated tests."""

    effective_config = config or WebSocketConfig()
    registry = _SessionRegistry(effective_config.max_sessions)
    result = APIRouter(tags=["realtime"])

    @result.websocket("/v1/realtime")
    async def realtime_endpoint(websocket: WebSocket) -> None:
        resolved = scheduler or getattr(websocket.app.state, "scheduler", None)
        await _serve_websocket(websocket, resolved, effective_config, registry)

    return result


router = create_websocket_router()

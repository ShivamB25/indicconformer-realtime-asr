"""FastAPI router assembly for the native realtime protocol."""

from fastapi import APIRouter, WebSocket

from app.api.realtime import ConnectionRegistry
from app.api.websocket.connection import _serve_websocket
from app.api.websocket.state import WebSocketConfig
from app.engine.scheduler import InferenceScheduler


def create_websocket_router(
    scheduler: InferenceScheduler | None = None,
    config: WebSocketConfig | None = None,
    registry: ConnectionRegistry | None = None,
) -> APIRouter:
    """Create a native router, optionally binding dependencies for isolated tests."""

    effective_config = config or WebSocketConfig()
    result = APIRouter(tags=["realtime"])

    @result.websocket("/v1/realtime/native")
    async def realtime_endpoint(websocket: WebSocket) -> None:
        resolved = scheduler or getattr(websocket.app.state, "scheduler", None)
        shared_registry = registry or getattr(websocket.app.state, "realtime_connections", None)
        if shared_registry is None:
            raise RuntimeError("application realtime connection registry is not configured")
        await _serve_websocket(websocket, resolved, effective_config, shared_registry)

    return result


router = create_websocket_router()

"""FastAPI router assembly for OpenAI realtime transcription."""

from typing import Any

from fastapi import APIRouter, WebSocket

from app.api.openai_realtime.connection import OpenAIRealtimeConfig, _serve
from app.api.realtime import ConnectionRegistry


def create_openai_realtime_router(
    scheduler: Any = None,
    config: OpenAIRealtimeConfig | None = None,
    registry: ConnectionRegistry | None = None,
) -> APIRouter:
    effective = config or OpenAIRealtimeConfig()
    result = APIRouter(tags=["openai-realtime-transcription"])

    @result.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        shared_registry = registry or getattr(websocket.app.state, "realtime_connections", None)
        if shared_registry is None:
            raise RuntimeError("application realtime connection registry is not configured")
        await _serve(
            websocket,
            scheduler or getattr(websocket.app.state, "scheduler", None),
            effective,
            shared_registry,
        )

    return result


router = create_openai_realtime_router()

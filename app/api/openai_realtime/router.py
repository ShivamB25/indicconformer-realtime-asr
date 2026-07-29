"""FastAPI router assembly for OpenAI realtime transcription."""

from typing import Any

from fastapi import APIRouter, WebSocket

from app.api.openai_realtime.connection import OpenAIRealtimeConfig, _serve


def create_openai_realtime_router(
    scheduler: Any = None, config: OpenAIRealtimeConfig | None = None
) -> APIRouter:
    result = APIRouter(tags=["openai-realtime-transcription"])
    effective = config or OpenAIRealtimeConfig()

    @result.websocket("/v1/realtime/transcription_sessions")
    async def transcription_sessions(websocket: WebSocket) -> None:
        await _serve(
            websocket, scheduler or getattr(websocket.app.state, "scheduler", None), effective
        )

    return result


router = create_openai_realtime_router()

"""Stable router exports for OpenAI realtime transcription."""

from app.api.openai_realtime.connection import OpenAIRealtimeConfig
from app.api.openai_realtime.router import create_openai_realtime_router, router

__all__ = ["OpenAIRealtimeConfig", "create_openai_realtime_router", "router"]

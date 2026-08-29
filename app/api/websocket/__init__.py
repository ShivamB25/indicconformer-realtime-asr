"""Stable router exports for the native realtime protocol."""

from app.api.websocket.router import create_websocket_router, router
from app.api.websocket.state import WebSocketConfig

__all__ = ["WebSocketConfig", "create_websocket_router", "router"]

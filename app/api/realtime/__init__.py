"""Responsibility boundaries shared by realtime protocols."""

from app.api.realtime.registry import DEFAULT_CONNECTION_LIMIT, ConnectionRegistry

__all__ = ["ConnectionRegistry", "DEFAULT_CONNECTION_LIMIT"]

"""Application-owned concurrency guard shared by realtime protocols."""

from __future__ import annotations

import asyncio

DEFAULT_CONNECTION_LIMIT = 128


class ConnectionRegistry:
    """Bound live WebSocket connections across every realtime protocol."""

    __slots__ = ("_active", "_limit", "_lock")

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("connection limit must be a positive integer")
        self._active = 0
        self._limit = limit
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def limit(self) -> int:
        return self._limit

    async def acquire(self) -> bool:
        async with self._lock:
            if self._active >= self._limit:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active == 0:
                raise RuntimeError("connection registry release without matching acquire")
            self._active -= 1


__all__ = ["ConnectionRegistry", "DEFAULT_CONNECTION_LIMIT"]

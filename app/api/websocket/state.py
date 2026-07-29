"""Configuration and connection-owned state for the native realtime protocol."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.audio.endpoint import AdaptivePartialCadence, EndpointDetector
from app.audio.pcm import PCM16_FRAME_BYTES, PCM16Buffer
from app.audio.stable_prefix import RollingStablePrefix
from app.audio.vad import EnergyVAD
from app.core.types import Decoder
from app.schemas.protocol import SessionStartEvent


@dataclass(frozen=True, slots=True)
class WebSocketConfig:
    max_sessions: int = 128
    max_session_seconds: float = 3_600.0
    max_frame_bytes: int = PCM16_FRAME_BYTES
    max_utterance_ms: int = 30_000
    idle_timeout_seconds: float = 30.0
    vad_threshold: float = 0.015
    speech_start_ms: int = 60
    speech_end_ms: int = 600
    partial_history: int = 3
    partial_latency_ms: int = 240
    partial_hybrid_ms: int = 400
    partial_accuracy_ms: int = 800
    partial_minimum_ms: int = 200
    partial_maximum_ms: int = 1_200

    def __post_init__(self) -> None:
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if self.max_session_seconds <= 0 or self.idle_timeout_seconds <= 0:
            raise ValueError("session and idle timeouts must be positive")
        if self.max_frame_bytes < PCM16_FRAME_BYTES:
            raise ValueError("max_frame_bytes cannot be smaller than one PCM frame")
        if self.max_utterance_ms <= 0 or self.max_utterance_ms % 20:
            raise ValueError("max_utterance_ms must be a positive multiple of 20 ms")
        if self.partial_history < 2:
            raise ValueError("partial_history must be at least two")
        cadences = (self.partial_latency_ms, self.partial_hybrid_ms, self.partial_accuracy_ms)
        if (
            not 0 < self.partial_minimum_ms <= min(cadences)
            or max(cadences) > self.partial_maximum_ms
        ):
            raise ValueError("partial cadence bounds are invalid")


class _SessionRegistry:
    """Concurrency guard shared by all connections on one router."""

    def __init__(self, limit: int) -> None:
        self._active = 0
        self._limit = limit
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            if self._active >= self._limit:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)


@dataclass(slots=True)
class _LiveSession:
    session_id: str
    start: SessionStartEvent
    partial_decoder: Decoder
    final_decoder: Decoder
    buffer: PCM16Buffer
    endpoint: EndpointDetector
    vad: EnergyVAD
    cadence: AdaptivePartialCadence
    stable_prefix: RollingStablePrefix
    revision: int = 0
    epoch: int = 0
    last_partial: str = ""
    partial_task: asyncio.Task[None] | None = None

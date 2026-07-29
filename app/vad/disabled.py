"""Explicit VAD bypass for manual-commit streaming sessions."""

from __future__ import annotations

from typing import Protocol

from app.vad.base import StreamCapacity, VADClosedError, VADSampleRate, expected_frame_bytes
from app.vad.runtime import VADRuntimeMetrics


class _ProviderMetrics(VADRuntimeMetrics, Protocol):
    pass


class DisabledVADProvider:
    """Treat every valid frame as speech without loading a classifier."""

    __slots__ = ("_capacity", "_closed", "_metrics", "_started")

    def __init__(
        self,
        *,
        max_streams: int,
        metrics: _ProviderMetrics | None = None,
    ) -> None:
        self._capacity = StreamCapacity(max_streams)
        self._metrics = metrics
        self._started = False
        self._closed = False

    @property
    def name(self) -> str:
        return "disabled"

    @property
    def default_threshold(self) -> float:
        return 0.5

    @property
    def active_streams(self) -> int:
        return self._capacity.active

    async def startup(self) -> None:
        if self._closed:
            raise VADClosedError("disabled VAD provider is closed")
        self._started = True

    def new_stream(self, input_sample_rate: VADSampleRate) -> _DisabledVADStream:
        if self._closed or not self._started:
            raise VADClosedError("disabled VAD provider is not running")
        if isinstance(input_sample_rate, bool) or input_sample_rate not in (16_000, 24_000):
            raise ValueError("VAD input sample rate must be 16000 or 24000 Hz")
        self._capacity.acquire()
        return _DisabledVADStream(
            capacity=self._capacity,
            input_sample_rate=input_sample_rate,
            metrics=self._metrics,
        )

    async def close(self) -> None:
        self._closed = True
        self._started = False


class _DisabledVADStream:
    __slots__ = ("_capacity", "_closed", "_expected_bytes", "_metrics")

    def __init__(
        self,
        *,
        capacity: StreamCapacity,
        input_sample_rate: VADSampleRate,
        metrics: _ProviderMetrics | None,
    ) -> None:
        self._capacity = capacity
        self._expected_bytes = expected_frame_bytes(input_sample_rate)
        self._metrics = metrics
        self._closed = False
        if metrics is not None:
            metrics.vad_stream_started("disabled")

    async def score(self, pcm16_20ms: bytes) -> float:
        if not isinstance(pcm16_20ms, bytes):
            raise ValueError("VAD frame must be bytes containing PCM16LE mono audio")
        if len(pcm16_20ms) != self._expected_bytes:
            raise ValueError(f"VAD frame must contain exactly {self._expected_bytes} bytes")
        if self._closed:
            raise VADClosedError("disabled VAD stream is closed")
        return 1.0

    def reset(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._capacity.release()
        if self._metrics is not None:
            self._metrics.vad_stream_ended("disabled")


__all__ = ["DisabledVADProvider"]

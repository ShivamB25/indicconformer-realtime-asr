"""Normalized RMS energy provider for streaming voice activity detection."""

from __future__ import annotations

import asyncio
import math
from typing import Protocol

import numpy as np

from app.vad.base import (
    StreamCapacity,
    VADCapacityError,
    VADClosedError,
    VADInferenceError,
    VADSampleRate,
    expected_frame_bytes,
)
from app.vad.runtime import BoundedVADRuntime, VADRuntimeMetrics

_DEFAULT_THRESHOLD = 0.015


class _ProviderMetrics(VADRuntimeMetrics, Protocol):
    pass


class EnergyVADProvider:
    """Process-owned bounded runtime for deterministic RMS scoring."""

    __slots__ = ("_capacity", "_closed", "_metrics", "_runtime", "_started", "_threshold")

    def __init__(
        self,
        *,
        max_streams: int,
        workers: int,
        pending_capacity: int,
        deadline_seconds: float,
        metrics: _ProviderMetrics | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        if isinstance(threshold, bool) or not math.isfinite(threshold):
            raise ValueError("energy VAD threshold must be finite")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("energy VAD threshold must be in (0, 1]")
        self._threshold = threshold
        self._capacity = StreamCapacity(max_streams)
        self._metrics = metrics
        self._runtime = BoundedVADRuntime(
            provider=self.name,
            workers=workers,
            pending_capacity=pending_capacity,
            deadline_seconds=deadline_seconds,
            metrics=metrics,
        )
        self._started = False
        self._closed = False

    @property
    def name(self) -> str:
        return "energy"

    @property
    def default_threshold(self) -> float:
        return self._threshold

    @property
    def active_streams(self) -> int:
        return self._capacity.active

    async def startup(self) -> None:
        if self._closed:
            raise VADClosedError("energy VAD provider is closed")
        if self._started:
            return
        await self._runtime.start()
        self._started = True

    def new_stream(self, input_sample_rate: VADSampleRate) -> _EnergyVADStream:
        if self._closed or not self._started:
            raise VADClosedError("energy VAD provider is not running")
        if isinstance(input_sample_rate, bool) or input_sample_rate not in (16_000, 24_000):
            raise ValueError("VAD input sample rate must be 16000 or 24000 Hz")
        self._capacity.acquire()
        try:
            return _EnergyVADStream(
                runtime=self._runtime,
                capacity=self._capacity,
                input_sample_rate=input_sample_rate,
                metrics=self._metrics,
            )
        except Exception:
            self._capacity.release()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._started = False
        await self._runtime.close()


class _EnergyVADStream:
    """Connection-owned RMS scorer with one provider stream lease."""

    __slots__ = (
        "_capacity",
        "_closed",
        "_expected_bytes",
        "_failed",
        "_in_flight",
        "_metrics",
        "_runtime",
    )

    def __init__(
        self,
        *,
        runtime: BoundedVADRuntime,
        capacity: StreamCapacity,
        input_sample_rate: VADSampleRate,
        metrics: _ProviderMetrics | None,
    ) -> None:
        self._runtime = runtime
        self._capacity = capacity
        self._expected_bytes = expected_frame_bytes(input_sample_rate)
        self._metrics = metrics
        self._in_flight = False
        self._failed = False
        self._closed = False
        if metrics is not None:
            metrics.vad_stream_started("energy")

    async def score(self, pcm16_20ms: bytes) -> float:
        if not isinstance(pcm16_20ms, bytes):
            raise ValueError("VAD frame must be bytes containing PCM16LE mono audio")
        if len(pcm16_20ms) != self._expected_bytes:
            raise ValueError(f"VAD frame must contain exactly {self._expected_bytes} bytes")
        if self._closed:
            raise VADClosedError("energy VAD stream is closed")
        if self._failed:
            raise VADInferenceError("energy VAD stream must be reset after a failed score")
        if self._in_flight:
            raise VADInferenceError("VAD stream score calls must be sequential")

        self._in_flight = True
        try:
            return await self._runtime.submit(lambda: _normalized_rms(pcm16_20ms))
        except (VADCapacityError, VADInferenceError, asyncio.CancelledError):
            self._failed = True
            raise
        finally:
            self._in_flight = False

    def reset(self) -> None:
        if self._closed:
            return
        if self._in_flight:
            raise VADInferenceError("cannot reset a VAD stream during a score call")
        self._failed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._capacity.release()
        if self._metrics is not None:
            self._metrics.vad_stream_ended("energy")


def _normalized_rms(pcm16le: bytes) -> float:
    samples = np.frombuffer(pcm16le, dtype="<i2")
    normalized = samples.astype(np.float64) / 32_768.0
    return float(np.sqrt(np.dot(normalized, normalized) / normalized.size))


__all__ = ["EnergyVADProvider"]

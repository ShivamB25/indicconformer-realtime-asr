"""WebRTC VAD provider with connection-owned classifier and resampler state."""

from __future__ import annotations

import asyncio
from typing import Protocol

import numpy as np
import soxr  # type: ignore[import-untyped]
import webrtcvad  # type: ignore[import-untyped]

from app.vad.base import (
    StreamCapacity,
    VADCapacityError,
    VADClosedError,
    VADConfigurationError,
    VADInferenceError,
    VADSampleRate,
    expected_frame_bytes,
)
from app.vad.runtime import BoundedVADRuntime, VADRuntimeMetrics

_WEBRTC_SAMPLE_RATE = 16_000
_WEBRTC_FRAME_SAMPLES = 320
_WEBRTC_FRAME_BYTES = _WEBRTC_FRAME_SAMPLES * 2


class _ProviderMetrics(VADRuntimeMetrics, Protocol):
    pass


class WebRTCVADProvider:
    """Process-owned bounded runtime for the WebRTC binary speech classifier."""

    __slots__ = ("_capacity", "_closed", "_metrics", "_mode", "_runtime", "_started")

    def __init__(
        self,
        *,
        max_streams: int,
        workers: int,
        pending_capacity: int,
        deadline_seconds: float,
        metrics: _ProviderMetrics | None = None,
        mode: int = 1,
    ) -> None:
        if isinstance(mode, bool) or not isinstance(mode, int) or mode not in range(4):
            raise ValueError("WebRTC VAD mode must be an integer from 0 through 3")
        self._mode = mode
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
        return "webrtc"

    @property
    def default_threshold(self) -> float:
        return 0.5

    @property
    def active_streams(self) -> int:
        return self._capacity.active

    async def startup(self) -> None:
        if self._closed:
            raise VADClosedError("WebRTC VAD provider is closed")
        if self._started:
            return
        try:
            webrtcvad.Vad(self._mode)
        except Exception as exc:
            raise VADConfigurationError("WebRTC VAD classifier could not be initialized") from exc
        await self._runtime.start()
        self._started = True

    def new_stream(self, input_sample_rate: VADSampleRate) -> _WebRTCVADStream:
        if self._closed or not self._started:
            raise VADClosedError("WebRTC VAD provider is not running")
        if isinstance(input_sample_rate, bool) or input_sample_rate not in (16_000, 24_000):
            raise ValueError("VAD input sample rate must be 16000 or 24000 Hz")
        self._capacity.acquire()
        try:
            return _WebRTCVADStream(
                runtime=self._runtime,
                capacity=self._capacity,
                input_sample_rate=input_sample_rate,
                mode=self._mode,
                metrics=self._metrics,
            )
        except Exception as exc:
            self._capacity.release()
            if isinstance(exc, VADConfigurationError):
                raise
            raise VADConfigurationError("WebRTC VAD stream could not be initialized") from exc

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._started = False
        await self._runtime.close()


class _WebRTCVADStream:
    """Sequential WebRTC classifier state held by exactly one connection."""

    __slots__ = (
        "_capacity",
        "_closed",
        "_expected_bytes",
        "_failed",
        "_handle",
        "_in_flight",
        "_input_sample_rate",
        "_metrics",
        "_mode",
        "_resampler",
        "_resampler_primed",
        "_runtime",
    )

    def __init__(
        self,
        *,
        runtime: BoundedVADRuntime,
        capacity: StreamCapacity,
        input_sample_rate: VADSampleRate,
        mode: int,
        metrics: _ProviderMetrics | None,
    ) -> None:
        self._runtime = runtime
        self._capacity = capacity
        self._input_sample_rate = input_sample_rate
        self._expected_bytes = expected_frame_bytes(input_sample_rate)
        self._mode = mode
        self._handle = webrtcvad.Vad(mode)
        self._resampler = self._new_resampler()
        self._resampler_primed = False
        self._metrics = metrics
        self._in_flight = False
        self._failed = False
        self._closed = False
        if metrics is not None:
            metrics.vad_stream_started("webrtc")

    async def score(self, pcm16_20ms: bytes) -> float:
        if not isinstance(pcm16_20ms, bytes):
            raise ValueError("VAD frame must be bytes containing PCM16LE mono audio")
        if len(pcm16_20ms) != self._expected_bytes:
            raise ValueError(f"VAD frame must contain exactly {self._expected_bytes} bytes")
        if self._closed:
            raise VADClosedError("WebRTC VAD stream is closed")
        if self._failed:
            raise VADInferenceError("WebRTC VAD stream must be reset after a failed score")
        if self._in_flight:
            raise VADInferenceError("VAD stream score calls must be sequential")

        self._in_flight = True
        try:
            return await self._runtime.submit(lambda: self._classify(pcm16_20ms))
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
        try:
            handle = webrtcvad.Vad(self._mode)
            resampler = self._new_resampler()
        except Exception as exc:
            self._failed = True
            raise VADInferenceError("WebRTC VAD stream could not be reset") from exc
        self._handle = handle
        self._resampler = resampler
        self._resampler_primed = False
        self._failed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._capacity.release()
        if self._metrics is not None:
            self._metrics.vad_stream_ended("webrtc")

    def _new_resampler(self) -> soxr.ResampleStream | None:
        if self._input_sample_rate == _WEBRTC_SAMPLE_RATE:
            return None
        return soxr.ResampleStream(
            self._input_sample_rate,
            _WEBRTC_SAMPLE_RATE,
            1,
            dtype="float32",
            quality="QQ",
        )

    def _classify(self, pcm16_20ms: bytes) -> float:
        frame, handle, resampler = pcm16_20ms, self._handle, self._resampler
        if resampler is not None:
            source = np.frombuffer(pcm16_20ms, dtype="<i2").astype(np.float32)
            if not self._resampler_primed:
                primed = np.empty(source.size + 1, dtype=np.float32)
                primed[0] = source[0]
                primed[1:] = source
                source = primed
            converted = resampler.resample_chunk(source, last=False)
            if self._resampler is resampler:
                self._resampler_primed = True
            if converted.ndim != 1 or converted.size != _WEBRTC_FRAME_SAMPLES:
                raise RuntimeError("WebRTC resampler did not produce one exact 20 ms frame")
            frame = np.clip(np.rint(converted), -32_768, 32_767).astype("<i2").tobytes()

        if len(frame) != _WEBRTC_FRAME_BYTES:
            raise RuntimeError("WebRTC VAD requires one exact 20 ms 16 kHz frame")
        return 1.0 if handle.is_speech(frame, _WEBRTC_SAMPLE_RATE) else 0.0


__all__ = ["WebRTCVADProvider"]

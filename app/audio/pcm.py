"""PCM16 framing and bounded audio buffers for streaming ASR."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ReadableBuffer: TypeAlias = bytes | bytearray | memoryview
SAMPLE_RATE = 16_000
FRAME_DURATION_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_DURATION_MS // 1_000
BYTES_PER_SAMPLE = 2
PCM16_FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE
_PCM16_LE = np.dtype("<i2")


class PCMError(ValueError):
    """Base class for invalid PCM input."""


class PCMFrameSizeError(PCMError):
    """Raised when a binary message is not one complete 20 ms frame."""


class PCMBufferOverflow(PCMError):
    """Raised before a bounded buffer would exceed its configured limit."""


def decode_pcm16_frame(data: ReadableBuffer) -> NDArray[np.int16]:
    """Decode exactly one 20 ms mono little-endian PCM16 frame."""
    view = memoryview(data)
    if view.nbytes != PCM16_FRAME_BYTES:
        raise PCMFrameSizeError(
            f"expected {PCM16_FRAME_BYTES} bytes of PCM16, received {view.nbytes}"
        )
    if not view.contiguous:
        raise PCMError("PCM frame must be contiguous")
    return np.frombuffer(view, dtype=_PCM16_LE, count=SAMPLES_PER_FRAME).copy()


def pcm16_to_float32(samples: NDArray[np.int16]) -> NDArray[np.float32]:
    """Normalize signed PCM16 samples to the engine float32 range."""
    if samples.ndim != 1:
        raise PCMError("PCM audio must be mono")
    return samples.astype(np.float32) / np.float32(32_768.0)


class PCM16Buffer:
    """Bounded append-only storage of complete 20 ms PCM16 frames."""

    __slots__ = ("_data", "_max_bytes")

    def __init__(self, max_duration_ms: int) -> None:
        if max_duration_ms < FRAME_DURATION_MS:
            raise ValueError("max_duration_ms must hold at least one frame")
        if max_duration_ms % FRAME_DURATION_MS:
            raise ValueError("max_duration_ms must be a multiple of 20 ms")
        self._max_bytes = max_duration_ms // FRAME_DURATION_MS * PCM16_FRAME_BYTES
        self._data = bytearray()

    @property
    def duration_ms(self) -> int:
        return len(self._data) // PCM16_FRAME_BYTES * FRAME_DURATION_MS

    @property
    def frame_count(self) -> int:
        return len(self._data) // PCM16_FRAME_BYTES

    @property
    def empty(self) -> bool:
        return not self._data

    def append(self, frame: ReadableBuffer) -> None:
        view = memoryview(frame)
        if view.nbytes != PCM16_FRAME_BYTES:
            raise PCMFrameSizeError(
                f"expected {PCM16_FRAME_BYTES} bytes of PCM16, received {view.nbytes}"
            )
        if len(self._data) + view.nbytes > self._max_bytes:
            raise PCMBufferOverflow("PCM session buffer is full")
        self._data.extend(view)

    def clear(self) -> None:
        self._data.clear()

    def to_int16(self) -> NDArray[np.int16]:
        return np.frombuffer(self._data, dtype=_PCM16_LE).copy()

    def to_float32(self) -> NDArray[np.float32]:
        return pcm16_to_float32(self.to_int16())

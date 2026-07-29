"""Provider-neutral contracts and failures for streaming voice activity detection."""

from __future__ import annotations

from threading import Lock
from typing import Literal, Protocol, runtime_checkable

VADSampleRate = Literal[16_000, 24_000]
VAD_FRAME_DURATION_MS = 20


class VADError(RuntimeError):
    """Base class for operational VAD failures."""


class VADConfigurationError(VADError):
    """The selected provider or its immutable artifacts are invalid."""


class VADCapacityError(VADError):
    """A bounded stream or inference capacity limit was reached."""


class VADInferenceError(VADError):
    """A provider failed to classify a frame."""


class VADClosedError(VADError):
    """A provider or stream was used after it closed."""


@runtime_checkable
class VADStream(Protocol):
    """Connection-owned, sequential streaming classifier state."""

    async def score(self, pcm16_20ms: bytes) -> float:
        """Return a finite speech score in ``[0, 1]`` for one exact frame."""

    def reset(self) -> None:
        """Reset all model, resampler, framing, and held-score state."""

    def close(self) -> None:
        """Idempotently release this stream's provider lease."""


@runtime_checkable
class VADProvider(Protocol):
    """Process-owned immutable weights and bounded classification resources."""

    @property
    def name(self) -> str: ...

    @property
    def default_threshold(self) -> float: ...

    @property
    def active_streams(self) -> int: ...

    async def startup(self) -> None: ...

    def new_stream(self, input_sample_rate: VADSampleRate) -> VADStream: ...

    async def close(self) -> None: ...


class StreamCapacity:
    """Thread-safe live-stream lease counter with a hard upper bound."""

    __slots__ = ("_active", "_limit", "_lock")

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or limit <= 0:
            raise ValueError("stream limit must be positive")
        self._limit = limit
        self._active = 0
        self._lock = Lock()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def acquire(self) -> None:
        with self._lock:
            if self._active >= self._limit:
                raise VADCapacityError("VAD live-stream limit reached")
            self._active += 1

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1


def expected_frame_bytes(sample_rate: VADSampleRate) -> int:
    """Return bytes in one mono PCM16LE 20 ms frame at ``sample_rate``."""

    return sample_rate * 2 * VAD_FRAME_DURATION_MS // 1_000

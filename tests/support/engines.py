"""Engine doubles for scheduler and pipeline tests.

Every double here extends the real :class:`~app.engine.mock.MockEngine` and
produces its text, so tests never depend on model weights, ONNX Runtime, or
CUDA. The doubles only add *observability* (what was submitted, in what order)
and *controllability* (when a blocking call is allowed to finish).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any

from app.engine.base import (
    EngineState,
    ProgressCallback,
    TranscriptionRequest,
    TranscriptionResult,
)
from app.engine.mock import MockEngine
from tests.support.audio import float_audio
from tests.support.waiting import wait_until

DEFAULT_TIMEOUT_SECONDS = 5.0


def make_request(
    *,
    duration_ms: int = 100,
    language: str = "hi",
    decoder: str = "ctc",
    level: float = 0.25,
) -> TranscriptionRequest:
    """Build a valid transcription request without touching any audio file."""

    return TranscriptionRequest(
        audio=float_audio(duration_ms, level=level),
        sample_rate=16_000,
        language=language,
        decoder=decoder,
    )


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One observed engine invocation, reduced to comparable metadata."""

    language: str
    decoder: str
    audio_duration_ms: int
    batch_size: int


class RecordingMockEngine(MockEngine):
    """A MockEngine that records the metadata of every call it serves."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._calls: list[CallRecord] = []
        self._batch_sizes: list[int] = []

    @property
    def calls(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._calls)

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._batch_sizes)

    def _record(self, request: TranscriptionRequest, batch_size: int) -> None:
        duration_ms = round(request.audio.size * 1_000 / request.sample_rate)
        with self._lock:
            self._calls.append(
                CallRecord(
                    language=request.language,
                    decoder=request.decoder,
                    audio_duration_ms=duration_ms,
                    batch_size=batch_size,
                )
            )

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self._record(request, batch_size=1)
        return super().transcribe(request)


class BatchingMockEngine(RecordingMockEngine):
    """A recording MockEngine that also exposes the optional batch entry point."""

    def transcribe_batch(
        self, requests: Sequence[TranscriptionRequest]
    ) -> list[TranscriptionResult]:
        with self._lock:
            self._batch_sizes.append(len(requests))
        results: list[TranscriptionResult] = []
        for request in requests:
            self._record(request, batch_size=len(requests))
            results.append(MockEngine.transcribe(self, request))
        return results


class GatedMockEngine(BatchingMockEngine):
    """A MockEngine whose blocking call parks until a test releases it.

    This makes queue ordering, partial replacement, and backpressure observable
    without sleeping on wall-clock time.
    """

    def __init__(self, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        super().__init__()
        self._timeout_seconds = timeout_seconds
        self._entered_count = 0
        self._gate = threading.Event()

    @property
    def entered_count(self) -> int:
        with self._lock:
            return self._entered_count

    def open_gate(self) -> None:
        """Allow all current and future blocking calls to complete."""

        self._gate.set()

    def close_gate(self) -> None:
        self._gate.clear()

    async def wait_until_entered(self, count: int = 1) -> None:
        """Wait, without blocking the loop, for ``count`` calls to start."""

        await wait_until(
            lambda: self.entered_count >= count,
            description=f"{count} engine call(s) started",
            timeout_seconds=self._timeout_seconds,
        )

    def _park(self) -> None:
        with self._lock:
            self._entered_count += 1
        if not self._gate.wait(timeout=self._timeout_seconds):
            raise AssertionError("engine gate was never opened")

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self._park()
        return super().transcribe(request)

    def transcribe_batch(
        self, requests: Sequence[TranscriptionRequest]
    ) -> list[TranscriptionResult]:
        self._park()
        return super().transcribe_batch(requests)


class FailingMockEngine(MockEngine):
    """A MockEngine that raises a caller-supplied error from ``transcribe``."""

    def __init__(self, error_factory: Callable[[], Exception] | None = None) -> None:
        super().__init__()
        self._error_factory = error_factory or (lambda: RuntimeError("engine exploded"))
        self.call_count = 0

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        self.call_count += 1
        raise self._error_factory()


class ShortBatchMockEngine(MockEngine):
    """A MockEngine that returns fewer results than requests, a contract break."""

    def transcribe_batch(
        self, requests: Sequence[TranscriptionRequest]
    ) -> list[TranscriptionResult]:
        first = MockEngine.transcribe(self, requests[0])
        return [first]


class ScriptedTextEngine(MockEngine):
    """A MockEngine that returns caller-chosen text, in order, then repeats.

    MockEngine encodes the audio duration in its transcript, so consecutive
    hypotheses always differ. Scripting the text is the only way to observe the
    stable-prefix contract, including the case where a hypothesis is repeated.
    """

    def __init__(self, texts: Sequence[str]) -> None:
        super().__init__()
        if not texts:
            raise ValueError("texts must not be empty")
        self._texts = tuple(texts)
        self.call_count = 0

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        base = super().transcribe(request)
        text = self._texts[min(self.call_count, len(self._texts) - 1)]
        self.call_count += 1
        return TranscriptionResult(
            text=text,
            language=base.language,
            decoder=base.decoder,
            audio_duration_ms=base.audio_duration_ms,
            inference_ms=base.inference_ms,
        )


class _FakeAwaitable:
    """Awaitable that is never awaited, so no coroutine warning can leak."""

    def __await__(self) -> Generator[None, None, TranscriptionResult]:
        raise AssertionError("this awaitable must never be awaited")


class AwaitableMockEngine(MockEngine):
    """A misbehaving engine that returns an awaitable from a blocking method."""

    def transcribe(self, request: TranscriptionRequest) -> Any:
        return _FakeAwaitable()


class NeverReadyMockEngine(MockEngine):
    """A MockEngine whose startup completes without ever reporting readiness.

    Startup returning is not the same observable fact as the engine being ready,
    so the health route must report the engine's own state rather than assume a
    completed startup implies readiness.
    """

    async def startup(self, progress: ProgressCallback | None = None) -> None:
        self._set_readiness(EngineState.STARTING, "loading_weights")
        if progress is not None:
            progress("mock_starting")


class ProgressRecordingMockEngine(MockEngine):
    """A MockEngine that remembers the startup progress labels it emitted."""

    def __init__(self) -> None:
        super().__init__()
        self.progress_labels: list[str] = []

    async def startup(self, progress: ProgressCallback | None = None) -> None:
        def record(label: str) -> None:
            self.progress_labels.append(label)
            if progress is not None:
                progress(label)

        await super().startup(record)


__all__ = [
    "AwaitableMockEngine",
    "BatchingMockEngine",
    "CallRecord",
    "FailingMockEngine",
    "GatedMockEngine",
    "NeverReadyMockEngine",
    "ProgressRecordingMockEngine",
    "RecordingMockEngine",
    "ScriptedTextEngine",
    "ShortBatchMockEngine",
    "make_request",
]

"""Dependency-free timing spans for local tracing and metric observation."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from time import perf_counter
from types import TracebackType
from typing import Literal, Self

DurationObserver = Callable[[float], None]
Clock = Callable[[], float]


class TraceOperation(StrEnum):
    """Closed operation names; spans never carry request content."""

    QUEUE_WAIT = "queue_wait"
    PARTIAL = "partial"
    FINAL = "final"
    ENCODER = "encoder"
    CTC = "ctc"
    RNNT = "rnnt"


class TraceTimer:
    """A single-use sync/async context manager measuring monotonic time."""

    __slots__ = (
        "operation",
        "_clock",
        "_observer",
        "_started_at",
        "_duration_seconds",
        "_succeeded",
    )

    def __init__(
        self,
        operation: TraceOperation | str,
        observer: DurationObserver | None = None,
        *,
        clock: Clock = perf_counter,
    ) -> None:
        self.operation = TraceOperation(operation)
        self._clock = clock
        self._observer = observer
        self._started_at: float | None = None
        self._duration_seconds: float | None = None
        self._succeeded: bool | None = None

    @property
    def duration_seconds(self) -> float:
        if self._duration_seconds is None:
            raise RuntimeError("trace timer has not completed")
        return self._duration_seconds

    @property
    def succeeded(self) -> bool:
        if self._succeeded is None:
            raise RuntimeError("trace timer has not completed")
        return self._succeeded

    def __enter__(self) -> Self:
        if self._started_at is not None:
            raise RuntimeError("trace timer instances cannot be reused")
        self._started_at = self._clock()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exception, traceback
        if self._started_at is None:
            raise RuntimeError("trace timer was not started")
        self._duration_seconds = max(0.0, self._clock() - self._started_at)
        self._succeeded = exception_type is None
        if self._observer is not None:
            self._observer(self._duration_seconds)
        return False

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return self.__exit__(exception_type, exception, traceback)


def trace_timing(
    operation: TraceOperation | str,
    observer: DurationObserver | None = None,
    *,
    clock: Clock = perf_counter,
) -> TraceTimer:
    """Create a local timing span with an optional numeric-only observer."""

    return TraceTimer(operation, observer, clock=clock)

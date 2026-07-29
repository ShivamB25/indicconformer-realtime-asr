"""Bounded asynchronous dispatch for CPU voice activity classification."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, TypeVar

from app.vad.base import VADCapacityError, VADClosedError, VADInferenceError

_Result = TypeVar("_Result")


class VADRuntimeMetrics(Protocol):
    def vad_stream_started(self, provider: str) -> None: ...

    def vad_stream_ended(self, provider: str) -> None: ...

    def set_vad_queue_depth(self, depth: int) -> None: ...

    def record_vad_queue_wait(self, provider: str, seconds: float) -> None: ...

    def record_vad_inference(self, provider: str, seconds: float) -> None: ...

    def record_vad_runtime_error(self, provider: str, code: str) -> None: ...


@dataclass(slots=True)
class _Job:
    function: Callable[[], Any]
    future: asyncio.Future[Any]
    enqueued_at: float


class BoundedVADRuntime:
    """Run blocking classifiers off-loop with bounded queueing and deadlines."""

    __slots__ = (
        "_deadline_seconds",
        "_metrics",
        "_provider",
        "_queue",
        "_running",
        "_worker_count",
        "_workers",
    )

    def __init__(
        self,
        *,
        provider: str,
        workers: int,
        pending_capacity: int,
        deadline_seconds: float,
        metrics: VADRuntimeMetrics | None = None,
    ) -> None:
        if isinstance(workers, bool) or workers <= 0:
            raise ValueError("VAD workers must be positive")
        if isinstance(pending_capacity, bool) or pending_capacity <= 0:
            raise ValueError("VAD pending capacity must be positive")
        if deadline_seconds <= 0:
            raise ValueError("VAD classification deadline must be positive")
        self._provider = provider
        self._worker_count = workers
        self._deadline_seconds = deadline_seconds
        self._metrics = metrics
        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue(maxsize=pending_capacity)
        self._workers: list[asyncio.Task[None]] = []
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"vad-{self._provider}-{index}")
            for index in range(self._worker_count)
        ]

    async def submit(self, function: Callable[[], _Result]) -> _Result:
        if not self._running:
            raise VADClosedError("VAD runtime is not accepting work")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_Result] = loop.create_future()
        job = _Job(function=function, future=future, enqueued_at=perf_counter())
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            self._record_error("capacity")
            raise VADCapacityError("VAD classification queue is full") from exc
        self._set_depth()
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=self._deadline_seconds)
        except TimeoutError as exc:
            future.cancel()
            self._record_error("deadline")
            raise VADCapacityError("VAD classification deadline exceeded") from exc
        except asyncio.CancelledError:
            future.cancel()
            raise
        except VADInferenceError:
            raise
        except Exception as exc:
            raise VADInferenceError("VAD classification failed") from exc

    async def close(self) -> None:
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._set_depth()

    async def _worker(self, index: int) -> None:
        del index
        while True:
            job = await self._queue.get()
            self._set_depth()
            try:
                if job is None:
                    return
                if job.future.cancelled():
                    continue
                started_at = perf_counter()
                self._record_queue_wait(started_at - job.enqueued_at)
                try:
                    result = await asyncio.to_thread(job.function)
                except Exception as exc:
                    self._record_error("inference")
                    if not job.future.done():
                        job.future.set_exception(
                            VADInferenceError("VAD classifier raised an exception")
                        )
                    del exc
                else:
                    self._record_inference(perf_counter() - started_at)
                    if not job.future.done():
                        job.future.set_result(result)
            finally:
                self._queue.task_done()

    def _set_depth(self) -> None:
        if self._metrics is not None:
            self._metrics.set_vad_queue_depth(self._queue.qsize())

    def _record_queue_wait(self, seconds: float) -> None:
        if self._metrics is not None:
            self._metrics.record_vad_queue_wait(self._provider, seconds)

    def _record_inference(self, seconds: float) -> None:
        if self._metrics is not None:
            self._metrics.record_vad_inference(self._provider, seconds)

    def _record_error(self, code: str) -> None:
        if self._metrics is not None:
            self._metrics.record_vad_runtime_error(self._provider, code)

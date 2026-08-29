"""Bounded priority micro-batching for blocking transcription engines."""

from __future__ import annotations

import asyncio
import heapq
import inspect
import itertools
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

import anyio

from app.engine.base import Engine, TranscriptionRequest, TranscriptionResult


class QueueMetrics(Protocol):
    def set_queue_depth(self, depth: int) -> None: ...


class SchedulerError(RuntimeError):
    """Base class for scheduler failures."""


class ServerBusyError(SchedulerError):
    """The bounded queue cannot accept another inference request."""

    code = "SERVER_BUSY"


class SchedulerClosedError(SchedulerError):
    """The scheduler is not accepting jobs."""


class StalePartialError(SchedulerError):
    """A newer partial or final superseded this partial."""


class PartialOutstandingError(SchedulerError):
    """A session already has a partial actively running."""


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_queue_size: int = 64
    worker_count: int = 1
    max_batch_size: int = 8
    max_batch_audio_ms: int = 120_000
    batch_wait_ms: int = 8
    length_buckets_ms: tuple[int, ...] = (2_000, 5_000, 10_000, 30_000)

    def __post_init__(self) -> None:
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_batch_audio_ms <= 0:
            raise ValueError("max_batch_audio_ms must be positive")
        if self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms cannot be negative")
        if any(value <= 0 for value in self.length_buckets_ms):
            raise ValueError("length buckets must be positive")
        if tuple(sorted(set(self.length_buckets_ms))) != self.length_buckets_ms:
            raise ValueError("length buckets must be unique and increasing")


@dataclass(slots=True)
class _Job:
    session_id: str
    request: TranscriptionRequest
    kind: Literal["final", "partial"]
    priority: int
    sequence: int
    generation: int
    future: asyncio.Future[TranscriptionResult]
    audio_ms: int
    bucket: int
    stale: bool = False
    running: bool = False


@dataclass(order=True, slots=True)
class _HeapItem:
    priority: int
    sequence: int
    job: _Job = field(compare=False)


class InferenceScheduler:
    """Run blocking engine work off-loop with final-first bounded batching.

    There is at most one live partial per session. A newer partial replaces a
    queued one; it never queues behind a running partial. Submitting a final
    invalidates that session's partial before queue capacity is evaluated.
    """

    def __init__(
        self,
        engine: Engine,
        config: SchedulerConfig | None = None,
        *,
        metrics: QueueMetrics | None = None,
    ) -> None:
        self._engine = engine
        self.config = config or SchedulerConfig()
        self._condition = asyncio.Condition()
        self._heap: list[_HeapItem] = []
        self._sequence = itertools.count()
        self._queued_count = 0
        self._metrics = metrics
        self._publish_queue_depth()
        self._partials: dict[str, _Job] = {}
        self._generations: dict[str, int] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def queued_count(self) -> int:
        return self._queued_count

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(), name=f"asr-scheduler-{index}")
            for index in range(self.config.worker_count)
        ]

    async def close(self) -> None:
        if not self._running:
            return
        async with self._condition:
            self._running = False
            while self._heap:
                job = heapq.heappop(self._heap).job
                if job.stale:
                    continue
                job.stale = True
                if not job.future.done():
                    job.future.set_exception(SchedulerClosedError("scheduler closed"))
            self._queued_count = 0
            self._publish_queue_depth()
            self._condition.notify_all()
        # Running inference owns engine state. Let it finish and settle its
        # futures before lifespan is allowed to tear the engine down.
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._partials.clear()
        self._generations.clear()

    async def submit_final(
        self, session_id: str, request: TranscriptionRequest
    ) -> TranscriptionResult:
        """Submit a priority final, invalidating any partial for the session."""

        self._validate_session_id(session_id)
        async with self._condition:
            self._require_running()
            generation = self._generations.get(session_id, 0) + 1
            self._invalidate_partial(session_id)
            job = self._enqueue(session_id, request, "final", generation)
            self._generations[session_id] = generation
            self._condition.notify()
        return await self._await_job(job)

    async def submit_partial(
        self, session_id: str, request: TranscriptionRequest
    ) -> TranscriptionResult:
        """Submit a partial, replacing an older queued partial for the session."""

        self._validate_session_id(session_id)
        async with self._condition:
            self._require_running()
            previous = self._partials.get(session_id)
            if previous is not None:
                if previous.running:
                    raise PartialOutstandingError("a partial is already running for this session")
                self._invalidate_partial(session_id)
            generation = self._generations.get(session_id, 0)
            job = self._enqueue(session_id, request, "partial", generation)
            self._partials[session_id] = job
            self._condition.notify()
        return await self._await_job(job)

    async def _await_job(self, job: _Job) -> TranscriptionResult:
        try:
            return await job.future
        except asyncio.CancelledError:
            async with self._condition:
                if not job.running and not job.stale:
                    job.stale = True
                    self._queued_count -= 1
                    self._publish_queue_depth()
                    if self._partials.get(job.session_id) is job:
                        self._partials.pop(job.session_id, None)
                self._release_generation(job)
                self._condition.notify_all()
            raise

    def _enqueue(
        self,
        session_id: str,
        request: TranscriptionRequest,
        kind: Literal["final", "partial"],
        generation: int,
    ) -> _Job:
        if self._queued_count >= self.config.max_queue_size:
            raise ServerBusyError("inference queue is full")
        audio_ms = self._audio_duration_ms(request)
        job = _Job(
            session_id=session_id,
            request=request,
            kind=kind,
            priority=0 if kind == "final" else 1,
            sequence=next(self._sequence),
            generation=generation,
            future=asyncio.get_running_loop().create_future(),
            audio_ms=audio_ms,
            bucket=self._length_bucket(audio_ms),
        )
        heapq.heappush(self._heap, _HeapItem(job.priority, job.sequence, job))
        self._queued_count += 1
        self._publish_queue_depth()
        return job

    def _invalidate_partial(self, session_id: str) -> None:
        previous = self._partials.pop(session_id, None)
        if previous is None:
            return
        previous.stale = True
        if not previous.running:
            self._queued_count -= 1
            self._publish_queue_depth()
        if not previous.future.done():
            previous.future.set_exception(
                StalePartialError("partial superseded by newer session audio")
            )

    async def _worker(self) -> None:
        while True:
            batch = await self._next_batch()
            if not batch:
                return
            try:
                results = await anyio.to_thread.run_sync(self._run_batch, batch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                for job in batch:
                    self._finish(job, exception=exc)
            else:
                if len(results) != len(batch):
                    error = SchedulerError("engine returned an invalid batch size")
                    for job in batch:
                        self._finish(job, exception=error)
                    continue
                for job, result in zip(batch, results, strict=True):
                    self._finish(job, result=result)

    async def _next_batch(self) -> list[_Job]:
        async with self._condition:
            while True:
                await self._condition.wait_for(lambda: bool(self._heap) or not self._running)
                if not self._running:
                    return []
                if self.config.batch_wait_ms:
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=self.config.batch_wait_ms / 1_000,
                        )
                    except TimeoutError:
                        pass

                first = self._pop_live()
                if first is not None:
                    break
            first.running = True
            batch = [first]
            total_audio_ms = first.audio_ms

            selected: list[_HeapItem] = []
            retained: list[_HeapItem] = []
            while self._heap:
                item = heapq.heappop(self._heap)
                job = item.job
                if job.stale or job.future.cancelled():
                    if not job.stale:
                        self._queued_count -= 1
                        self._publish_queue_depth()
                        job.stale = True
                    self._release_generation(job)
                    continue
                compatible = (
                    job.priority == first.priority
                    and job.request.language == first.request.language
                    and job.request.decoder == first.request.decoder
                    and job.bucket == first.bucket
                    and len(batch) + len(selected) < self.config.max_batch_size
                    and total_audio_ms + job.audio_ms <= self.config.max_batch_audio_ms
                )
                if compatible:
                    selected.append(item)
                    total_audio_ms += job.audio_ms
                else:
                    retained.append(item)
            self._heap = retained
            heapq.heapify(self._heap)
            for item in selected:
                item.job.running = True
                batch.append(item.job)
            self._queued_count -= len(selected)
            self._publish_queue_depth()
            return batch

    def _pop_live(self) -> _Job | None:
        while self._heap:
            job = heapq.heappop(self._heap).job
            if job.stale or job.future.cancelled():
                if not job.stale:
                    self._queued_count -= 1
                    self._publish_queue_depth()
                    job.stale = True
                self._release_generation(job)
                continue
            self._queued_count -= 1
            self._publish_queue_depth()
            return job
        return None

    def _run_batch(self, jobs: list[_Job]) -> list[TranscriptionResult]:
        requests = [job.request for job in jobs]
        batch_method = getattr(self._engine, "transcribe_batch", None)
        if callable(batch_method) and len(requests) > 1:
            returned = batch_method(requests)
            if inspect.isawaitable(returned):
                raise SchedulerError("blocking engine returned an awaitable")
            return list(cast(Any, returned))

        results: list[TranscriptionResult] = []
        for request in requests:
            returned = self._engine.transcribe(request)
            if inspect.isawaitable(returned):
                raise SchedulerError("blocking engine returned an awaitable")
            results.append(returned)
        return results

    def _finish(
        self,
        job: _Job,
        *,
        result: TranscriptionResult | None = None,
        exception: Exception | None = None,
    ) -> None:
        if self._partials.get(job.session_id) is job:
            self._partials.pop(job.session_id, None)
        stale = job.stale or (
            job.kind == "partial" and job.generation != self._generations.get(job.session_id, 0)
        )
        if job.future.done():
            self._release_generation(job)
            return
        if stale:
            job.future.set_exception(StalePartialError("partial result is stale"))
        elif exception is not None:
            job.future.set_exception(exception)
        elif result is not None:
            job.future.set_result(result)
        else:
            job.future.set_exception(SchedulerError("inference completed without a result"))
        self._release_generation(job)

    def _release_generation(self, job: _Job) -> None:
        if self._generations.get(job.session_id) != job.generation:
            return
        if job.kind == "partial" and self._partials.get(job.session_id) is not None:
            return
        if any(item.job.session_id == job.session_id and not item.job.stale for item in self._heap):
            return
        self._generations.pop(job.session_id, None)

    def _length_bucket(self, audio_ms: int) -> int:
        for index, upper_bound in enumerate(self.config.length_buckets_ms):
            if audio_ms <= upper_bound:
                return index
        return len(self.config.length_buckets_ms)

    @staticmethod
    def _audio_duration_ms(request: TranscriptionRequest) -> int:
        if request.sample_rate <= 0:
            raise ValueError("request sample_rate must be positive")
        if request.audio.ndim != 1:
            raise ValueError("request audio must be mono")
        return round(request.audio.size * 1_000 / request.sample_rate)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not session_id or len(session_id) > 128:
            raise ValueError("session_id must contain 1 to 128 characters")

    def _publish_queue_depth(self) -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.set_queue_depth(self._queued_count)
        except Exception:
            # Metrics are observational and must never alter scheduler state or cleanup.
            pass

    def _require_running(self) -> None:
        if not self._running:
            raise SchedulerClosedError("scheduler is not running")

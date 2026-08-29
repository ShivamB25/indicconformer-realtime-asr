"""Contracts for bounded CPU VAD dispatch and stream leases."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest

from app.vad.base import (
    StreamCapacity,
    VADCapacityError,
    VADClosedError,
    VADInferenceError,
    expected_frame_bytes,
)
from app.vad.runtime import BoundedVADRuntime


@dataclass(slots=True)
class RuntimeMetrics:
    depths: list[int] = field(default_factory=list)
    queue_waits: list[tuple[str, float]] = field(default_factory=list)
    streams: list[tuple[str, str]] = field(default_factory=list)

    def vad_stream_started(self, provider: str) -> None:
        self.streams.append((provider, "started"))

    def vad_stream_ended(self, provider: str) -> None:
        self.streams.append((provider, "ended"))

    inference: list[tuple[str, float]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    queue_waited: asyncio.Event = field(default_factory=asyncio.Event)

    def set_vad_queue_depth(self, depth: int) -> None:
        self.depths.append(depth)

    def record_vad_queue_wait(self, provider: str, seconds: float) -> None:
        self.queue_waits.append((provider, seconds))
        self.queue_waited.set()

    def record_vad_inference(self, provider: str, seconds: float) -> None:
        self.inference.append((provider, seconds))

    def record_vad_runtime_error(self, provider: str, code: str) -> None:
        self.errors.append((provider, code))


def runtime(
    *,
    workers: int = 1,
    capacity: int = 1,
    deadline: float = 1.0,
    metrics: RuntimeMetrics | None = None,
) -> BoundedVADRuntime:
    return BoundedVADRuntime(
        provider="test",
        workers=workers,
        pending_capacity=capacity,
        deadline_seconds=deadline,
        metrics=metrics,
    )


def test_frame_sizes_and_stream_capacity_are_hard_bounds() -> None:
    assert expected_frame_bytes(16_000) == 640
    assert expected_frame_bytes(24_000) == 960

    capacity = StreamCapacity(1)
    capacity.acquire()
    assert capacity.active == 1
    with pytest.raises(VADCapacityError, match="live-stream limit"):
        capacity.acquire()
    capacity.release()
    capacity.release()
    assert capacity.active == 0


@pytest.mark.asyncio
async def test_runtime_rejects_use_before_start_and_after_close() -> None:
    dispatcher = runtime()
    with pytest.raises(VADClosedError, match="not accepting"):
        await dispatcher.submit(lambda: 1)
    await dispatcher.start()
    assert await dispatcher.submit(lambda: 7) == 7
    await dispatcher.close()
    with pytest.raises(VADClosedError, match="not accepting"):
        await dispatcher.submit(lambda: 2)


@pytest.mark.asyncio
async def test_runtime_bounds_pending_work_without_dropping_accepted_jobs() -> None:
    dispatcher = runtime()
    entered = threading.Event()
    release = threading.Event()

    def blocked() -> int:
        entered.set()
        assert release.wait(timeout=2.0)
        return 1

    await dispatcher.start()
    first = asyncio.create_task(dispatcher.submit(blocked))
    assert await asyncio.to_thread(entered.wait, 1.0)
    second = asyncio.create_task(dispatcher.submit(lambda: 2))
    await asyncio.sleep(0)
    with pytest.raises(VADCapacityError, match="queue is full"):
        await dispatcher.submit(lambda: 3)
    release.set()
    assert await first == 1
    assert await second == 2
    await dispatcher.close()


def _slow_result() -> int:
    time.sleep(0.05)
    return 1


@pytest.mark.asyncio
async def test_runtime_deadline_is_capacity_failure_and_late_result_is_ignored() -> None:
    metrics = RuntimeMetrics()
    dispatcher = runtime(deadline=0.01, metrics=metrics)
    await dispatcher.start()
    with pytest.raises(VADCapacityError, match="deadline exceeded"):
        await dispatcher.submit(_slow_result)
    await asyncio.sleep(0.06)
    assert ("test", "deadline") in metrics.errors
    await dispatcher.close()


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_runtime_skips_executor_queued_work_after_abandonment(
    cancel_caller: bool,
) -> None:
    asyncio.run(_assert_executor_queued_work_is_skipped(cancel_caller))


async def _assert_executor_queued_work_is_skipped(cancel_caller: bool) -> None:
    loop = asyncio.get_running_loop()
    occupied = asyncio.Event()
    release = threading.Event()
    late_started = threading.Event()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))

    def occupy_executor() -> None:
        loop.call_soon_threadsafe(occupied.set)
        assert release.wait(timeout=2.0)

    def late_job() -> int:
        late_started.set()
        return 1

    blocker = loop.run_in_executor(None, occupy_executor)
    metrics = RuntimeMetrics()
    dispatcher = runtime(deadline=1.0 if cancel_caller else 0.05, metrics=metrics)
    await dispatcher.start()
    try:
        await occupied.wait()
        submission = asyncio.create_task(dispatcher.submit(late_job))
        await metrics.queue_waited.wait()

        if cancel_caller:
            submission.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submission
        else:
            with pytest.raises(VADCapacityError, match="deadline exceeded"):
                await submission
        assert not late_started.is_set()

        release.set()
        await blocker
        await dispatcher.close()
        assert not late_started.is_set()
        assert metrics.inference == []
    finally:
        release.set()
        await asyncio.gather(blocker, return_exceptions=True)
        await dispatcher.close()


@pytest.mark.asyncio
async def test_runtime_wraps_classifier_fault_and_records_bounded_metrics() -> None:
    metrics = RuntimeMetrics()
    dispatcher = runtime(metrics=metrics)

    def explode() -> float:
        raise ValueError("private implementation detail")

    await dispatcher.start()
    with pytest.raises(VADInferenceError, match="classifier raised"):
        await dispatcher.submit(explode)
    await dispatcher.close()

    assert ("test", "inference") in metrics.errors
    assert metrics.depths
    assert all(depth >= 0 for depth in metrics.depths)
    assert len(metrics.queue_waits) == 1
    assert metrics.queue_waits[0][0] == "test"

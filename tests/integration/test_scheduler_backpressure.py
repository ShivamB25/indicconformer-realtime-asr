"""Bounded-queue backpressure.

Rejecting work is a feature: an overloaded service must fail a request quickly
with a retryable signal rather than queue without limit. Capacity accounting is
asserted alongside the rejection, because a leaked slot degrades into a service
that refuses everything.
"""

from __future__ import annotations

import asyncio

import pytest

from app.engine.base import TranscriptionResult
from app.engine.scheduler import (
    InferenceScheduler,
    SchedulerConfig,
    ServerBusyError,
    StalePartialError,
)
from tests.support.engines import GatedMockEngine, make_request
from tests.support.waiting import wait_until

SINGLE_SLOT = SchedulerConfig(max_queue_size=1, worker_count=1, batch_wait_ms=0)


async def occupy_worker(
    scheduler: InferenceScheduler, engine: GatedMockEngine
) -> asyncio.Task[TranscriptionResult]:
    """Park one job inside the engine so the queue can be observed filling."""

    blocker = asyncio.create_task(scheduler.submit_final("blocker", make_request(duration_ms=900)))
    await engine.wait_until_entered()
    return blocker


class TestQueueLimit:
    async def test_a_full_queue_rejects_further_finals(self, gated_engine: GatedMockEngine) -> None:
        scheduler = InferenceScheduler(gated_engine, SINGLE_SLOT)
        await scheduler.start()
        try:
            blocker = await occupy_worker(scheduler, gated_engine)
            queued = asyncio.create_task(scheduler.submit_final("s1", make_request()))
            await wait_until(lambda: scheduler.queued_count == 1, description="the queue is full")

            with pytest.raises(ServerBusyError, match="inference queue is full"):
                await scheduler.submit_final("s2", make_request())

            gated_engine.open_gate()
            await asyncio.gather(blocker, queued)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

    async def test_a_full_queue_rejects_further_partials(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, SINGLE_SLOT)
        await scheduler.start()
        try:
            blocker = await occupy_worker(scheduler, gated_engine)
            queued = asyncio.create_task(scheduler.submit_final("s1", make_request()))
            await wait_until(lambda: scheduler.queued_count == 1, description="the queue is full")

            with pytest.raises(ServerBusyError):
                await scheduler.submit_partial("s2", make_request())

            gated_engine.open_gate()
            await asyncio.gather(blocker, queued)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

    async def test_a_rejected_submission_never_reaches_the_engine(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, SINGLE_SLOT)
        await scheduler.start()
        try:
            blocker = await occupy_worker(scheduler, gated_engine)
            queued = asyncio.create_task(
                scheduler.submit_final("s1", make_request(duration_ms=300))
            )
            await wait_until(lambda: scheduler.queued_count == 1, description="the queue is full")
            with pytest.raises(ServerBusyError):
                await scheduler.submit_final("rejected", make_request(duration_ms=500))

            gated_engine.open_gate()
            await asyncio.gather(blocker, queued)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert 500 not in [call.audio_duration_ms for call in gated_engine.calls]

    async def test_capacity_is_released_when_work_completes(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, SINGLE_SLOT)
        await scheduler.start()
        try:
            blocker = await occupy_worker(scheduler, gated_engine)
            queued = asyncio.create_task(scheduler.submit_final("s1", make_request()))
            await wait_until(lambda: scheduler.queued_count == 1, description="the queue is full")
            with pytest.raises(ServerBusyError):
                await scheduler.submit_final("s2", make_request())

            gated_engine.open_gate()
            await asyncio.gather(blocker, queued)
            await wait_until(lambda: scheduler.queued_count == 0, description="the queue drains")

            # The scheduler is healthy again, not stuck in a rejecting state.
            assert (await scheduler.submit_final("s3", make_request())).text
        finally:
            gated_engine.open_gate()
            await scheduler.close()

    async def test_a_rejection_does_not_disturb_queued_work(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, SINGLE_SLOT)
        await scheduler.start()
        try:
            blocker = await occupy_worker(scheduler, gated_engine)
            queued = asyncio.create_task(
                scheduler.submit_final("s1", make_request(duration_ms=300))
            )
            await wait_until(lambda: scheduler.queued_count == 1, description="the queue is full")
            for _ in range(3):
                with pytest.raises(ServerBusyError):
                    await scheduler.submit_final("noisy", make_request())

            gated_engine.open_gate()
            results = await asyncio.gather(blocker, queued)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert [result.audio_duration_ms for result in results] == [900, 300]


class TestFinalPriorityUnderBackpressure:
    async def test_a_final_reclaims_its_own_sessions_partial_slot(
        self, gated_engine: GatedMockEngine
    ) -> None:
        """A session can always finalize, even when the queue is otherwise full."""

        scheduler = InferenceScheduler(gated_engine, SINGLE_SLOT)
        await scheduler.start()
        try:
            blocker = await occupy_worker(scheduler, gated_engine)
            partial = asyncio.create_task(
                scheduler.submit_partial("s1", make_request(duration_ms=100))
            )
            await wait_until(
                lambda: scheduler.queued_count == 1,
                description="the partial fills the only slot",
            )

            final = asyncio.create_task(scheduler.submit_final("s1", make_request(duration_ms=300)))
            with pytest.raises(StalePartialError):
                await partial

            gated_engine.open_gate()
            assert (await final).audio_duration_ms == 300
            await blocker
        finally:
            gated_engine.open_gate()
            await scheduler.close()

    async def test_another_sessions_final_is_still_rejected_when_full(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, SINGLE_SLOT)
        await scheduler.start()
        try:
            blocker = await occupy_worker(scheduler, gated_engine)
            partial = asyncio.create_task(scheduler.submit_partial("s1", make_request()))
            await wait_until(lambda: scheduler.queued_count == 1, description="the queue is full")

            with pytest.raises(ServerBusyError):
                await scheduler.submit_final("other-session", make_request())

            gated_engine.open_gate()
            await asyncio.gather(blocker, partial)
        finally:
            gated_engine.open_gate()
            await scheduler.close()


class TestCapacityAccounting:
    async def test_the_queue_never_exceeds_its_configured_depth(
        self, gated_engine: GatedMockEngine
    ) -> None:
        config = SchedulerConfig(
            max_queue_size=3, worker_count=1, batch_wait_ms=0, max_batch_size=1
        )
        scheduler = InferenceScheduler(gated_engine, config)
        await scheduler.start()
        accepted: list[asyncio.Task[object]] = []
        rejections = 0
        try:
            blocker = await occupy_worker(scheduler, gated_engine)
            for index in range(6):
                try:
                    task = asyncio.create_task(scheduler.submit_final(f"s{index}", make_request()))
                    await asyncio.sleep(0)
                    if task.done() and isinstance(task.exception(), ServerBusyError):
                        rejections += 1
                        continue
                    accepted.append(task)
                except ServerBusyError:
                    rejections += 1
                assert scheduler.queued_count <= 3

            gated_engine.open_gate()
            await asyncio.gather(blocker, *accepted)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert len(accepted) == 3
        assert rejections == 3

    async def test_the_queue_is_empty_again_after_a_drain(
        self, gated_engine: GatedMockEngine
    ) -> None:
        gated_engine.open_gate()
        scheduler = InferenceScheduler(gated_engine, SchedulerConfig(batch_wait_ms=0))
        await scheduler.start()
        try:
            for index in range(5):
                await scheduler.submit_final(f"s{index}", make_request())
            assert scheduler.queued_count == 0
        finally:
            await scheduler.close()

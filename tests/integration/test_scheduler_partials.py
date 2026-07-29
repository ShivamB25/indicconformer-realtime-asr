"""Partial-inference lifecycle: at most one live partial per session.

These are the rules that keep a realtime session from emitting a transcript
older than one it already sent: a newer partial replaces a queued one, a final
invalidates any partial, and a superseded partial reports itself as stale
instead of returning text.
"""

from __future__ import annotations

import asyncio

import pytest

from app.engine.base import TranscriptionResult
from app.engine.scheduler import (
    InferenceScheduler,
    PartialOutstandingError,
    SchedulerConfig,
    SchedulerError,
    ServerBusyError,
    StalePartialError,
)
from tests.conftest import SchedulerFactory
from tests.support.engines import BatchingMockEngine, GatedMockEngine, make_request
from tests.support.waiting import wait_until

IMMEDIATE = SchedulerConfig(batch_wait_ms=0)


class TestErrorContract:
    @pytest.mark.parametrize("error", [StalePartialError, PartialOutstandingError, ServerBusyError])
    def test_every_scheduler_rejection_is_a_scheduler_error(self, error: type[Exception]) -> None:
        assert issubclass(error, SchedulerError)

    def test_the_busy_error_carries_a_stable_wire_code(self) -> None:
        assert ServerBusyError.code == "SERVER_BUSY"


class TestPartialSubmission:
    async def test_a_partial_returns_the_engine_result(
        self, scheduler_factory: SchedulerFactory, recording_engine: BatchingMockEngine
    ) -> None:
        scheduler = await scheduler_factory(recording_engine, IMMEDIATE)
        result = await scheduler.submit_partial("s1", make_request(duration_ms=200))
        assert result.audio_duration_ms == 200
        assert scheduler.queued_count == 0

    async def test_partials_can_be_submitted_repeatedly_when_each_completes(
        self, scheduler_factory: SchedulerFactory, recording_engine: BatchingMockEngine
    ) -> None:
        scheduler = await scheduler_factory(recording_engine, IMMEDIATE)
        for duration_ms in (100, 200, 300):
            result = await scheduler.submit_partial("s1", make_request(duration_ms=duration_ms))
            assert result.audio_duration_ms == duration_ms
        assert len(recording_engine.calls) == 3


class TestPartialReplacement:
    async def test_a_newer_partial_replaces_a_queued_one(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(
                scheduler.submit_final("block", make_request(duration_ms=900))
            )
            await gated_engine.wait_until_entered()

            first = asyncio.create_task(
                scheduler.submit_partial("s1", make_request(duration_ms=100))
            )
            await wait_until(
                lambda: scheduler.queued_count == 1, description="the first partial is queued"
            )
            second = asyncio.create_task(
                scheduler.submit_partial("s1", make_request(duration_ms=200))
            )
            await wait_until(
                lambda: scheduler.queued_count == 1,
                description="the replacement occupies the single slot",
            )

            with pytest.raises(StalePartialError):
                await first

            gated_engine.open_gate()
            assert (await second).audio_duration_ms == 200
            await blocker
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        # The superseded partial never reached the engine.
        durations = [call.audio_duration_ms for call in gated_engine.calls]
        assert durations == [900, 200]

    async def test_replacement_does_not_leak_queue_capacity(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(
                scheduler.submit_final("block", make_request(duration_ms=900))
            )
            await gated_engine.wait_until_entered()

            superseded: list[asyncio.Task[TranscriptionResult]] = []
            for duration_ms in (100, 120, 140, 160):
                superseded.append(
                    asyncio.create_task(
                        scheduler.submit_partial("s1", make_request(duration_ms=duration_ms))
                    )
                )
                await wait_until(
                    lambda: scheduler.queued_count == 1,
                    description="only one partial is ever queued",
                )

            for task in superseded[:-1]:
                with pytest.raises(StalePartialError):
                    await task

            gated_engine.open_gate()
            assert (await superseded[-1]).audio_duration_ms == 160
            await blocker
        finally:
            gated_engine.open_gate()
            await scheduler.close()

    async def test_a_running_partial_is_not_replaced_but_reported(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            running = asyncio.create_task(scheduler.submit_partial("s1", make_request()))
            await gated_engine.wait_until_entered()

            with pytest.raises(PartialOutstandingError):
                await scheduler.submit_partial("s1", make_request(duration_ms=200))

            gated_engine.open_gate()
            await running
        finally:
            gated_engine.open_gate()
            await scheduler.close()

    async def test_a_new_partial_is_accepted_once_the_previous_one_finishes(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        gated_engine.open_gate()
        await scheduler.start()
        try:
            await scheduler.submit_partial("s1", make_request(duration_ms=100))
            second = await scheduler.submit_partial("s1", make_request(duration_ms=200))
            assert second.audio_duration_ms == 200
        finally:
            await scheduler.close()

    async def test_sessions_do_not_replace_each_others_partials(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(
                scheduler.submit_final("block", make_request(duration_ms=900))
            )
            await gated_engine.wait_until_entered()

            first = asyncio.create_task(scheduler.submit_partial("s1", make_request(language="hi")))
            await wait_until(
                lambda: scheduler.queued_count == 1, description="one partial is queued"
            )
            second = asyncio.create_task(
                scheduler.submit_partial("s2", make_request(language="ta"))
            )
            await wait_until(
                lambda: scheduler.queued_count == 2, description="both partials are queued"
            )

            gated_engine.open_gate()
            results = await asyncio.gather(first, second)
            await blocker
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert {result.language for result in results} == {"hi", "ta"}


class TestFinalInvalidatesPartials:
    async def test_a_final_supersedes_a_queued_partial(self, gated_engine: GatedMockEngine) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(
                scheduler.submit_final("block", make_request(duration_ms=900))
            )
            await gated_engine.wait_until_entered()

            partial = asyncio.create_task(
                scheduler.submit_partial("s1", make_request(duration_ms=100))
            )
            await wait_until(
                lambda: scheduler.queued_count == 1, description="the partial is queued"
            )
            final = asyncio.create_task(scheduler.submit_final("s1", make_request(duration_ms=300)))
            await wait_until(
                lambda: scheduler.queued_count == 1,
                description="the final replaced the partial in the queue",
            )

            with pytest.raises(StalePartialError):
                await partial

            gated_engine.open_gate()
            assert (await final).audio_duration_ms == 300
            await blocker
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert [call.audio_duration_ms for call in gated_engine.calls] == [900, 300]

    async def test_a_final_supersedes_a_partial_that_is_already_running(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            partial = asyncio.create_task(
                scheduler.submit_partial("s1", make_request(duration_ms=100))
            )
            await gated_engine.wait_until_entered()

            final = asyncio.create_task(scheduler.submit_final("s1", make_request(duration_ms=400)))
            with pytest.raises(StalePartialError):
                await partial

            gated_engine.open_gate()
            assert (await final).audio_duration_ms == 400
        finally:
            gated_engine.open_gate()
            await scheduler.close()

    async def test_a_partial_after_a_final_starts_a_fresh_generation(
        self, scheduler_factory: SchedulerFactory, recording_engine: BatchingMockEngine
    ) -> None:
        scheduler = await scheduler_factory(recording_engine, IMMEDIATE)
        await scheduler.submit_final("s1", make_request(duration_ms=500))
        result = await scheduler.submit_partial("s1", make_request(duration_ms=100))
        assert result.audio_duration_ms == 100

    async def test_a_partial_queued_behind_a_final_still_runs_after_it(
        self, gated_engine: GatedMockEngine
    ) -> None:
        """Only partials that already exist are invalidated, and finals go first."""

        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(
                scheduler.submit_final("block", make_request(duration_ms=900))
            )
            await gated_engine.wait_until_entered()

            final = asyncio.create_task(scheduler.submit_final("s1", make_request(duration_ms=300)))
            await wait_until(lambda: scheduler.queued_count == 1, description="the final is queued")
            partial = asyncio.create_task(
                scheduler.submit_partial("s1", make_request(duration_ms=100))
            )
            await wait_until(
                lambda: scheduler.queued_count == 2, description="both jobs are queued"
            )

            gated_engine.open_gate()
            assert (await final).audio_duration_ms == 300
            assert (await partial).audio_duration_ms == 100
            await blocker
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert [call.audio_duration_ms for call in gated_engine.calls] == [900, 300, 100]

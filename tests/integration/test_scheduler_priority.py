"""Scheduler lifecycle, priority, and batching contracts.

The engine used here is a MockEngine whose blocking call parks on a gate, so
"what the scheduler chose to run next" is observable without timing guesses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from app.engine.scheduler import (
    InferenceScheduler,
    SchedulerClosedError,
    SchedulerConfig,
    SchedulerError,
)
from tests.conftest import SchedulerFactory
from tests.support.engines import (
    AwaitableMockEngine,
    BatchingMockEngine,
    FailingMockEngine,
    GatedMockEngine,
    ShortBatchMockEngine,
    make_request,
)
from tests.support.waiting import wait_until


def queue_depth_is(scheduler: InferenceScheduler, expected: int) -> Callable[[], bool]:
    """Bind the expected depth once, so the predicate cannot read a later value."""

    return lambda: scheduler.queued_count == expected


IMMEDIATE = SchedulerConfig(batch_wait_ms=0)


class TestConfiguration:
    def test_defaults_are_the_documented_bounds(self) -> None:
        config = SchedulerConfig()
        assert config.max_queue_size == 64
        assert config.worker_count == 1
        assert config.max_batch_size == 8
        assert config.max_batch_audio_ms == 120_000
        assert config.batch_wait_ms == 8
        assert config.length_buckets_ms == (2_000, 5_000, 10_000, 30_000)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"max_queue_size": 0}, "max_queue_size must be positive"),
            ({"worker_count": 0}, "worker_count must be positive"),
            ({"max_batch_size": -1}, "max_batch_size must be positive"),
            ({"max_batch_audio_ms": 0}, "max_batch_audio_ms must be positive"),
            ({"batch_wait_ms": -1}, "batch_wait_ms cannot be negative"),
            ({"length_buckets_ms": (0, 100)}, "length buckets must be positive"),
            ({"length_buckets_ms": (100, 100)}, "unique and increasing"),
            ({"length_buckets_ms": (200, 100)}, "unique and increasing"),
        ],
    )
    def test_invalid_configuration_is_refused(
        self, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SchedulerConfig(**overrides)  # type: ignore[arg-type]

    def test_zero_batch_wait_is_allowed(self) -> None:
        assert SchedulerConfig(batch_wait_ms=0).batch_wait_ms == 0


class TestLifecycle:
    async def test_a_new_scheduler_is_not_running(
        self, recording_engine: BatchingMockEngine
    ) -> None:
        scheduler = InferenceScheduler(recording_engine)
        assert scheduler.running is False
        assert scheduler.queued_count == 0

    async def test_submitting_before_start_is_refused(
        self, recording_engine: BatchingMockEngine
    ) -> None:
        scheduler = InferenceScheduler(recording_engine)
        with pytest.raises(SchedulerClosedError):
            await scheduler.submit_final("s1", make_request())
        with pytest.raises(SchedulerClosedError):
            await scheduler.submit_partial("s1", make_request())

    async def test_start_is_idempotent(self, scheduler_factory: SchedulerFactory) -> None:
        engine = BatchingMockEngine()
        await engine.startup()
        scheduler = await scheduler_factory(engine, IMMEDIATE)
        await scheduler.start()
        assert scheduler.running is True

    async def test_close_is_idempotent_and_stops_accepting_work(
        self, recording_engine: BatchingMockEngine
    ) -> None:
        scheduler = InferenceScheduler(recording_engine, IMMEDIATE)
        await scheduler.start()
        await scheduler.close()
        await scheduler.close()
        assert scheduler.running is False
        with pytest.raises(SchedulerClosedError):
            await scheduler.submit_final("s1", make_request())

    async def test_queued_work_is_failed_when_the_scheduler_closes(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()

        blocker = asyncio.create_task(scheduler.submit_final("block", make_request()))
        await gated_engine.wait_until_entered()
        queued = asyncio.create_task(scheduler.submit_final("s1", make_request()))
        await wait_until(lambda: scheduler.queued_count == 1, description="one job is queued")

        await scheduler.close()
        with pytest.raises(SchedulerClosedError):
            await queued

        gated_engine.open_gate()
        blocker.cancel()
        await asyncio.gather(blocker, return_exceptions=True)

    async def test_a_completed_request_returns_the_engine_result(
        self, scheduler_factory: SchedulerFactory, recording_engine: BatchingMockEngine
    ) -> None:
        scheduler = await scheduler_factory(recording_engine, IMMEDIATE)
        result = await scheduler.submit_final(
            "s1", make_request(duration_ms=500, language="ta", decoder="rnnt")
        )
        assert result.language == "ta"
        assert result.decoder == "rnnt"
        assert result.audio_duration_ms == 500
        assert result.text == "mock transcript language=ta decoder=rnnt duration_ms=500"
        assert scheduler.queued_count == 0

    @pytest.mark.parametrize("session_id", ["", "x" * 129])
    async def test_session_ids_are_bounded(
        self, scheduler_factory: SchedulerFactory, session_id: str
    ) -> None:
        engine = BatchingMockEngine()
        await engine.startup()
        scheduler = await scheduler_factory(engine, IMMEDIATE)
        with pytest.raises(ValueError, match="session_id must contain 1 to 128"):
            await scheduler.submit_final(session_id, make_request())
        with pytest.raises(ValueError, match="session_id must contain 1 to 128"):
            await scheduler.submit_partial(session_id, make_request())


class TestPriority:
    async def test_a_final_overtakes_an_already_queued_partial(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(
                scheduler.submit_final("block", make_request(language="hi"))
            )
            await gated_engine.wait_until_entered()

            partial = asyncio.create_task(
                scheduler.submit_partial("partial-session", make_request(language="bn"))
            )
            await wait_until(
                lambda: scheduler.queued_count == 1, description="the partial is queued"
            )
            final = asyncio.create_task(
                scheduler.submit_final("final-session", make_request(language="ta"))
            )
            await wait_until(
                lambda: scheduler.queued_count == 2, description="both jobs are queued"
            )

            gated_engine.open_gate()
            await asyncio.gather(blocker, final, partial)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert [call.language for call in gated_engine.calls] == ["hi", "ta", "bn"]

    async def test_equal_priority_work_runs_in_submission_order(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(
            gated_engine, SchedulerConfig(batch_wait_ms=0, max_batch_size=1)
        )
        await scheduler.start()
        try:
            blocker = asyncio.create_task(scheduler.submit_final("block", make_request()))
            await gated_engine.wait_until_entered()

            tasks = []
            for index, language in enumerate(("as", "bn", "gu")):
                tasks.append(
                    asyncio.create_task(
                        scheduler.submit_final(f"s{index}", make_request(language=language))
                    )
                )
                await wait_until(
                    queue_depth_is(scheduler, index + 1),
                    description="the submission is queued",
                )

            gated_engine.open_gate()
            await asyncio.gather(blocker, *tasks)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert [call.language for call in gated_engine.calls] == ["hi", "as", "bn", "gu"]

    async def test_several_workers_run_concurrently(self, gated_engine: GatedMockEngine) -> None:
        scheduler = InferenceScheduler(
            gated_engine,
            SchedulerConfig(worker_count=2, batch_wait_ms=0, max_batch_size=1),
        )
        await scheduler.start()
        try:
            # Different languages cannot share a batch, so each needs its own worker.
            first = asyncio.create_task(scheduler.submit_final("a", make_request(language="hi")))
            second = asyncio.create_task(scheduler.submit_final("b", make_request(language="ta")))
            await gated_engine.wait_until_entered(2)

            gated_engine.open_gate()
            results = await asyncio.gather(first, second)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert {result.language for result in results} == {"hi", "ta"}


class TestBatching:
    async def test_compatible_finals_are_batched_together(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(scheduler.submit_final("block", make_request()))
            await gated_engine.wait_until_entered()

            tasks = [
                asyncio.create_task(
                    scheduler.submit_final(f"s{index}", make_request(duration_ms=1_000))
                )
                for index in range(3)
            ]
            await wait_until(
                lambda: scheduler.queued_count == 3, description="three jobs are queued"
            )

            gated_engine.open_gate()
            await asyncio.gather(blocker, *tasks)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert 3 in gated_engine.batch_sizes

    async def test_different_languages_are_never_batched(
        self, gated_engine: GatedMockEngine
    ) -> None:
        await self._assert_not_batched(
            gated_engine,
            [make_request(language="hi"), make_request(language="ta")],
        )

    async def test_different_decoders_are_never_batched(
        self, gated_engine: GatedMockEngine
    ) -> None:
        await self._assert_not_batched(
            gated_engine,
            [make_request(decoder="ctc"), make_request(decoder="rnnt")],
        )

    async def test_different_length_buckets_are_never_batched(
        self, gated_engine: GatedMockEngine
    ) -> None:
        await self._assert_not_batched(
            gated_engine,
            [make_request(duration_ms=1_000), make_request(duration_ms=6_000)],
        )

    async def test_a_batch_never_exceeds_the_audio_budget(
        self, gated_engine: GatedMockEngine
    ) -> None:
        await self._assert_not_batched(
            gated_engine,
            [make_request(duration_ms=1_000), make_request(duration_ms=1_000)],
            config=SchedulerConfig(batch_wait_ms=0, max_batch_audio_ms=1_500),
        )

    async def test_a_batch_never_exceeds_the_configured_size(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(
            gated_engine, SchedulerConfig(batch_wait_ms=0, max_batch_size=2)
        )
        await scheduler.start()
        try:
            blocker = asyncio.create_task(scheduler.submit_final("block", make_request()))
            await gated_engine.wait_until_entered()
            tasks = [
                asyncio.create_task(scheduler.submit_final(f"s{index}", make_request()))
                for index in range(4)
            ]
            await wait_until(
                lambda: scheduler.queued_count == 4, description="four jobs are queued"
            )
            gated_engine.open_gate()
            await asyncio.gather(blocker, *tasks)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        assert gated_engine.batch_sizes
        assert max(gated_engine.batch_sizes) <= 2

    async def _assert_not_batched(
        self,
        engine: GatedMockEngine,
        requests: list[object],
        config: SchedulerConfig = IMMEDIATE,
    ) -> None:
        scheduler = InferenceScheduler(engine, config)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(scheduler.submit_final("block", make_request()))
            await engine.wait_until_entered()

            tasks = [
                asyncio.create_task(scheduler.submit_final(f"s{index}", request))  # type: ignore[arg-type]
                for index, request in enumerate(requests)
            ]
            await wait_until(
                lambda: scheduler.queued_count == len(requests),
                description="every job is queued",
            )
            engine.open_gate()
            await asyncio.gather(blocker, *tasks)
        finally:
            engine.open_gate()
            await scheduler.close()

        assert all(call.batch_size == 1 for call in engine.calls)


class TestEngineFailures:
    async def test_an_engine_error_reaches_the_caller(
        self, scheduler_factory: SchedulerFactory
    ) -> None:
        engine = FailingMockEngine(lambda: RuntimeError("engine exploded"))
        await engine.startup()
        scheduler = await scheduler_factory(engine, IMMEDIATE)
        with pytest.raises(RuntimeError, match="engine exploded"):
            await scheduler.submit_final("s1", make_request())
        assert engine.call_count == 1

    async def test_the_scheduler_recovers_after_an_engine_error(
        self, scheduler_factory: SchedulerFactory
    ) -> None:
        engine = FailingMockEngine()
        await engine.startup()
        scheduler = await scheduler_factory(engine, IMMEDIATE)
        for index in range(3):
            with pytest.raises(RuntimeError):
                await scheduler.submit_final(f"s{index}", make_request())
        assert scheduler.running is True
        assert scheduler.queued_count == 0

    async def test_an_async_engine_is_rejected_as_a_contract_error(
        self, scheduler_factory: SchedulerFactory
    ) -> None:
        engine = AwaitableMockEngine()
        await engine.startup()
        scheduler = await scheduler_factory(engine, IMMEDIATE)
        with pytest.raises(SchedulerError, match="returned an awaitable"):
            await scheduler.submit_final("s1", make_request())

    async def test_a_short_batch_fails_every_job_in_that_batch(
        self, gated_engine: GatedMockEngine
    ) -> None:
        engine = ShortBatchMockEngine()
        await engine.startup()
        scheduler = InferenceScheduler(engine, IMMEDIATE)
        await scheduler.start()
        try:
            first = asyncio.create_task(scheduler.submit_final("s1", make_request()))
            second = asyncio.create_task(scheduler.submit_final("s2", make_request()))
            results = await asyncio.gather(first, second, return_exceptions=True)
        finally:
            await scheduler.close()

        assert any(isinstance(result, SchedulerError) for result in results)
        for result in results:
            assert not hasattr(result, "text")

    async def test_an_unready_engine_reports_its_own_failure(
        self, scheduler_factory: SchedulerFactory
    ) -> None:
        engine = BatchingMockEngine()  # deliberately never started
        scheduler = await scheduler_factory(engine, IMMEDIATE)
        with pytest.raises(RuntimeError, match="not ready"):
            await scheduler.submit_final("s1", make_request())


class TestCancellation:
    async def test_a_cancelled_request_releases_its_queue_slot(
        self, gated_engine: GatedMockEngine
    ) -> None:
        scheduler = InferenceScheduler(gated_engine, IMMEDIATE)
        await scheduler.start()
        try:
            blocker = asyncio.create_task(scheduler.submit_final("block", make_request()))
            await gated_engine.wait_until_entered()

            abandoned = asyncio.create_task(
                scheduler.submit_final("abandoned", make_request(language="bn"))
            )
            await wait_until(lambda: scheduler.queued_count == 1, description="the job is queued")
            abandoned.cancel()
            await asyncio.gather(abandoned, return_exceptions=True)

            survivor = asyncio.create_task(
                scheduler.submit_final("survivor", make_request(language="ta"))
            )
            await wait_until(
                lambda: scheduler.queued_count >= 1, description="the survivor is queued"
            )
            gated_engine.open_gate()
            await asyncio.gather(blocker, survivor)
        finally:
            gated_engine.open_gate()
            await scheduler.close()

        languages = [call.language for call in gated_engine.calls]
        assert "bn" not in languages
        assert "ta" in languages
        assert scheduler.queued_count == 0

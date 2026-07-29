"""Contract tests for readiness state and its ready/not-ready transitions.

A load balancer only sees ``ready``, so the exact set of statuses that count as
ready is the whole contract of this module.
"""

from __future__ import annotations

import dataclasses
import threading

import pytest

from app.core.readiness import CheckStatus, ReadinessSnapshot, ReadinessTracker

NOT_READY_STATUSES = [
    CheckStatus.PENDING,
    CheckStatus.STARTING,
    CheckStatus.FAILED,
    CheckStatus.STOPPED,
]


class TestInitialState:
    def test_a_new_tracker_is_not_ready(self) -> None:
        snapshot = ReadinessTracker().snapshot()
        assert snapshot.stage == "created"
        assert snapshot.checks == {
            "engine": CheckStatus.PENDING,
            "scheduler": CheckStatus.PENDING,
        }
        assert snapshot.detail is None
        assert snapshot.ready is False

    def test_check_status_set_is_closed(self) -> None:
        assert [status.value for status in CheckStatus] == [
            "pending",
            "starting",
            "ready",
            "disabled",
            "failed",
            "stopped",
        ]

    def test_trackers_do_not_share_their_check_dictionaries(self) -> None:
        first = ReadinessTracker()
        second = ReadinessTracker()
        first.update(engine=CheckStatus.READY)
        assert second.snapshot().checks["engine"] == CheckStatus.PENDING


class TestReadyTransitions:
    def test_startup_sequence_becomes_ready_only_at_the_end(self) -> None:
        tracker = ReadinessTracker()
        assert tracker.snapshot().ready is False

        tracker.update(stage="starting_engine", engine=CheckStatus.STARTING)
        assert tracker.snapshot().ready is False

        tracker.update(stage="starting_scheduler", engine=CheckStatus.READY)
        assert tracker.snapshot().ready is False

        tracker.update(stage="serving", scheduler=CheckStatus.READY)
        snapshot = tracker.snapshot()
        assert snapshot.ready is True
        assert snapshot.stage == "serving"

    def test_a_disabled_component_still_counts_as_ready(self) -> None:
        tracker = ReadinessTracker()
        tracker.update(engine=CheckStatus.READY, scheduler=CheckStatus.DISABLED)
        assert tracker.snapshot().ready is True

    @pytest.mark.parametrize("status", NOT_READY_STATUSES)
    def test_no_other_status_counts_as_ready(self, status: CheckStatus) -> None:
        tracker = ReadinessTracker()
        tracker.update(engine=CheckStatus.READY, scheduler=status)
        assert tracker.snapshot().ready is False

    def test_readiness_is_lost_again_on_failure(self) -> None:
        tracker = ReadinessTracker()
        tracker.update(engine=CheckStatus.READY, scheduler=CheckStatus.READY)
        assert tracker.snapshot().ready is True

        tracker.update(stage="degraded", engine=CheckStatus.FAILED, detail="RuntimeError")
        snapshot = tracker.snapshot()
        assert snapshot.ready is False
        assert snapshot.stage == "degraded"
        assert snapshot.detail == "RuntimeError"

    def test_shutdown_reports_stopped_and_not_ready(self) -> None:
        tracker = ReadinessTracker()
        tracker.update(engine=CheckStatus.READY, scheduler=CheckStatus.READY)
        tracker.update(stage="stopped", engine=CheckStatus.STOPPED, scheduler=CheckStatus.STOPPED)
        assert tracker.snapshot().ready is False


class TestUpdateSemantics:
    def test_omitted_fields_keep_their_previous_value(self) -> None:
        tracker = ReadinessTracker()
        tracker.update(stage="serving", engine=CheckStatus.READY)
        tracker.update(scheduler=CheckStatus.READY)
        snapshot = tracker.snapshot()
        assert snapshot.stage == "serving"
        assert snapshot.checks["engine"] == CheckStatus.READY

    def test_detail_is_cleared_by_any_later_update(self) -> None:
        """Details describe the current stage only, never an earlier failure."""

        tracker = ReadinessTracker()
        tracker.update(stage="loading", detail="downloading assets")
        assert tracker.snapshot().detail == "downloading assets"

        tracker.update(stage="serving", engine=CheckStatus.READY)
        assert tracker.snapshot().detail is None

    def test_only_the_declared_checks_exist(self) -> None:
        tracker = ReadinessTracker()
        tracker.update(engine=CheckStatus.READY, scheduler=CheckStatus.READY)
        assert sorted(tracker.snapshot().checks) == ["engine", "scheduler"]


class TestSnapshotIsolation:
    def test_a_snapshot_is_a_copy_of_the_live_checks(self) -> None:
        tracker = ReadinessTracker()
        snapshot = tracker.snapshot()
        snapshot.checks["engine"] = CheckStatus.READY
        assert tracker.snapshot().checks["engine"] == CheckStatus.PENDING

    def test_later_updates_do_not_mutate_an_earlier_snapshot(self) -> None:
        tracker = ReadinessTracker()
        before = tracker.snapshot()
        tracker.update(engine=CheckStatus.READY, scheduler=CheckStatus.READY)
        assert before.ready is False
        assert tracker.snapshot().ready is True

    def test_a_snapshot_is_immutable(self) -> None:
        snapshot = ReadinessTracker().snapshot()
        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot.stage = "tampered"  # type: ignore[misc]

    def test_snapshots_are_plain_value_objects(self) -> None:
        snapshot = ReadinessSnapshot("serving", {"engine": CheckStatus.READY}, None)
        assert snapshot.ready is True
        assert snapshot.stage == "serving"


class TestConcurrentUpdates:
    def test_updates_from_many_threads_all_land(self) -> None:
        tracker = ReadinessTracker()
        start = threading.Barrier(8)

        def worker(index: int) -> None:
            start.wait(timeout=5)
            for _ in range(50):
                tracker.update(
                    stage=f"stage-{index}",
                    engine=CheckStatus.READY,
                    scheduler=CheckStatus.READY,
                )

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()

        snapshot = tracker.snapshot()
        assert snapshot.ready is True
        assert snapshot.stage.startswith("stage-")

    def test_a_snapshot_taken_during_updates_is_internally_consistent(self) -> None:
        tracker = ReadinessTracker()
        stop = threading.Event()

        def flip() -> None:
            while not stop.is_set():
                tracker.update(engine=CheckStatus.READY, scheduler=CheckStatus.READY)
                tracker.update(engine=CheckStatus.PENDING, scheduler=CheckStatus.PENDING)

        writer = threading.Thread(target=flip)
        writer.start()
        try:
            for _ in range(200):
                checks = tracker.snapshot().checks
                assert sorted(checks) == ["engine", "scheduler"]
                assert set(checks.values()) <= set(CheckStatus)
        finally:
            stop.set()
            writer.join(timeout=10)
            assert not writer.is_alive()

"""Health endpoint contracts across the full startup and shutdown sequence.

These tests drive the real application over HTTP with the deterministic
MockEngine. ``TestClient`` only runs the lifespan inside a ``with`` block, which
is exactly what makes the pre-startup, serving, and post-shutdown readiness
states observable from a client's point of view.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.readiness import CheckStatus
from app.schemas.rest import LiveResponse, ReadyResponse
from tests.support.asgi import SchedulerDouble, mock_engine_app, scheduler_app
from tests.support.engines import NeverReadyMockEngine, RecordingMockEngine


class TestLiveness:
    def test_liveness_is_served_before_startup(self) -> None:
        client = TestClient(mock_engine_app())

        response = client.get("/health/live")

        assert response.status_code == 200
        assert response.json() == {"status": "live"}
        assert LiveResponse.model_validate(response.json()).status == "live"

    def test_liveness_is_served_while_ready(self) -> None:
        with TestClient(mock_engine_app()) as client:
            assert client.get("/health/live").json() == {"status": "live"}

    def test_liveness_never_runs_inference(self) -> None:
        engine = RecordingMockEngine()
        with TestClient(mock_engine_app(engine)) as client:
            for _ in range(3):
                assert client.get("/health/live").status_code == 200

        assert tuple(engine.calls) == ()

    def test_liveness_is_still_served_after_shutdown(self) -> None:
        app = mock_engine_app()
        with TestClient(app):
            pass

        assert TestClient(app).get("/health/live").status_code == 200


class TestReadinessTransitions:
    def test_readiness_is_refused_before_startup(self) -> None:
        client = TestClient(mock_engine_app())

        response = client.get("/health/ready")

        assert response.status_code == 503
        payload = ReadyResponse.model_validate(response.json())
        assert payload.status == "not_ready"
        assert payload.stage == "created"
        assert payload.checks == {
            "engine": CheckStatus.PENDING,
            "scheduler": CheckStatus.PENDING,
        }

    def test_readiness_is_granted_once_startup_completed(self) -> None:
        with TestClient(mock_engine_app()) as client:
            response = client.get("/health/ready")

        assert response.status_code == 200
        payload = ReadyResponse.model_validate(response.json())
        assert payload.status == "ready"
        assert payload.stage == "ready"
        assert payload.checks == {
            "engine": CheckStatus.READY,
            "scheduler": CheckStatus.READY,
        }
        assert payload.detail is None

    def test_readiness_is_refused_after_shutdown(self) -> None:
        app = mock_engine_app()
        with TestClient(app) as client:
            assert client.get("/health/ready").status_code == 200

        response = TestClient(app).get("/health/ready")

        assert response.status_code == 503
        payload = ReadyResponse.model_validate(response.json())
        assert payload.status == "not_ready"
        assert payload.stage == "stopped"
        assert payload.checks == {
            "engine": CheckStatus.STOPPED,
            "scheduler": CheckStatus.STOPPED,
        }

    def test_the_whole_transition_sequence_is_monotonic(self) -> None:
        app = mock_engine_app()
        observed: list[tuple[int, str]] = []

        observed.append(self._probe(TestClient(app)))
        with TestClient(app) as running:
            observed.append(self._probe(running))
        observed.append(self._probe(TestClient(app)))

        assert observed == [
            (503, "created"),
            (200, "ready"),
            (503, "stopped"),
        ]

    @staticmethod
    def _probe(client: TestClient) -> tuple[int, str]:
        response = client.get("/health/ready")
        return response.status_code, str(response.json()["stage"])

    def test_readiness_is_refused_while_scheduler_close_is_blocked(self) -> None:
        threading = __import__("threading")

        class BlockingCloseScheduler(SchedulerDouble):
            def __init__(self) -> None:
                super().__init__()
                self.close_entered = threading.Event()

            async def close(self) -> None:
                asyncio = __import__("asyncio")
                self.closed += 1
                self.loop = asyncio.get_running_loop()
                self.release_close = asyncio.Event()
                self.close_entered.set()
                await self.release_close.wait()

            def release(self) -> None:
                if self.close_entered.is_set():
                    self.loop.call_soon_threadsafe(self.release_close.set)

        scheduler = BlockingCloseScheduler()
        app = scheduler_app(scheduler)
        shutdown_errors: list[BaseException] = []

        def run_lifespan() -> None:
            try:
                with TestClient(app):
                    pass
            except BaseException as exc:
                shutdown_errors.append(exc)

        shutdown = threading.Thread(target=run_lifespan)
        shutdown.start()
        try:
            assert scheduler.close_entered.wait(timeout=5)
            response = TestClient(app).get("/health/ready")
            assert response.status_code == 503
            assert response.json() == {
                "status": "not_ready",
                "stage": "stopping",
                "checks": {
                    "engine": CheckStatus.STOPPING,
                    "scheduler": CheckStatus.STOPPING,
                },
                "detail": None,
            }
        finally:
            scheduler.release()
            shutdown.join(timeout=5)

        assert not shutdown.is_alive()
        assert shutdown_errors == []


class TestReadinessReflectsDependencies:
    def test_an_engine_that_never_reports_ready_keeps_the_service_unready(self) -> None:
        with TestClient(mock_engine_app(NeverReadyMockEngine())) as client:
            response = client.get("/health/ready")

        assert response.status_code == 503
        payload = ReadyResponse.model_validate(response.json())
        assert payload.status == "not_ready"
        assert payload.checks["engine"] == "starting"

    def test_a_stopped_scheduler_keeps_the_service_unready(self) -> None:
        scheduler = SchedulerDouble()
        with TestClient(scheduler_app(scheduler)) as client:
            assert client.get("/health/ready").status_code == 200

            scheduler.running = False
            response = client.get("/health/ready")

        assert response.status_code == 503
        payload = ReadyResponse.model_validate(response.json())
        assert payload.checks["scheduler"] == CheckStatus.STOPPED
        assert payload.checks["engine"] == CheckStatus.READY

    def test_readiness_never_runs_inference(self) -> None:
        engine = RecordingMockEngine()
        with TestClient(mock_engine_app(engine)) as client:
            assert client.get("/health/ready").status_code == 200

        assert tuple(engine.calls) == ()

    def test_the_lifespan_starts_and_closes_the_scheduler_exactly_once(self) -> None:
        scheduler = SchedulerDouble()
        with TestClient(scheduler_app(scheduler)):
            assert (scheduler.started, scheduler.closed) == (1, 0)

        assert (scheduler.started, scheduler.closed) == (1, 1)

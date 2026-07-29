"""Small, side-effect-free readiness state shared by lifespan and health routes."""

from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock


class CheckStatus(StrEnum):
    PENDING = "pending"
    STARTING = "starting"
    READY = "ready"
    DISABLED = "disabled"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    stage: str
    checks: dict[str, str]
    detail: str | None

    @property
    def ready(self) -> bool:
        return all(
            value in {CheckStatus.READY, CheckStatus.DISABLED} for value in self.checks.values()
        )


@dataclass(slots=True)
class ReadinessTracker:
    """Thread-safe startup progress; details must never contain request data."""

    stage: str = "created"
    checks: dict[str, str] = field(
        default_factory=lambda: {
            "engine": CheckStatus.PENDING,
            "scheduler": CheckStatus.PENDING,
        }
    )
    detail: str | None = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def update(
        self,
        *,
        stage: str | None = None,
        engine: CheckStatus | None = None,
        scheduler: CheckStatus | None = None,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            if stage is not None:
                self.stage = stage
            if engine is not None:
                self.checks["engine"] = engine
            if scheduler is not None:
                self.checks["scheduler"] = scheduler
            self.detail = detail

    def snapshot(self) -> ReadinessSnapshot:
        with self._lock:
            return ReadinessSnapshot(self.stage, dict(self.checks), self.detail)

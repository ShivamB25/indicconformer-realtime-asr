"""Engine contracts shared by every inference backend."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.types import SUPPORTED_LANGUAGE_CODES

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio: np.ndarray
    sample_rate: int
    language: str
    decoder: str

    def __post_init__(self) -> None:
        if not isinstance(self.audio, np.ndarray):
            raise ValueError("audio must be a numpy array")
        if self.sample_rate != 16_000:
            raise ValueError("sample_rate must be 16000 Hz")
        if self.audio.ndim != 1:
            raise ValueError("audio must be a mono one-dimensional array")
        if self.language not in SUPPORTED_LANGUAGE_CODES:
            raise ValueError("language is not supported")
        if self.decoder not in {"ctc", "rnnt"}:
            raise ValueError("decoder must be ctc or rnnt")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str
    decoder: str
    audio_duration_ms: int
    inference_ms: float


class EngineState(StrEnum):
    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class EngineReadiness:
    state: EngineState
    stage: str
    detail: str | None = None

    @property
    def ready(self) -> bool:
        return self.state is EngineState.READY


@runtime_checkable
class Engine(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def readiness(self) -> EngineReadiness: ...

    async def startup(self, progress: ProgressCallback | None = None) -> None: ...

    async def shutdown(self) -> None: ...

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...


class BaseEngine(ABC):
    """Convenience base with observable startup state for concrete engines."""

    def __init__(self) -> None:
        self._readiness = EngineReadiness(EngineState.NEW, "created")

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def readiness(self) -> EngineReadiness:
        return self._readiness

    def _set_readiness(
        self,
        state: EngineState,
        stage: str,
        detail: str | None = None,
    ) -> None:
        self._readiness = EngineReadiness(state, stage, detail)

    @abstractmethod
    async def startup(self, progress: ProgressCallback | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise NotImplementedError

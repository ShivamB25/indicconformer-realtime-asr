"""Deterministic, dependency-light engine used by CPU tests."""

from app.engine.base import (
    BaseEngine,
    EngineState,
    ProgressCallback,
    TranscriptionRequest,
    TranscriptionResult,
)


class MockEngine(BaseEngine):
    """A zero-I/O engine whose output depends only on request metadata."""

    @property
    def name(self) -> str:
        return "mock"

    async def startup(self, progress: ProgressCallback | None = None) -> None:
        self._set_readiness(EngineState.STARTING, "starting")
        if progress is not None:
            progress("mock_starting")
        self._set_readiness(EngineState.READY, "ready")
        if progress is not None:
            progress("mock_ready")

    async def shutdown(self) -> None:
        self._set_readiness(EngineState.STOPPED, "stopped")

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        if not self.readiness.ready:
            raise RuntimeError("mock engine is not ready")
        duration_ms = round(request.audio.size * 1000 / request.sample_rate)
        text = (
            f"mock transcript language={request.language} "
            f"decoder={request.decoder} duration_ms={duration_ms}"
        )
        return TranscriptionResult(
            text=text,
            language=request.language,
            decoder=request.decoder,
            audio_duration_ms=duration_ms,
            inference_ms=0.0,
        )

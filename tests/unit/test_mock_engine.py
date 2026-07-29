"""Contract tests for the engine interface and its deterministic mock.

MockEngine is the only engine this suite ever runs, so its startup states, its
refusal to work before it is ready, and the exact shape of its output are part
of the test contract itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.types import SUPPORTED_LANGUAGE_CODES
from app.engine.base import (
    BaseEngine,
    Engine,
    EngineReadiness,
    EngineState,
    TranscriptionRequest,
    TranscriptionResult,
)
from app.engine.mock import MockEngine
from tests.support.audio import float_audio
from tests.support.engines import ProgressRecordingMockEngine, make_request


class TestRequestValidation:
    def test_a_valid_request_is_accepted(self) -> None:
        request = make_request(duration_ms=100, language="hi", decoder="ctc")
        assert request.sample_rate == 16_000
        assert request.audio.size == 1_600

    @pytest.mark.parametrize("sample_rate", [0, 8_000, 44_100, 16_001, -16_000])
    def test_only_sixteen_kilohertz_audio_is_accepted(self, sample_rate: int) -> None:
        with pytest.raises(ValueError, match="sample_rate must be 16000 Hz"):
            TranscriptionRequest(
                audio=float_audio(100), sample_rate=sample_rate, language="hi", decoder="ctc"
            )

    def test_multichannel_audio_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mono one-dimensional"):
            TranscriptionRequest(
                audio=np.zeros((2, 100), dtype=np.float32),
                sample_rate=16_000,
                language="hi",
                decoder="ctc",
            )

    @pytest.mark.parametrize("language", sorted(SUPPORTED_LANGUAGE_CODES))
    def test_every_supported_language_is_accepted(self, language: str) -> None:
        assert make_request(language=language).language == language

    @pytest.mark.parametrize("language", ["", "en", "HI", "hi-IN", "xx", " hi", "hin"])
    def test_languages_outside_the_closed_set_are_refused(self, language: str) -> None:
        with pytest.raises(ValueError, match="language is not supported"):
            TranscriptionRequest(
                audio=float_audio(100), sample_rate=16_000, language=language, decoder="ctc"
            )

    def test_non_array_audio_is_refused(self) -> None:
        with pytest.raises(ValueError, match="audio must be a numpy array"):
            TranscriptionRequest(
                audio=[0.0, 0.1],  # type: ignore[arg-type]
                sample_rate=16_000,
                language="hi",
                decoder="ctc",
            )

    @pytest.mark.parametrize("decoder", ["", "beam", "CTC", "rnn-t", "greedy"])
    def test_only_the_two_supported_decoders_are_accepted(self, decoder: str) -> None:
        with pytest.raises(ValueError, match="decoder must be ctc or rnnt"):
            TranscriptionRequest(
                audio=float_audio(100), sample_rate=16_000, language="hi", decoder=decoder
            )

    def test_a_request_is_immutable(self) -> None:
        request = make_request()
        with pytest.raises(AttributeError):
            request.language = "ta"  # type: ignore[misc]

    def test_empty_audio_is_representable(self) -> None:
        request = TranscriptionRequest(
            audio=float_audio(0), sample_rate=16_000, language="hi", decoder="ctc"
        )
        assert request.audio.size == 0


class TestReadinessLifecycle:
    def test_a_new_engine_is_not_ready(self) -> None:
        engine = MockEngine()
        assert engine.readiness.state is EngineState.NEW
        assert engine.readiness.stage == "created"
        assert engine.readiness.ready is False

    async def test_startup_reaches_the_ready_state(self) -> None:
        engine = MockEngine()
        await engine.startup()
        assert engine.readiness.state is EngineState.READY
        assert engine.readiness.stage == "ready"
        assert engine.readiness.ready is True

    async def test_shutdown_leaves_the_engine_unusable(self) -> None:
        engine = MockEngine()
        await engine.startup()
        await engine.shutdown()
        assert engine.readiness.state is EngineState.STOPPED
        assert engine.readiness.ready is False
        with pytest.raises(RuntimeError, match="not ready"):
            engine.transcribe(make_request())

    def test_transcription_before_startup_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="not ready"):
            MockEngine().transcribe(make_request())

    async def test_startup_reports_staged_progress(self) -> None:
        labels: list[str] = []
        engine = MockEngine()
        await engine.startup(labels.append)
        assert labels == ["mock_starting", "mock_ready"]

    async def test_progress_callbacks_are_optional(self) -> None:
        engine = ProgressRecordingMockEngine()
        await engine.startup()
        assert engine.progress_labels == ["mock_starting", "mock_ready"]

    async def test_startup_is_repeatable(self) -> None:
        engine = MockEngine()
        await engine.startup()
        await engine.startup()
        assert engine.readiness.ready is True

    def test_readiness_is_an_immutable_value(self) -> None:
        readiness = EngineReadiness(EngineState.READY, "ready")
        assert readiness.ready is True
        with pytest.raises(AttributeError):
            readiness.stage = "tampered"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "state",
        [EngineState.NEW, EngineState.STARTING, EngineState.FAILED, EngineState.STOPPED],
    )
    def test_only_the_ready_state_reports_ready(self, state: EngineState) -> None:
        assert EngineReadiness(state, "stage").ready is False

    def test_the_engine_state_set_is_closed(self) -> None:
        assert [state.value for state in EngineState] == [
            "new",
            "starting",
            "ready",
            "failed",
            "stopped",
        ]


class TestMockTranscription:
    async def test_output_is_derived_only_from_request_metadata(
        self, mock_engine: MockEngine
    ) -> None:
        result = mock_engine.transcribe(
            make_request(duration_ms=1_500, language="mni", decoder="rnnt")
        )
        assert isinstance(result, TranscriptionResult)
        assert result.text == "mock transcript language=mni decoder=rnnt duration_ms=1500"
        assert result.language == "mni"
        assert result.decoder == "rnnt"
        assert result.audio_duration_ms == 1_500
        assert result.inference_ms == 0.0

    async def test_the_same_request_always_produces_the_same_text(
        self, mock_engine: MockEngine
    ) -> None:
        request = make_request(duration_ms=320, language="ta", decoder="ctc")
        assert mock_engine.transcribe(request) == mock_engine.transcribe(request)

    async def test_duration_is_computed_from_the_sample_count(
        self, mock_engine: MockEngine
    ) -> None:
        for duration_ms in (0, 20, 100, 999, 60_000):
            result = mock_engine.transcribe(make_request(duration_ms=duration_ms))
            assert result.audio_duration_ms == duration_ms

    async def test_audio_content_never_changes_the_output(self, mock_engine: MockEngine) -> None:
        quiet = mock_engine.transcribe(make_request(duration_ms=200, level=0.0))
        loud = mock_engine.transcribe(make_request(duration_ms=200, level=0.9))
        assert quiet.text == loud.text

    async def test_the_engine_is_named_for_logs_and_metrics(self, mock_engine: MockEngine) -> None:
        assert mock_engine.name == "mock"

    async def test_the_mock_satisfies_the_engine_protocol(self, mock_engine: MockEngine) -> None:
        assert isinstance(mock_engine, Engine)
        assert isinstance(mock_engine, BaseEngine)

    async def test_results_are_immutable(self, mock_engine: MockEngine) -> None:
        result = mock_engine.transcribe(make_request())
        with pytest.raises(AttributeError):
            result.text = "tampered"  # type: ignore[misc]

"""The streaming audio path, composed end to end without a WebSocket.

Voice activity detection, endpointing, the PCM buffer, the partial cadence, the
stable prefix, and a real ``InferenceScheduler`` over ``MockEngine`` are wired
together here and driven one 20 ms frame at a time. Because each submission is
awaited, the resulting event sequence is exact rather than merely plausible.

Transport-level behaviour (JSON events, close codes) belongs to
``tests/websocket``; this module owns the audio-to-transcript contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.audio.endpoint import (
    AdaptivePartialCadence,
    EndpointConfig,
    EndpointDetector,
    EndpointEvent,
    PartialCadenceConfig,
)
from app.audio.pcm import SAMPLE_RATE, PCM16Buffer, decode_pcm16_frame
from app.audio.stable_prefix import RollingStablePrefix
from app.audio.vad import EnergyVAD, EnergyVADConfig
from app.engine.base import Engine, TranscriptionRequest
from app.engine.scheduler import InferenceScheduler
from tests.support.audio import silence_frame, speech_frame, speech_frames
from tests.support.engines import ScriptedTextEngine

PARTIAL_CADENCE = PartialCadenceConfig(initial_ms=400, minimum_ms=200, maximum_ms=1_200)


@dataclass(frozen=True, slots=True)
class Emitted:
    """One observable pipeline event, reduced to what a client would see."""

    kind: str
    text: str = ""
    duration_ms: int = 0
    revision: int = 0
    is_stable: bool = False
    decoder: str = ""


@dataclass
class Pipeline:
    """Sequence the audio components exactly as a realtime session does."""

    scheduler: InferenceScheduler
    language: str = "hi"
    vad_enabled: bool = True
    partial_decoder: str = "ctc"
    final_decoder: str = "rnnt"
    endpoint_config: EndpointConfig = field(default_factory=EndpointConfig)
    cadence_config: PartialCadenceConfig = field(default=PARTIAL_CADENCE)
    session_id: str = "session"
    revision: int = 0
    last_partial: str = ""

    def __post_init__(self) -> None:
        self.buffer = PCM16Buffer(self.endpoint_config.max_utterance_ms)
        self.endpoint = EndpointDetector(self.endpoint_config)
        self.vad = EnergyVAD(EnergyVADConfig())
        self.cadence = AdaptivePartialCadence(self.cadence_config)
        self.stable_prefix = RollingStablePrefix(3)

    async def push(self, frame: bytes) -> list[Emitted]:
        """Feed one 20 ms frame and return everything it caused."""

        emitted: list[Emitted] = []
        if self.vad_enabled:
            was_active = self.endpoint.active
            is_speech = self.vad.is_speech(decode_pcm16_frame(frame))
            event = self.endpoint.process(is_speech)
            keep_frame = was_active or is_speech
        else:
            event, keep_frame = EndpointEvent.NONE, True
        if keep_frame:
            self.buffer.append(frame)
        elif not self.endpoint.active:
            self.buffer.clear()

        if event is EndpointEvent.SPEECH_STARTED:
            emitted.append(Emitted(kind="speech.started"))
        if event in (EndpointEvent.UTTERANCE_ENDED, EndpointEvent.UTTERANCE_LIMIT):
            emitted.append(await self.finalize())
            return emitted

        active = self.endpoint.active if self.vad_enabled else not self.buffer.empty
        if active and self.cadence.due(self.buffer.duration_ms):
            emitted.append(await self.emit_partial())
        return emitted

    async def push_all(self, frames: list[bytes]) -> list[Emitted]:
        emitted: list[Emitted] = []
        for frame in frames:
            emitted.extend(await self.push(frame))
        return emitted

    async def emit_partial(self) -> Emitted:
        audio_ms = self.buffer.duration_ms
        self.cadence.mark_submitted(audio_ms)
        result = await self.scheduler.submit_partial(
            self.session_id,
            TranscriptionRequest(
                audio=self.buffer.to_float32(),
                sample_rate=SAMPLE_RATE,
                language=self.language,
                decoder=self.partial_decoder,
            ),
        )
        self.cadence.observe(result.text != self.last_partial, audio_ms)
        self.last_partial = result.text
        self.revision += 1
        stable = self.stable_prefix.add(result.text)
        return Emitted(
            kind="transcript.partial",
            text=result.text,
            duration_ms=result.audio_duration_ms,
            revision=self.revision,
            is_stable=bool(result.text) and stable == result.text,
            decoder=result.decoder,
        )

    async def finalize(self) -> Emitted:
        request = TranscriptionRequest(
            audio=self.buffer.to_float32(),
            sample_rate=SAMPLE_RATE,
            language=self.language,
            decoder=self.final_decoder,
        )
        self.buffer.clear()
        self.endpoint.reset()
        result = await self.scheduler.submit_final(self.session_id, request)
        self.cadence.reset()
        self.stable_prefix.reset()
        self.last_partial = ""
        return Emitted(
            kind="transcript.final",
            text=result.text,
            duration_ms=result.audio_duration_ms,
            decoder=result.decoder,
        )

    async def commit(self) -> Emitted:
        self.endpoint.commit()
        return await self.finalize()


@pytest.fixture
async def pipeline(mock_engine: Engine, scheduler_factory: object) -> Pipeline:
    scheduler = await scheduler_factory(mock_engine)  # type: ignore[operator]
    return Pipeline(scheduler)


def kinds(emitted: list[Emitted]) -> list[str]:
    return [event.kind for event in emitted]


class TestSpeechOnset:
    async def test_silence_alone_produces_nothing(self, pipeline: Pipeline) -> None:
        emitted = await pipeline.push_all([silence_frame()] * 100)
        assert emitted == []
        assert pipeline.buffer.empty
        assert pipeline.endpoint.active is False

    async def test_speech_started_fires_after_sixty_milliseconds_of_speech(
        self, pipeline: Pipeline
    ) -> None:
        assert await pipeline.push(speech_frame()) == []
        assert await pipeline.push(speech_frame()) == []
        assert kinds(await pipeline.push(speech_frame())) == ["speech.started"]
        assert pipeline.buffer.duration_ms == 60

    async def test_speech_started_fires_once_per_utterance(self, pipeline: Pipeline) -> None:
        emitted = await pipeline.push_all(speech_frames(400))
        assert kinds(emitted).count("speech.started") == 1

    async def test_leading_silence_is_discarded(self, pipeline: Pipeline) -> None:
        await pipeline.push_all([silence_frame()] * 25)
        assert pipeline.buffer.empty
        await pipeline.push_all(speech_frames(200))
        final = await pipeline.commit()
        assert final.duration_ms == 200

    async def test_a_single_speech_frame_is_not_an_utterance(self, pipeline: Pipeline) -> None:
        await pipeline.push(speech_frame())
        await pipeline.push(silence_frame())
        assert pipeline.buffer.empty
        assert pipeline.endpoint.active is False


class TestAutomaticEndpoint:
    async def test_trailing_silence_closes_the_utterance(self, pipeline: Pipeline) -> None:
        emitted = await pipeline.push_all(speech_frames(200) + [silence_frame()] * 30)
        finals = [event for event in emitted if event.kind == "transcript.final"]
        assert len(finals) == 1
        assert finals[0].duration_ms == 800
        assert finals[0].decoder == "rnnt"
        assert finals[0].text == ("mock transcript language=hi decoder=rnnt duration_ms=800")

    async def test_the_endpoint_waits_for_the_full_silence_window(self, pipeline: Pipeline) -> None:
        emitted = await pipeline.push_all(speech_frames(200) + [silence_frame()] * 29)
        assert "transcript.final" not in kinds(emitted)
        assert pipeline.endpoint.active is True
        assert kinds(await pipeline.push(silence_frame())) == ["transcript.final"]

    async def test_a_short_utterance_still_needs_the_minimum_duration(
        self, pipeline: Pipeline
    ) -> None:
        emitted = await pipeline.push_all(speech_frames(60) + [silence_frame()] * 30)
        finals = [event for event in emitted if event.kind == "transcript.final"]
        assert len(finals) == 1
        assert finals[0].duration_ms == 660

    async def test_the_detector_is_reusable_for_the_next_utterance(
        self, pipeline: Pipeline
    ) -> None:
        first = await pipeline.push_all(speech_frames(200) + [silence_frame()] * 30)
        second = await pipeline.push_all(speech_frames(400) + [silence_frame()] * 30)
        assert kinds(first).count("speech.started") == 1
        assert kinds(second).count("speech.started") == 1
        first_final = [event for event in first if event.kind == "transcript.final"][0]
        second_final = [event for event in second if event.kind == "transcript.final"][0]
        assert (first_final.duration_ms, second_final.duration_ms) == (800, 1_000)


class TestUtteranceLimit:
    async def test_a_long_utterance_is_cut_at_the_limit(
        self, mock_engine: Engine, scheduler_factory: object
    ) -> None:
        scheduler = await scheduler_factory(mock_engine)  # type: ignore[operator]
        pipeline = Pipeline(
            scheduler,
            endpoint_config=EndpointConfig(max_utterance_ms=200),
        )
        emitted = await pipeline.push_all(speech_frames(600))
        finals = [event for event in emitted if event.kind == "transcript.final"]
        assert [event.duration_ms for event in finals] == [200, 200, 200]
        assert kinds(emitted).count("speech.started") == 3

    async def test_the_buffer_never_outgrows_the_endpoint_limit(
        self, mock_engine: Engine, scheduler_factory: object
    ) -> None:
        scheduler = await scheduler_factory(mock_engine)  # type: ignore[operator]
        pipeline = Pipeline(scheduler, endpoint_config=EndpointConfig(max_utterance_ms=400))
        for _ in range(200):
            await pipeline.push(speech_frame())
            assert pipeline.buffer.duration_ms <= 400
            if pipeline.endpoint.active:
                assert pipeline.buffer.duration_ms == pipeline.endpoint.utterance_duration_ms

    async def test_without_vad_the_buffer_limit_is_the_only_bound(
        self, mock_engine: Engine, scheduler_factory: object
    ) -> None:
        scheduler = await scheduler_factory(mock_engine)  # type: ignore[operator]
        pipeline = Pipeline(
            scheduler,
            vad_enabled=False,
            endpoint_config=EndpointConfig(max_utterance_ms=200),
        )
        emitted = await pipeline.push_all([silence_frame()] * 10)
        assert pipeline.buffer.duration_ms == 200
        assert "transcript.final" not in kinds(emitted)
        final = await pipeline.commit()
        assert final.duration_ms == 200
        assert pipeline.buffer.empty


class TestExplicitCommit:
    async def test_commit_finalizes_audio_below_the_minimum_duration(
        self, pipeline: Pipeline
    ) -> None:
        await pipeline.push_all(speech_frames(80))
        final = await pipeline.commit()
        assert final.duration_ms == 80
        assert pipeline.endpoint.active is False
        assert pipeline.buffer.empty

    async def test_commit_uses_the_final_decoder_of_the_session(
        self, mock_engine: Engine, scheduler_factory: object
    ) -> None:
        scheduler = await scheduler_factory(mock_engine)  # type: ignore[operator]
        pipeline = Pipeline(scheduler, final_decoder="ctc", language="sat")
        await pipeline.push_all(speech_frames(100))
        final = await pipeline.commit()
        assert final.decoder == "ctc"
        assert "language=sat" in final.text

    async def test_commit_without_audio_produces_an_empty_final(self, pipeline: Pipeline) -> None:
        final = await pipeline.commit()
        assert final.duration_ms == 0
        assert final.text.endswith("duration_ms=0")


class TestPartials:
    async def test_partials_follow_the_audio_cadence(self, pipeline: Pipeline) -> None:
        emitted = await pipeline.push_all(speech_frames(1_200))
        partials = [event for event in emitted if event.kind == "transcript.partial"]
        assert [event.duration_ms for event in partials] == [400, 800, 1120]
        assert [event.revision for event in partials] == [1, 2, 3]

    async def test_a_changing_transcript_shortens_the_cadence(self, pipeline: Pipeline) -> None:
        assert pipeline.cadence.interval_ms == 400
        await pipeline.push_all(speech_frames(400))
        assert pipeline.cadence.interval_ms == 320

    async def test_partials_use_the_partial_decoder_not_the_final_one(
        self, pipeline: Pipeline
    ) -> None:
        emitted = await pipeline.push_all(speech_frames(400))
        partials = [event for event in emitted if event.kind == "transcript.partial"]
        assert [event.decoder for event in partials] == ["ctc"]

    async def test_revisions_never_repeat_across_utterances(self, pipeline: Pipeline) -> None:
        first = await pipeline.push_all(speech_frames(800) + [silence_frame()] * 30)
        second = await pipeline.push_all(speech_frames(800) + [silence_frame()] * 30)
        revisions = [
            event.revision for event in first + second if event.kind == "transcript.partial"
        ]
        assert revisions == sorted(revisions)
        assert len(set(revisions)) == len(revisions)

    async def test_no_partial_is_emitted_before_the_first_cadence_point(
        self, pipeline: Pipeline
    ) -> None:
        emitted = await pipeline.push_all(speech_frames(380))
        assert "transcript.partial" not in kinds(emitted)

    async def test_the_cadence_restarts_after_a_final(self, pipeline: Pipeline) -> None:
        await pipeline.push_all(speech_frames(800) + [silence_frame()] * 30)
        assert pipeline.cadence.interval_ms == 400
        emitted = await pipeline.push_all(speech_frames(400))
        partials = [event for event in emitted if event.kind == "transcript.partial"]
        assert [event.duration_ms for event in partials] == [400]


class TestPartialStability:
    async def test_a_changing_hypothesis_is_never_reported_as_stable(
        self, pipeline: Pipeline
    ) -> None:
        emitted = await pipeline.push_all(speech_frames(1_200))
        partials = [event for event in emitted if event.kind == "transcript.partial"]
        assert partials
        assert all(event.is_stable is False for event in partials)

    async def test_a_repeated_hypothesis_becomes_stable(self, scheduler_factory: object) -> None:
        engine = ScriptedTextEngine(["ek do", "ek do", "ek do tin"])
        await engine.startup()
        scheduler = await scheduler_factory(engine)  # type: ignore[operator]
        pipeline = Pipeline(scheduler)
        emitted = await pipeline.push_all(speech_frames(1_400))
        partials = [event for event in emitted if event.kind == "transcript.partial"]
        assert [event.text for event in partials] == ["ek do", "ek do", "ek do tin"]
        assert [event.duration_ms for event in partials] == [400, 800, 1_280]
        assert [event.is_stable for event in partials] == [False, True, False]
        await engine.shutdown()

    async def test_an_empty_hypothesis_is_never_stable(self, scheduler_factory: object) -> None:
        engine = ScriptedTextEngine([""])
        await engine.startup()
        scheduler = await scheduler_factory(engine)  # type: ignore[operator]
        pipeline = Pipeline(scheduler)
        emitted = await pipeline.push_all(speech_frames(1_200))
        partials = [event for event in emitted if event.kind == "transcript.partial"]
        assert partials
        assert all(event.is_stable is False for event in partials)

    async def test_stability_does_not_survive_a_final(self, scheduler_factory: object) -> None:
        engine = ScriptedTextEngine(["ek do"])
        await engine.startup()
        scheduler = await scheduler_factory(engine)  # type: ignore[operator]
        pipeline = Pipeline(scheduler)
        await pipeline.push_all(speech_frames(800))
        await pipeline.commit()
        emitted = await pipeline.push_all(speech_frames(400))
        partials = [event for event in emitted if event.kind == "transcript.partial"]
        assert [event.is_stable for event in partials] == [False]
        await engine.shutdown()


class TestLanguageCoverage:
    @pytest.mark.parametrize("language", ["as", "brx", "kok", "mni", "sat", "ur"])
    async def test_the_session_language_reaches_the_engine_unchanged(
        self, language: str, mock_engine: Engine, scheduler_factory: object
    ) -> None:
        scheduler = await scheduler_factory(mock_engine)  # type: ignore[operator]
        pipeline = Pipeline(scheduler, language=language)
        await pipeline.push_all(speech_frames(100))
        final = await pipeline.commit()
        assert f"language={language}" in final.text

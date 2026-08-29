"""Contract tests for utterance endpointing and adaptive partial cadence.

Endpointing decides when ``speech.started`` and ``transcript.final`` happen, so
every threshold is driven frame by frame in units of whole 20 ms frames. Nothing
here reads a clock: cadence is a function of accumulated audio time only.
"""

from __future__ import annotations

import pytest

from app.audio.endpoint import (
    AdaptivePartialCadence,
    EndpointConfig,
    EndpointDetector,
    EndpointEvent,
    EndpointState,
    PartialCadenceConfig,
)
from app.audio.pcm import FRAME_DURATION_MS


def feed(detector: EndpointDetector, is_speech: bool, frames: int) -> list[EndpointEvent]:
    return [detector.process(is_speech) for _ in range(frames)]


class TestEndpointConfiguration:
    def test_defaults_are_the_documented_thresholds(self) -> None:
        config = EndpointConfig()
        assert config.speech_start_ms == 60
        assert config.speech_end_ms == 600
        assert config.min_utterance_ms == 200
        assert config.max_utterance_ms == 30_000

    @pytest.mark.parametrize(
        "overrides",
        [
            {"speech_start_ms": 0},
            {"speech_start_ms": -20},
            {"speech_start_ms": 30},
            {"speech_end_ms": 610},
            {"min_utterance_ms": 55},
            {"max_utterance_ms": 0},
        ],
    )
    def test_durations_must_be_positive_whole_frames(self, overrides: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="positive multiples of 20 ms"):
            EndpointConfig(**overrides)

    def test_minimum_cannot_exceed_maximum(self) -> None:
        with pytest.raises(ValueError, match="min_utterance_ms must not exceed"):
            EndpointConfig(min_utterance_ms=1_000, max_utterance_ms=500)

    def test_equal_minimum_and_maximum_are_allowed(self) -> None:
        config = EndpointConfig(min_utterance_ms=200, max_utterance_ms=200)
        assert config.min_utterance_ms == config.max_utterance_ms


class TestSpeechOnset:
    def test_onset_requires_the_configured_run_of_speech_frames(self) -> None:
        detector = EndpointDetector()  # 60 ms onset == three frames
        assert detector.process(True) is EndpointEvent.NONE
        assert detector.process(True) is EndpointEvent.NONE
        assert detector.process(True) is EndpointEvent.SPEECH_STARTED
        assert detector.state is EndpointState.SPEECH
        assert detector.active is True

    def test_onset_is_announced_exactly_once_per_utterance(self) -> None:
        detector = EndpointDetector()
        feed(detector, True, 3)
        assert feed(detector, True, 5) == [EndpointEvent.NONE] * 5

    def test_a_silent_frame_resets_a_partial_onset_run(self) -> None:
        detector = EndpointDetector()
        feed(detector, True, 2)
        assert detector.pending_speech_frames == 2

        assert detector.process(False) is EndpointEvent.NONE
        assert detector.pending_speech_frames == 0

        assert detector.process(True) is EndpointEvent.NONE
        assert detector.process(True) is EndpointEvent.NONE
        assert detector.process(True) is EndpointEvent.SPEECH_STARTED

    def test_silence_while_idle_never_produces_an_event(self) -> None:
        detector = EndpointDetector()
        assert feed(detector, False, 50) == [EndpointEvent.NONE] * 50
        assert detector.active is False

    def test_onset_counts_the_triggering_frames_as_utterance_audio(self) -> None:
        detector = EndpointDetector()
        feed(detector, True, 3)
        assert detector.utterance_duration_ms == 60

    def test_pending_frames_are_only_reported_while_idle(self) -> None:
        detector = EndpointDetector()
        feed(detector, True, 3)
        assert detector.pending_speech_frames == 0

    def test_a_single_frame_onset_can_be_configured(self) -> None:
        detector = EndpointDetector(EndpointConfig(speech_start_ms=20))
        assert detector.process(True) is EndpointEvent.SPEECH_STARTED


class TestUtteranceEnd:
    def test_trailing_silence_ends_the_utterance(self) -> None:
        config = EndpointConfig(speech_start_ms=20, speech_end_ms=100, min_utterance_ms=20)
        detector = EndpointDetector(config)
        assert detector.process(True) is EndpointEvent.SPEECH_STARTED

        assert feed(detector, False, 4) == [EndpointEvent.NONE] * 4
        assert detector.process(False) is EndpointEvent.UTTERANCE_ENDED
        assert detector.state is EndpointState.IDLE
        assert detector.utterance_duration_ms == 0

    def test_speech_resets_the_silence_run(self) -> None:
        config = EndpointConfig(speech_start_ms=20, speech_end_ms=100, min_utterance_ms=20)
        detector = EndpointDetector(config)
        detector.process(True)

        feed(detector, False, 4)
        assert detector.process(True) is EndpointEvent.NONE
        assert feed(detector, False, 4) == [EndpointEvent.NONE] * 4
        assert detector.process(False) is EndpointEvent.UTTERANCE_ENDED

    def test_short_utterances_are_held_open_until_the_minimum(self) -> None:
        config = EndpointConfig(speech_start_ms=20, speech_end_ms=40, min_utterance_ms=200)
        detector = EndpointDetector(config)
        detector.process(True)

        # The silence threshold is met long before the minimum duration is.
        events = feed(detector, False, 7)
        assert events == [EndpointEvent.NONE] * 7
        assert detector.active is True

        assert detector.utterance_duration_ms == 160
        assert detector.process(False) is EndpointEvent.NONE
        assert detector.process(False) is EndpointEvent.UTTERANCE_ENDED

    def test_a_new_utterance_can_start_after_an_end(self) -> None:
        config = EndpointConfig(speech_start_ms=20, speech_end_ms=40, min_utterance_ms=20)
        detector = EndpointDetector(config)
        detector.process(True)
        feed(detector, False, 1)
        assert detector.process(False) is EndpointEvent.UTTERANCE_ENDED
        assert detector.process(True) is EndpointEvent.SPEECH_STARTED


class TestUtteranceLimit:
    def test_the_limit_fires_at_the_exact_maximum(self) -> None:
        config = EndpointConfig(speech_start_ms=20, min_utterance_ms=20, max_utterance_ms=100)
        detector = EndpointDetector(config)
        assert detector.process(True) is EndpointEvent.SPEECH_STARTED

        assert feed(detector, True, 3) == [EndpointEvent.NONE] * 3
        assert detector.utterance_duration_ms == 80
        assert detector.process(True) is EndpointEvent.UTTERANCE_LIMIT

    def test_the_detector_is_reset_after_the_limit(self) -> None:
        config = EndpointConfig(speech_start_ms=20, min_utterance_ms=20, max_utterance_ms=40)
        detector = EndpointDetector(config)
        detector.process(True)
        assert detector.process(True) is EndpointEvent.UTTERANCE_LIMIT
        assert detector.state is EndpointState.IDLE
        assert detector.utterance_duration_ms == 0
        assert detector.process(True) is EndpointEvent.SPEECH_STARTED

    def test_the_limit_takes_precedence_over_a_silence_endpoint(self) -> None:
        config = EndpointConfig(
            speech_start_ms=20,
            speech_end_ms=40,
            min_utterance_ms=20,
            max_utterance_ms=60,
        )
        detector = EndpointDetector(config)
        detector.process(True)
        assert detector.process(False) is EndpointEvent.NONE
        # This frame satisfies both the silence threshold and the hard limit.
        assert detector.process(False) is EndpointEvent.UTTERANCE_LIMIT


class TestExplicitCommit:
    def test_commit_ends_an_active_utterance(self) -> None:
        detector = EndpointDetector()
        feed(detector, True, 3)
        assert detector.commit() is EndpointEvent.UTTERANCE_ENDED
        assert detector.state is EndpointState.IDLE

    def test_commit_ignores_the_minimum_duration(self) -> None:
        config = EndpointConfig(speech_start_ms=20, min_utterance_ms=1_000)
        detector = EndpointDetector(config)
        detector.process(True)
        assert detector.utterance_duration_ms == 20
        assert detector.commit() is EndpointEvent.UTTERANCE_ENDED

    def test_commit_flushes_audio_that_never_reached_the_onset_threshold(self) -> None:
        detector = EndpointDetector()
        feed(detector, True, 2)
        assert detector.active is False
        assert detector.commit() is EndpointEvent.UTTERANCE_ENDED

    def test_commit_without_any_audio_is_a_no_op(self) -> None:
        detector = EndpointDetector()
        assert detector.commit() is EndpointEvent.NONE

    def test_commit_after_silence_only_is_a_no_op(self) -> None:
        detector = EndpointDetector()
        feed(detector, False, 10)
        assert detector.commit() is EndpointEvent.NONE

    def test_a_second_commit_is_a_no_op(self) -> None:
        detector = EndpointDetector()
        feed(detector, True, 3)
        assert detector.commit() is EndpointEvent.UTTERANCE_ENDED
        assert detector.commit() is EndpointEvent.NONE

    def test_reset_clears_all_accumulated_state(self) -> None:
        detector = EndpointDetector()
        feed(detector, True, 4)
        detector.reset()
        assert detector.state is EndpointState.IDLE
        assert detector.utterance_duration_ms == 0
        assert detector.pending_speech_frames == 0
        assert detector.commit() is EndpointEvent.NONE


class TestEventVocabulary:
    def test_the_event_set_is_closed(self) -> None:
        assert [event.value for event in EndpointEvent] == [
            "none",
            "speech_started",
            "utterance_ended",
            "utterance_limit",
        ]

    def test_states_are_serializable_strings(self) -> None:
        assert EndpointState.IDLE.value == "idle"
        assert EndpointState.SPEECH.value == "speech"


class TestPartialCadenceConfiguration:
    def test_defaults_are_the_documented_cadence(self) -> None:
        config = PartialCadenceConfig()
        assert config.initial_ms == 300
        assert config.minimum_ms == 200
        assert config.maximum_ms == 1_200
        assert config.unchanged_growth == pytest.approx(1.5)
        assert config.changed_shrink == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"minimum_ms": 0},
            {"minimum_ms": 400},
            {"maximum_ms": 100},
            {"initial_ms": 100},
        ],
    )
    def test_bounds_must_be_ordered(self, overrides: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="cadence bounds are invalid"):
            PartialCadenceConfig(**overrides)

    @pytest.mark.parametrize("growth", [1.0, 0.9, 0.0])
    def test_growth_must_actually_grow(self, growth: float) -> None:
        with pytest.raises(ValueError, match="unchanged_growth"):
            PartialCadenceConfig(unchanged_growth=growth)

    @pytest.mark.parametrize("shrink", [0.0, -0.5, 1.5])
    def test_shrink_must_be_a_fraction(self, shrink: float) -> None:
        with pytest.raises(ValueError, match="changed_shrink"):
            PartialCadenceConfig(changed_shrink=shrink)


class TestPartialCadence:
    def test_the_first_partial_is_due_at_the_initial_interval(self) -> None:
        cadence = AdaptivePartialCadence()
        assert cadence.interval_ms == 300
        assert cadence.due(280) is False
        assert cadence.due(300) is True
        assert cadence.due(320) is True

    def test_submission_schedules_the_next_partial_one_interval_later(self) -> None:
        cadence = AdaptivePartialCadence()
        cadence.mark_submitted(300)
        assert cadence.due(300) is False
        assert cadence.due(580) is False
        assert cadence.due(600) is True

    def test_unchanged_text_slows_the_cadence_down(self) -> None:
        cadence = AdaptivePartialCadence()
        cadence.mark_submitted(300)
        cadence.observe(changed=False, audio_duration_ms=300)
        assert cadence.interval_ms == 450
        assert cadence.due(700) is False
        assert cadence.due(750) is True

    def test_changed_text_speeds_the_cadence_up(self) -> None:
        cadence = AdaptivePartialCadence(
            PartialCadenceConfig(initial_ms=500, minimum_ms=200, maximum_ms=1_200)
        )
        cadence.mark_submitted(500)
        cadence.observe(changed=True, audio_duration_ms=500)
        assert cadence.interval_ms == 400

    def test_the_interval_never_leaves_its_bounds(self) -> None:
        cadence = AdaptivePartialCadence()
        for step in range(20):
            cadence.observe(changed=False, audio_duration_ms=300 + step * 100)
        assert cadence.interval_ms == 1_200

        for step in range(40):
            cadence.observe(changed=True, audio_duration_ms=2_300 + step * 100)
        assert cadence.interval_ms == 200

    def test_the_next_due_time_never_moves_earlier(self) -> None:
        cadence = AdaptivePartialCadence()
        cadence.mark_submitted(1_000)
        assert cadence.due(1_200) is False

        cadence.observe(changed=True, audio_duration_ms=1_000)
        assert cadence.due(1_200) is False
        assert cadence.due(1_300) is True

    def test_cadence_depends_only_on_audio_time(self) -> None:
        first = AdaptivePartialCadence()
        second = AdaptivePartialCadence()
        for audio_ms in (300, 600, 900, 1_200):
            first.mark_submitted(audio_ms)
            first.observe(changed=True, audio_duration_ms=audio_ms)
            second.mark_submitted(audio_ms)
            second.observe(changed=True, audio_duration_ms=audio_ms)
        assert first.interval_ms == second.interval_ms
        assert first.due(5_000) == second.due(5_000)

    def test_reset_restores_the_initial_schedule(self) -> None:
        cadence = AdaptivePartialCadence()
        cadence.mark_submitted(1_000)
        cadence.observe(changed=False, audio_duration_ms=1_000)
        cadence.reset()
        assert cadence.interval_ms == 300
        assert cadence.due(300) is True

    def test_frame_aligned_audio_time_reaches_the_first_partial(self) -> None:
        cadence = AdaptivePartialCadence()
        audio_ms = 0
        due_at = None
        for _ in range(20):
            audio_ms += FRAME_DURATION_MS
            if cadence.due(audio_ms):
                due_at = audio_ms
                break
        assert due_at == 300

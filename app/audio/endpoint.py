"""Streaming utterance endpointing and adaptive partial cadence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.audio.pcm import FRAME_DURATION_MS


class EndpointState(StrEnum):
    IDLE = "idle"
    SPEECH = "speech"


class EndpointEvent(StrEnum):
    NONE = "none"
    SPEECH_STARTED = "speech_started"
    UTTERANCE_ENDED = "utterance_ended"
    UTTERANCE_LIMIT = "utterance_limit"


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    speech_start_ms: int = 60
    speech_end_ms: int = 600
    min_utterance_ms: int = 200
    max_utterance_ms: int = 30_000

    def __post_init__(self) -> None:
        values = (
            self.speech_start_ms,
            self.speech_end_ms,
            self.min_utterance_ms,
            self.max_utterance_ms,
        )
        if any(value <= 0 or value % FRAME_DURATION_MS for value in values):
            raise ValueError("endpoint durations must be positive multiples of 20 ms")
        if self.min_utterance_ms > self.max_utterance_ms:
            raise ValueError("min_utterance_ms must not exceed max_utterance_ms")


class EndpointDetector:
    """Debounce frame VAD decisions into deterministic utterance events."""

    __slots__ = (
        "config",
        "state",
        "_speech_run",
        "_silence_run",
        "_utterance_frames",
    )

    def __init__(self, config: EndpointConfig | None = None) -> None:
        self.config = config or EndpointConfig()
        self.state = EndpointState.IDLE
        self._speech_run = 0
        self._silence_run = 0
        self._utterance_frames = 0

    @property
    def active(self) -> bool:
        return self.state is EndpointState.SPEECH

    @property
    def pending_speech_frames(self) -> int:
        return self._speech_run if self.state is EndpointState.IDLE else 0

    @property
    def utterance_duration_ms(self) -> int:
        return self._utterance_frames * FRAME_DURATION_MS

    def process(self, is_speech: bool) -> EndpointEvent:
        if self.state is EndpointState.IDLE:
            if not is_speech:
                self._speech_run = 0
                return EndpointEvent.NONE
            self._speech_run += 1
            if self._speech_run * FRAME_DURATION_MS < self.config.speech_start_ms:
                return EndpointEvent.NONE
            self.state = EndpointState.SPEECH
            self._utterance_frames = self._speech_run
            self._silence_run = 0
            self._speech_run = 0
            return EndpointEvent.SPEECH_STARTED

        self._utterance_frames += 1
        self._silence_run = 0 if is_speech else self._silence_run + 1
        if self.utterance_duration_ms >= self.config.max_utterance_ms:
            self.reset()
            return EndpointEvent.UTTERANCE_LIMIT
        if (
            self.utterance_duration_ms >= self.config.min_utterance_ms
            and self._silence_run * FRAME_DURATION_MS >= self.config.speech_end_ms
        ):
            self.reset()
            return EndpointEvent.UTTERANCE_ENDED
        return EndpointEvent.NONE

    def commit(self) -> EndpointEvent:
        """Force an endpoint for explicit input.commit, regardless of VAD state."""

        had_audio = self.active or self._speech_run > 0
        self.reset()
        return EndpointEvent.UTTERANCE_ENDED if had_audio else EndpointEvent.NONE

    def reset(self) -> None:
        self.state = EndpointState.IDLE
        self._speech_run = 0
        self._silence_run = 0
        self._utterance_frames = 0


@dataclass(frozen=True, slots=True)
class PartialCadenceConfig:
    initial_ms: int = 300
    minimum_ms: int = 200
    maximum_ms: int = 1_200
    unchanged_growth: float = 1.5
    changed_shrink: float = 0.8

    def __post_init__(self) -> None:
        if not 0 < self.minimum_ms <= self.initial_ms <= self.maximum_ms:
            raise ValueError("partial cadence bounds are invalid")
        if self.unchanged_growth <= 1.0:
            raise ValueError("unchanged_growth must be greater than one")
        if not 0.0 < self.changed_shrink <= 1.0:
            raise ValueError("changed_shrink must be in (0, 1]")


class AdaptivePartialCadence:
    """Use audio time, not wall-clock jitter, to schedule rolling partials."""

    __slots__ = ("config", "_interval_ms", "_next_audio_ms")

    def __init__(self, config: PartialCadenceConfig | None = None) -> None:
        self.config = config or PartialCadenceConfig()
        self._interval_ms = self.config.initial_ms
        self._next_audio_ms = self.config.initial_ms

    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    def due(self, audio_duration_ms: int) -> bool:
        return audio_duration_ms >= self._next_audio_ms

    def mark_submitted(self, audio_duration_ms: int) -> None:
        self._next_audio_ms = audio_duration_ms + self._interval_ms

    def observe(self, changed: bool, audio_duration_ms: int) -> None:
        factor = self.config.changed_shrink if changed else self.config.unchanged_growth
        self._interval_ms = min(
            self.config.maximum_ms,
            max(self.config.minimum_ms, round(self._interval_ms * factor)),
        )
        self._next_audio_ms = max(
            self._next_audio_ms,
            audio_duration_ms + self._interval_ms,
        )

    def reset(self) -> None:
        self._interval_ms = self.config.initial_ms
        self._next_audio_ms = self.config.initial_ms

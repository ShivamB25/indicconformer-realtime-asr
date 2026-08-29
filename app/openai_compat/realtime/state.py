"""Mutable realtime session state and immutable committed audio turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import soxr  # type: ignore[import-untyped]

from app.core.types import LanguageCode
from app.openai_compat.realtime.schemas import (
    PCMFormat,
    ServerVAD,
    SessionAudio,
    SessionAudioInput,
    SessionTranscription,
    SessionUpdate,
    TranscriptionSession,
)

_OPENAI_SAMPLE_RATE: Literal[24_000] = 24_000
_BYTES_PER_MILLISECOND = _OPENAI_SAMPLE_RATE * 2 // 1_000
_VAD_FRAME_BYTES = _OPENAI_SAMPLE_RATE * 2 * 20 // 1_000


@dataclass(frozen=True, slots=True)
class TurnSnapshot:
    """An inference input detached from subsequent buffer/session mutations."""

    item_id: str
    previous_item_id: str | None
    client_event_id: str | None
    pcm16: bytes
    language: LanguageCode

    def to_engine_audio(self) -> np.ndarray:
        samples = np.frombuffer(self.pcm16, dtype="<i2").astype(np.float32)
        samples /= np.float32(32_768.0)
        resampled = soxr.resample(samples, _OPENAI_SAMPLE_RATE, 16_000, quality="HQ")
        return np.ascontiguousarray(resampled, dtype=np.float32)


@dataclass(slots=True)
class RealtimeSessionState:
    """Mutable connection state; committed turns leave as immutable snapshots."""

    session_id: str
    expires_at: int
    max_audio_bytes: int
    max_audio_seconds: float
    model: str
    language: LanguageCode | None = None
    turn_detection: ServerVAD | None = field(default_factory=lambda: ServerVAD(type="server_vad"))
    audio: bytearray = field(default_factory=bytearray)
    total_audio_bytes: int = 0
    buffer_start_bytes: int = 0
    previous_item_id: str | None = None
    pending_item_id: str | None = None
    vad_scan_bytes: int = 0
    speech_run_frames: int = 0
    silence_run_frames: int = 0
    speech_active: bool = False

    @property
    def configured(self) -> bool:
        return self.language is not None

    def apply_update(self, update: SessionUpdate, canonical_model: str | None) -> None:
        """Merge a session.update patch, then validate and publish it atomically."""

        if self.audio:
            raise ValueError("session cannot be updated while the input audio buffer is nonempty")
        model = self.model
        language = self.language
        turn_detection = self.turn_detection
        audio_patch = update.audio
        input_patch = None if audio_patch is None else audio_patch.input
        if input_patch is not None:
            transcription = input_patch.transcription
            if transcription is not None:
                if "model" in transcription.model_fields_set:
                    if canonical_model is None:
                        raise ValueError("transcription model cannot be null")
                    model = canonical_model
                if "languages" in transcription.model_fields_set:
                    language = (
                        None if transcription.languages is None else transcription.languages[0]
                    )
                elif "language" in transcription.model_fields_set:
                    language = transcription.language
            if "turn_detection" in input_patch.model_fields_set:
                turn_detection = input_patch.turn_detection
        if language is None:
            raise ValueError("exactly one transcription language is required")
        self.model = model
        self.language = language
        self.turn_detection = turn_detection
        self.reset_vad()

    def append(self, payload: bytes) -> None:
        prospective_bytes = self.total_audio_bytes + len(payload)
        if prospective_bytes > self.max_audio_bytes:
            raise OverflowError("session audio exceeds the configured byte limit")
        if prospective_bytes / (_OPENAI_SAMPLE_RATE * 2) > self.max_audio_seconds:
            raise OverflowError("session audio exceeds the configured duration limit")
        self.audio.extend(payload)
        self.total_audio_bytes = prospective_bytes

    def clear(self) -> None:
        self.audio.clear()
        self.buffer_start_bytes = self.total_audio_bytes
        self.pending_item_id = None
        self.reset_vad()

    def reset_vad(self) -> None:
        self.vad_scan_bytes = 0
        self.speech_run_frames = 0
        self.silence_run_frames = 0
        self.speech_active = False

    def next_vad_frame(self) -> tuple[bytes, int] | None:
        """Copy the next complete 20 ms frame and return its absolute end time."""

        end = self.vad_scan_bytes + _VAD_FRAME_BYTES
        if end > len(self.audio):
            return None
        frame = bytes(self.audio[self.vad_scan_bytes : end])
        self.vad_scan_bytes = end
        absolute_end_ms = (self.buffer_start_bytes + end) * 1_000 // (_OPENAI_SAMPLE_RATE * 2)
        return frame, absolute_end_ms

    def trim_pre_speech(self, prefix_padding_ms: int) -> None:
        """Keep only bounded prefix plus speech-start debounce audio before a turn."""

        retained = (
            prefix_padding_ms * _BYTES_PER_MILLISECOND + self.speech_run_frames * _VAD_FRAME_BYTES
        )
        discard = max(0, self.vad_scan_bytes - retained)
        if discard == 0:
            return
        del self.audio[:discard]
        self.buffer_start_bytes += discard
        self.vad_scan_bytes -= discard

    def snapshot(
        self, *, item_id: str, client_event_id: str | None, byte_count: int | None = None
    ) -> TurnSnapshot:
        if self.language is None:
            raise ValueError("session.update with exactly one language is required before audio")
        size = len(self.audio) if byte_count is None else byte_count
        if size <= 0:
            raise ValueError("input audio buffer is empty")
        if size > len(self.audio):
            raise ValueError("input audio buffer commit exceeds buffered audio")
        if size % 2:
            raise ValueError("PCM16 audio must contain complete samples")
        pcm = bytes(self.audio[:size])
        del self.audio[:size]
        self.buffer_start_bytes += size
        snapshot = TurnSnapshot(
            item_id=item_id,
            previous_item_id=self.previous_item_id,
            client_event_id=client_event_id,
            pcm16=pcm,
            language=self.language,
        )
        self.previous_item_id = item_id
        self.pending_item_id = None
        self.reset_vad()
        return snapshot

    def session_payload(self) -> TranscriptionSession:
        languages = [] if self.language is None else [self.language]
        return TranscriptionSession(
            id=self.session_id,
            expires_at=self.expires_at,
            audio=SessionAudio(
                input=SessionAudioInput(
                    format=PCMFormat(type="audio/pcm", rate=_OPENAI_SAMPLE_RATE),
                    transcription=SessionTranscription(model=self.model, languages=languages),
                    turn_detection=self.turn_detection,
                )
            ),
        )

"""Deterministic synthetic PCM helpers.

Nothing here is speech and nothing here claims to be. Frames are built from
exact arithmetic so that energy-based assertions are reproducible bit for bit:

* ``int16_frame(amplitude)`` alternates ``+amplitude``/``-amplitude``, so its
  normalized RMS is exactly ``amplitude / 32768``.
* ``float_frame(level)`` is constant, so its RMS is exactly ``level``.

No randomness, no audio files, no sample rates other than 16 kHz.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.audio.pcm import (
    BYTES_PER_SAMPLE,
    FRAME_DURATION_MS,
    PCM16_FRAME_BYTES,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)

PCM16_DTYPE = np.dtype("<i2")
INT16_PEAK = 32_768.0

# Comfortably above and below EnergyVADConfig.speech_threshold (0.015).
SPEECH_AMPLITUDE = 6_000
QUIET_AMPLITUDE = 64


def frames_for_ms(duration_ms: int) -> int:
    """Return the number of whole 20 ms frames in ``duration_ms``."""

    if duration_ms < 0 or duration_ms % FRAME_DURATION_MS:
        raise ValueError("duration_ms must be a non-negative multiple of 20 ms")
    return duration_ms // FRAME_DURATION_MS


def int16_frame(amplitude: int = SPEECH_AMPLITUDE) -> NDArray[np.int16]:
    """One 20 ms frame whose normalized RMS is exactly ``amplitude / 32768``."""

    if not 0 <= amplitude <= 32_767:
        raise ValueError("amplitude must fit in non-negative int16 range")
    samples = np.empty(SAMPLES_PER_FRAME, dtype=PCM16_DTYPE)
    samples[0::2] = amplitude
    samples[1::2] = -amplitude
    return samples


def float_frame(level: float) -> NDArray[np.float32]:
    """One 20 ms float32 frame whose RMS is exactly ``level``."""

    return np.full(SAMPLES_PER_FRAME, level, dtype=np.float32)


def frame_bytes(samples: NDArray[np.int16]) -> bytes:
    """Serialize int16 samples as little-endian PCM16 wire bytes."""

    return samples.astype(PCM16_DTYPE, copy=False).tobytes()


def speech_frame() -> bytes:
    """One 20 ms wire frame that the energy VAD classifies as speech."""

    return frame_bytes(int16_frame(SPEECH_AMPLITUDE))


def silence_frame() -> bytes:
    """One 20 ms wire frame of pure digital silence."""

    return frame_bytes(np.zeros(SAMPLES_PER_FRAME, dtype=PCM16_DTYPE))


def speech_frames(duration_ms: int) -> list[bytes]:
    """A list of identical speech frames covering ``duration_ms``."""

    return [speech_frame() for _ in range(frames_for_ms(duration_ms))]


def silence_frames(duration_ms: int) -> list[bytes]:
    """A list of silent frames covering ``duration_ms``."""

    return [silence_frame() for _ in range(frames_for_ms(duration_ms))]


def utterance_frames(speech_ms: int, trailing_silence_ms: int) -> list[bytes]:
    """Speech followed by trailing silence, as separate 20 ms wire frames."""

    return speech_frames(speech_ms) + silence_frames(trailing_silence_ms)


def float_audio(duration_ms: int, level: float = 0.25) -> NDArray[np.float32]:
    """Constant float32 mono audio of an exact duration, for engine requests."""

    if duration_ms < 0:
        raise ValueError("duration_ms must not be negative")
    sample_count = duration_ms * SAMPLE_RATE // 1_000
    return np.full(sample_count, level, dtype=np.float32)


def wire_bytes_for_ms(duration_ms: int, amplitude: int = SPEECH_AMPLITUDE) -> bytes:
    """A single contiguous PCM16 byte string spanning whole frames."""

    frame = frame_bytes(int16_frame(amplitude))
    return frame * frames_for_ms(duration_ms)


__all__ = [
    "BYTES_PER_SAMPLE",
    "FRAME_DURATION_MS",
    "INT16_PEAK",
    "PCM16_DTYPE",
    "PCM16_FRAME_BYTES",
    "QUIET_AMPLITUDE",
    "SAMPLES_PER_FRAME",
    "SAMPLE_RATE",
    "SPEECH_AMPLITUDE",
    "float_audio",
    "float_frame",
    "frame_bytes",
    "frames_for_ms",
    "int16_frame",
    "silence_frame",
    "silence_frames",
    "speech_frame",
    "speech_frames",
    "utterance_frames",
    "wire_bytes_for_ms",
]

"""Bounded decoding for containerized OpenAI transcription uploads."""

from __future__ import annotations

import io

import av
import numpy as np
from numpy.typing import NDArray

TARGET_SAMPLE_RATE = 16_000


class InvalidAudioError(ValueError):
    """The upload is not a decodable, non-empty audio container."""


class AudioDurationExceeded(ValueError):
    """Decoded audio would exceed the configured duration bound."""


def decode_audio_file(payload: bytes, *, max_audio_seconds: int) -> NDArray[np.float32]:
    """Decode, downmix, and resample an upload to bounded mono 16 kHz float32."""

    if not payload:
        raise InvalidAudioError("The uploaded audio file is empty")
    if max_audio_seconds <= 0:
        raise ValueError("max_audio_seconds must be positive")

    max_samples = max_audio_seconds * TARGET_SAMPLE_RATE
    chunks: list[NDArray[np.float32]] = []
    sample_count = 0

    def append_frames(frames: list[av.AudioFrame]) -> None:
        nonlocal sample_count
        for frame in frames:
            values = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
            if values.size > max_samples - sample_count:
                raise AudioDurationExceeded(
                    "The audio file exceeds the configured maximum duration"
                )
            if values.size:
                chunks.append(values)
                sample_count += values.size

    try:
        with av.open(io.BytesIO(payload), mode="r") as container:
            if not container.streams.audio:
                raise InvalidAudioError("The uploaded file does not contain an audio stream")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format="fltp",
                layout="mono",
                rate=TARGET_SAMPLE_RATE,
            )
            for frame in container.decode(stream):
                append_frames(resampler.resample(frame))
            append_frames(resampler.resample(None))
    except (InvalidAudioError, AudioDurationExceeded):
        raise
    except Exception as exc:
        raise InvalidAudioError("The uploaded audio file could not be decoded") from exc

    if not chunks:
        raise InvalidAudioError("The uploaded audio file contains no audio samples")
    waveform = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise InvalidAudioError("The uploaded audio file contains invalid samples")
    return waveform
